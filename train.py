import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from deepspeed.ops.adam import DeepSpeedCPUAdam
from sampler import Sampler
from mega_transformer import Mega
from data_utils import *

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, codebook_size=1024, codebook_num=8, length_per_codebook=512, model_dim=128, model_depth = 6, num_chunks = 2, steps=50, batch_size=4, lr=1e-4):
        super().__init__()
        token_size = codebook_size
        max_len = length_per_codebook*codebook_num
        self.token_size = token_size
        self.max_len = max_len
        self.sampler = Sampler(token_size=token_size*codebook_num,max_len=length_per_codebook*codebook_num,steps=steps,bn=batch_size)
        self.denoiser =  Mega(
        num_tokens = token_size,       # number of tokens
        dim = model_dim,                   # model dimensions
        depth = model_depth,                   # depth
        ema_heads = 16,              # number of EMA heads
        attn_dim_qk = model_dim // 2,            # dimension of queries / keys in attention
        attn_dim_value = model_dim*2,        # dimensino of values in attention
        laplacian_attn_fn = True,    # whether to use softmax (false) or laplacian attention activation fn (true)
        causal=False,
        codebook_size = codebook_num,
        chunk_size = max_len // num_chunks)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lr = lr

    def forward(self, x, step = None):
      t = torch.randint(1,self.sampler.steps+1,(1,)).item() if step is None else step
      x = self.denoiser.preprocess(x)
      nmask = torch.zeros_like(x) * self.sampler.mask_id
      x = x.view(x.size(0), -1)
      x_masked = self.sampler.add_noise(x,t,nmask)
      x_pred = self.denoiser(x)
      x_pred = x_pred.permute(0,2,1)
      loss = self.criterion(x_pred,x)
      return t,nmask,x,x_masked,x_pred,loss

    def configure_optimizers(self):
      #return torch.optim.AdamW(self.parameters(), lr=self.lr)
      return DeepSpeedCPUAdam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
      x = train_batch.tile((batch_size,1,1))
      t = torch.randint(1,self.sampler.steps+1,(1,))
      x = self.denoiser.preprocess(x)
      print(t)
      print(x[:,:32])
      nmask = torch.zeros_like(x) * self.sampler.mask_id
      x = x.view(x.size(0), -1)
      x_masked = self.sampler.add_noise(x,t,nmask)
      print(x_masked[:,:32])
      x_pred = self.denoiser(x_masked).permute(0,2,1)
      print(x_pred[:,:,:32])
      loss = self.criterion(x_pred,x)
      self.log('train_loss', loss)
      return loss

    def validation_step(self, val_batch, batch_idx):
      x = val_batch.tile((batch_size,1,1))
      t = torch.randint(1,self.sampler.steps+1,(1,))
      x = self.denoiser.preprocess(x)
      nmask = torch.zeros_like(x) * self.sampler.mask_id
      x = x.view(x.size(0), -1)
      x_masked = self.sampler.add_noise(x,t,nmask)
      x_pred = self.denoiser(x_masked).permute(0,2,1)
      loss = self.criterion(x_pred,x)
      self.log('val_loss', loss)
      return loss

# data
#dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
#mnist_train, mnist_val = random_split(dataset, [55000, 5000])

#train_loader = DataLoader(mnist_train, batch_size=32)
#val_loader = DataLoader(mnist_val, batch_size=32)

codebook_size=1024
codebook_num=8
length=1024
batch_size=4
model_dim = 80
model_depth = 1
chunk_size = 256
num_chunks = length*codebook_num // chunk_size

# model
model = LitAutoEncoder(codebook_size=codebook_size,
codebook_num=codebook_num,
length_per_codebook=length,
batch_size=batch_size,
model_dim = model_dim,
model_depth = model_depth,
num_chunks = num_chunks)

with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    loss = model.denoiser(torch.randint(0,codebook_num*codebook_size+1,(1,length*codebook_num)))
    print(loss)

quit()

data_list = getData('/content/drive/MyDrive/wave_encodec/',length)
d = EnCodecData(data_list)
print(len(data_list))

train, val = torch.utils.data.random_split(d, [int(len(d)*0.9),len(d)-int(len(d)*0.9)])
train_loader = DataLoader(train, batch_size=1, shuffle=True)
val_loader = DataLoader(val, batch_size=1, shuffle=False)

# training
trainer = Trainer(
    accelerator='gpu',
    devices=1,
    strategy='deepspeed_stage_2_offload',
    precision=16,
    max_epochs=1,
    gradient_clip_val=0.5,
    auto_lr_find=False,
    overfit_batches=10,
    track_grad_norm=2,
    log_every_n_steps=1
)



trainer.fit(model, train_loader, val_loader)
#trainer.tune(model,train_loader)
