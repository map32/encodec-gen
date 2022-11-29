import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sampler import Sampler
from mega_transformer import Mega
from data_utils import *
import pickle
import argparse

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, code=1024, depth=8, length=512, model_dim=128, model_enc_layers = 6, chunk = 512, steps=50, batch=4, initial_lr=5e-4):
        super().__init__()
        codebook_size=code
        codebook_num=depth
        length_per_codebook=length
        model_depth=model_enc_layers
        token_size = codebook_size
        max_len = length*depth
        self.code_size = code
        self.max_len = max_len
        self.sampler = Sampler(token_size=code*depth,max_len=max_len,steps=steps,bn=batch)
        self.denoiser =  Mega(
        num_tokens = code,       # number of tokens
        dim = model_dim,                   # model dimensions
        depth = model_enc_layers,                   # depth
        ema_heads = 16,              # number of EMA heads
        attn_dim_qk = model_dim // 2,            # dimension of queries / keys in attention
        attn_dim_value = model_dim*2,        # dimensino of values in attention
        laplacian_attn_fn = True,    # whether to use softmax (false) or laplacian attention activation fn (true)
        causal=causal,
        codebook_size = depth,
        chunk_size = chunk)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lr = initial_lr
        self.batch = batch

    def forward(self, x, step = None,nmask=None):
      t = torch.randint(1,self.sampler.steps+1,(1,)).item() if step is None else step
      x = self.denoiser.preprocess(x)
      if nmask is None:
            nmask = torch.randint_like(x,self.sampler.token_size-1)
      x = x.view(x.size(0), -1)
      x_masked = self.sampler.add_noise(x,t,nmask)
      x_pred = self.denoiser(x_masked)
      x_pred = x_pred.permute(0,2,1)
      loss = self.criterion(x_pred,x)
      return x,x_masked,loss

    def configure_optimizers(self):
      return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
      x = train_batch
      t = torch.randint(1,self.sampler.steps+1,(1,)).item()
      x = self.denoiser.preprocess(x)
      nmask = torch.randint_like(x,self.sampler.token_size-1)
      x = x.view(x.size(0), -1)
      x_masked = self.sampler.add_noise(x,t,nmask)
      x_pred = self.denoiser(x_masked).permute(0,2,1)
      loss = self.criterion(x_pred,x)
      self.log('train_loss', loss)
      return loss

    def validation_step(self, val_batch, batch_idx):
      x = val_batch
      t = torch.randint(1,self.sampler.steps+1,(1,)).item()
      x = self.denoiser.preprocess(x)
      nmask = torch.randint_like(x,self.sampler.token_size-1)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'encodec-gen',
        description = 'What the program does',
        epilog = 'Text at the bottom of help')
    
    parser.add_argument('filepath',required=True)
    parser.add_argument('-c','--code',default=1024)
    parser.add_argument('-d','--depth',default=8)
    parser.add_argument('-l','--length',default=1024)
    parser.add_argument('-b','--batch',default=4)
    parser.add_argument('-m','--model_dim',default=128)
    parser.add_argument('-ml','--model_enc_layers',default=6)
    parser.add_argument('-ch','--chunk',default=1024)
    parser.add_argument('-ca','--causal',default=False)
    parser.add_argument('-lr','--initial_lr',default=3e-4)
    parser.add_argument('-ck','--checkpoint_steps',default=1000)
    parser.add_argument('-w','--warmup_steps',default=5000)
    parser.add_argument('-s','--save_path',default='./models')
    args = parser.parse_args()
    assert args.length*args.depth % args.chunk_size == 0

    # model
    model = LitAutoEncoder(**args)
    
    cached_path = args.filepath
    try:
        datalist = torch.load(cached_path)
    except:
        print(cached_path+' does not exist.')
        break
    
    transforms = data_augment(args.length,args.depth)
    
    dataset = EnCodecData(datalist,transforms=transforms)
    
    train, val = torch.utils.data.random_split(d, [int(len(d)*0.95),len(d)-int(len(d)*0.95)])
    train_loader = DataLoader(train, sampler=RepeatingSampler(train,batch_size,shuffle=True))
    val_loader = DataLoader(val, sampler=RepeatingSampler(val`,batch_size,shuffle=False))
    
    callbacks=[ModelCheckpoint(dirpath=args.save_path,monitor='train_loss',mode='min',every_n_train_steps=args.checkpoint_steps)]
    
    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        precision=16,
        max_epochs=1000,
        callbacks=callbacks
    )



    trainer.fit(model, train_loader, val_loader)
    

