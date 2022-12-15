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
    def __init__(self, code=1024, depth=4, length=512, model_dim=256, model_qk_dim=64, model_enc_layers = 6, causal = False, chunk = 1024, steps=50, batch=4, initial_lr=5e-4,**kwargs):
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
        attn_dim_qk = model_qk_dim,            # dimension of queries / keys in attention
        attn_dim_value = model_dim*2,        # dimensino of values in attention
        laplacian_attn_fn = True,    # whether to use softmax (false) or laplacian attention activation fn (true)
        causal=causal,
        codebook_size = depth,
        chunk_size = chunk)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1,ignore_index=code*depth)
        self.lr = initial_lr
        self.batch = batch

    def forward(self, x, step = None,nmask=None):
      #t = torch.randint(1,self.sampler.steps+1,(1,)).item() if step is None else step
      #x = self.denoiser.preprocess(x)
      #if nmask is None:
      #      nmask = torch.randint_like(x,self.sampler.token_size)
      #x = x.view(x.size(0), -1)
      #x_masknoised, targets = self.sampler.add_noise(x,t,nmask)
      x_pred = self.denoiser(x)
      return x_pred
    
    def sample(self, x_pred_prob):
        dist = torch.distributions.Categorical(x_pred_prob)
        return dist.sample()

    def configure_optimizers(self):
      return torch.optim.AdamW(self.parameters(), lr=self.lr, eps=3e-7, weight_decay=1e-3)

    def training_step(self, train_batch, batch_idx):
      tmax = self.global_step // 1000 + 1
      x = train_batch
      t = torch.randint(1,min(tmax,self.sampler.steps-1)+1,(1,)).item()
      x = self.denoiser.preprocess(x)
      nmask = torch.randint_like(x,self.sampler.token_size)
      x = x.view(x.size(0), -1)
      x_masknoised, targets = self.sampler.add_noise(x,t,nmask)
      x_pred = self.denoiser(x_masknoised).permute(0,2,1)
      loss = self.criterion(x_pred,x)
      self.log('train_loss', loss)
      return loss

    def validation_step(self, val_batch, batch_idx):
      tmax = self.global_step // 1000 + 1
      x = val_batch
      t = torch.randint(1,min(tmax,self.sampler.steps-1)+1,(1,)).clip(max=tmax).item()
      x = self.denoiser.preprocess(x)
      nmask = torch.randint_like(x,self.sampler.token_size)
      x = x.view(x.size(0), -1)
      x_masknoised, targets = self.sampler.add_noise(x,t,nmask)
      x_pred = self.denoiser(x_masknoised).permute(0,2,1)
      loss = self.criterion(x_pred,x)
      self.log('val_loss', loss)
      return loss
