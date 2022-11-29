import math, torch
from torch import nn
from einops import rearrange
import numpy

class Sampler(nn.Module):
  def __init__(self,steps=50,token_size=1024,max_len=1024, bn=16):
      super().__init__()
      self.steps = steps
      self.token_size = token_size+1
      self.batch_size = bn
      self.L = max_len
      self.mask_id = token_size
      self.beta = 0.9
      self.register_buffer('ind_num', (self.gamma(torch.arange(0,51))*self.L).round().long().clamp_min(8))
      self.register_buffer('mask',torch.ones(self.batch_size,self.L).long() * self.mask_id)
  def sample_t(self,t=None) -> torch.Tensor:
      if t == -1:
        return torch.randint(1,self.steps+1,(self.batch_size,))
      #t=1 means right after start (t=0 is all unmasked)
      return torch.randint(1,t+1,(self.batch_size,))

  def sample(self,x:torch.Tensor,step:int):
      t = self.sample_t(step).type(torch.FloatTensor)/self.steps
      x,x_orig = self.add_noise(x,t)
      return x,x_orig,t

  def gamma(self,r:torch.Tensor) -> torch.Tensor:
      return (r * torch.pi / 2 /self.steps).cos()

  @torch.no_grad()
  def add_noise(self, x, t, noise_mask=None):
    if len(x.size()) == 3:
        x = rearrange(x,'b q l -> b (q l)')
    indices_to_pick = self.ind_num[t].item()
    noised_indices = int((1 - self.beta)*indices_to_pick)
    mask = self.mask
    if noise_mask is None:
        noise_mask = torch.randint(0,self.token_size,(self.batch_size,self.L)).to(x)
    idx = torch.sort(torch.randint(
        0, self.L - indices_to_pick, (self.batch_size, indices_to_pick)
    ), axis=1).values + torch.arange(0, indices_to_pick).reshape(1, -1)
    idx = idx.to(x)
    x = torch.scatter(x,dim=-1,index=idx, src=mask)
    if noised_indices > 0:
        idx2 = torch.multinomial(torch.ones(self.batch_size, indices_to_pick),noised_indices)
        idx2 = idx2.to(x)
        x = torch.scatter(x,dim=-1,index=idx[torch.arange(len(idx)).unsqueeze(-1),idx2],src=noise_mask)
    return x
  
  def __call__(self, x: torch.Tensor, step: int):
    return self.sample(x,step)

