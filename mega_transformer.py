from os import X_OK
from torch.nn.init import xavier_uniform_
import math,torch
from functools import partial
from torch import nn, einsum, rand_like
import torch.nn.functional as F
from torch.fft import rfft, irfft
from einops import rearrange
from scipy.fftpack import next_fast_len
from positional_encodings.torch_encodings import PositionalEncodingPermute1D,PositionalEncoding2D,PositionalEncoding1D, Summer
import numpy as np
import argparse

# functions

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def default(val, d):
    return val if exists(val) else d

def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

def conv1d_fft(x, weights, dim = -2, weight_dim = -1):
    # O(N log(N)) 1d convolution using some fourier trick

    assert weight_dim >= dim

    N = x.shape[dim]
    M = weights.shape[weight_dim]

    fast_len = next_fast_len(N + M - 1)

    f_x = rfft(x, n = fast_len, dim = dim)
    f_weight = rfft(weights, n = fast_len, dim = weight_dim)

    f_v_weight = f_x * append_dims(f_weight.conj(), weight_dim - dim)
    out = irfft(f_v_weight, fast_len, dim = dim)
    out = out.roll(-1, dims = (dim,))

    indices = torch.arange(start = fast_len - N, end = fast_len, dtype = torch.long, device = x.device)
    out = out.index_select(dim, indices)
    return out

# positional bias for single-headed attention

class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale

# classes

class LaplacianAttnFn(nn.Module):
    def forward(self, x):
        mu = math.sqrt(0.5)
        std = math.sqrt(0.25 * math.pi)
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

class SingleHeadedAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_qk,
        dim_value,
        causal = False,
        laplacian_attn_fn = False,
        chunk_size = -1
    ):
        super().__init__()
        self.causal = causal
        self.laplacian_attn_fn = laplacian_attn_fn

        self.chunk_size = chunk_size

        self.attn_fn = partial(F.softmax, dim = -1) if not laplacian_attn_fn else LaplacianAttnFn()

        self.rel_pos_bias = T5RelativePositionBias(causal = causal, scale = dim_qk ** 0.5)

        self.to_qk = nn.Sequential(
            nn.Linear(dim, dim_qk),
            nn.SiLU()
        )

        self.offsetscale = OffsetScale(dim_qk, heads = 2)

        self.to_v = nn.Sequential(
            nn.Linear(dim, dim_value),
            nn.SiLU()
        )

    def forward(self, x, v_input = None):
        seq_len, dim, device, dtype = *x.shape[-2:], x.device, x.dtype
        # x, v_input: (B, L, D)
        v_input = default(v_input, x)
        chunks = 1
        if self.chunk_size > 0:
          assert seq_len % self.chunk_size == 0
          chunks = x.shape[-2] // self.chunk_size
          x = rearrange(x,'b (k c) d -> b k c d',k=chunks,c=self.chunk_size)
          if v_input.shape != x.shape:
            assert v_input.shape[-2] % self.chunk_size == 0
            v_input = rearrange(v_input,'b (k c) d -> b k c d',k=chunks,c=self.chunk_size)
        else:
          x = x.unsqueeze(-3)
          v_input = v_input.unsqueeze(-3)

        qk, v = self.to_qk(x), self.to_v(v_input)
        q, k = self.offsetscale(qk)

        scale = (seq_len ** -1) if self.laplacian_attn_fn else (dim ** -0.5)
        if self.chunk_size > 0:
          assert(q.shape[-3:-1] == (chunks,self.chunk_size))
          assert(k.shape[-3:-1] == (chunks,self.chunk_size))
        # (B, K, C, D) x (B, K, C, D) -> (B, K, C, C)
        # Lq = Lk
        sim = einsum('... i d, ... j d -> ... i j', q, k) * scale

        sim = sim + self.rel_pos_bias(sim)

        if self.causal:
            seq_ = seq_len if self.chunk_size < 1 else x.shape[-2] // self.chunk_size
            causal_mask = torch.ones((seq_, seq_), device = device, dtype = torch.bool).triu(1)

        if self.causal and not self.laplacian_attn_fn:
            # is softmax attention and using large negative value pre-softmax
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = self.attn_fn(sim)

        if self.causal and self.laplacian_attn_fn:
            # if using laplacian attention function, zero out upper triangular with 0s
            attn = attn.masked_fill(causal_mask, 0.)

        # (B, K, C, C) x (B, K, C, D) -> (B, K, C, D)
        res = einsum('... i j, ... j d -> ... i d', attn, v)
        res = res.reshape(*res.shape[:-3],res.shape[-3]*res.shape[-2],res.shape[-1])
        return res

class MultiHeadedEMA(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        bidirectional = False,
        dim_head = None
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.expansion = nn.Parameter(torch.randn(heads * (2 if bidirectional else 1), dim))
        self.reduction = nn.Parameter(torch.randn(heads * (2 if bidirectional else 1), dim))

        # learned alpha and dampening factors

        self.alphas = nn.Parameter(torch.randn(heads))
        self.dampen_factors = nn.Parameter(torch.randn(heads))

        if bidirectional:
            self.reverse_alphas = nn.Parameter(torch.randn(heads))
            self.reverse_dampen_factors = nn.Parameter(torch.randn(heads))

    def forward(self, x):
        device, seq_len = x.device, x.shape[1]

        # project in and split heads
        x = einsum('... d, h d -> ... h d', x, self.expansion)
        if self.bidirectional:
            x, x_reversed = x.chunk(2, dim = -2)
            x_reversed = torch.flip(x_reversed, dims = (1,))
        # weights derived from alphas (learned exponential smoothing decay rate)

        def apply_learned_ema_with_damping(x, alphas, dampen_factors):
            alphas = alphas.sigmoid()
            dampen_factors = dampen_factors.sigmoid()

            reversed_powers = torch.arange(seq_len - 1, -1, -1, device = device)
            K = alphas * (((1 - alphas) * dampen_factors) ** rearrange(reversed_powers, '... l -> ... l 1'))
            # conv1d fft O(nlog(n))
            return conv1d_fft(x, K, dim = -3, weight_dim = -2)

        x = apply_learned_ema_with_damping(x, self.alphas, self.dampen_factors)
        if self.bidirectional:
            x_reversed = apply_learned_ema_with_damping(x_reversed, self.reverse_alphas, self.reverse_dampen_factors)
            x_reversed = torch.flip(x_reversed, dims = (1,))
            x = torch.cat((x, x_reversed), dim = -2)
        # combine heads and out

        return einsum('... h d, h d -> ... d', x, self.reduction)

# Mega Layer
# Single headed Attention + Multi-headed EMA, then GRU-esque gating

class MegaLayer(nn.Module):
    def __init__(
        self,
        *,
        dim = 128,
        ema_heads = 16,
        attn_dim_qk = 64,
        attn_dim_value = 256,
        laplacian_attn_fn = False,
        causal = True,
        ema_dim_head = None,
        chunk_size = -1,
        deb = False
    ):
        super().__init__()

        self.single_headed_attn = SingleHeadedAttention(
            dim = dim,
            dim_qk = attn_dim_qk,
            dim_value = attn_dim_value,
            causal = causal,
            laplacian_attn_fn = laplacian_attn_fn,
            chunk_size = chunk_size
        )

        self.multi_headed_ema = MultiHeadedEMA(
            dim = dim,
            heads = ema_heads,
            bidirectional = not causal,
            dim_head = ema_dim_head
        )

        self.to_reset_gate = nn.Sequential(
            nn.Linear(dim, attn_dim_value),
            nn.SiLU()
        )

        self.to_update_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # equation 14, for calculating H

        self.Wh = nn.Parameter(torch.randn(dim, dim))
        self.Uh = nn.Parameter(torch.randn(attn_dim_value, dim))
        self.bh = nn.Parameter(torch.randn(dim))

        self.ch = chunk_size

    def forward(self, x, residual = None):
        residual = default(residual, x)

        ema_output = self.multi_headed_ema(x)
        attn_output = self.single_headed_attn(ema_output, x)

        reset_gate = self.to_reset_gate(ema_output)
        update_gate = self.to_update_gate(ema_output)

        gated_attn_output = attn_output * reset_gate
        # equation 14

        H = F.silu(ema_output @ self.Wh + gated_attn_output @ self.Uh + self.bh)
        # update gate
        
        ans = update_gate * H + (1 - update_gate) * residual

        if(torch.isnan(ans[0,0,0]).item()):
            print('residual/x')
            print(residual.shape,residual)
            print('ema_out')
            print(ema_output.shape,ema_output)
            print('attn_out')
            print(attn_output.shape,attn_output)
        return ans

# Mega

def FeedForward(dim, ff_mult):
    dim_hidden = int(dim * ff_mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Linear(dim_hidden, dim)
    )
    

class Mega(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        ff_mult = 2,
        pre_norm = False,
        codebook_size = 8,
        pos_enc = True,
        **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens*codebook_size+1, dim)
        self.pos_enc = Summer(PositionalEncoding2D(dim))
        self.pre_norm = pre_norm
        self.ff = torch.tensor([num_tokens*n for n in range(codebook_size)]).unsqueeze(0).unsqueeze(-1)
        self.layers = nn.ModuleList([])
        self.codebook_size = codebook_size
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MegaLayer(dim = dim, **kwargs),
                nn.LayerNorm(dim),
                FeedForward(dim = dim, ff_mult = ff_mult),
                nn.LayerNorm(dim)
            ]))

        self.to_logits = nn.Sequential(
            nn.Linear(dim, num_tokens*codebook_size*ff_mult),
            nn.GELU(),
            nn.Linear(num_tokens*codebook_size*ff_mult, num_tokens*codebook_size)
        )

    def preprocess(self,x):
        ff = self.ff.to(x)
        x = x + ff
        x = rearrange(x,'b l q -> b (q l)')
        return x

    def forward(self, x):
        pre_norm = self.pre_norm
        post_norm = not self.pre_norm
        x = x.int()
        x = rearrange(self.token_emb(x), 'b (l q) d -> b l q d', q=self.codebook_size)
        x = rearrange(self.pos_enc(x), 'b l q d -> b (l q) d')

        for mega_layer, mega_norm, ff, ff_norm in self.layers:
            mega_maybe_prenorm = mega_norm if pre_norm else identity
            ff_maybe_prenorm = ff_norm if pre_norm else identity

            mega_maybe_postnorm = mega_norm if post_norm else identity
            ff_maybe_postnorm = ff_norm if post_norm else identity
            x = mega_layer(mega_maybe_prenorm(x), x)
            x = mega_maybe_postnorm(x)

            x = ff(ff_maybe_prenorm(x)) + x

            x = ff_maybe_postnorm(x)

        x = self.to_logits(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'encodec-gen',
        description = 'What the program does',
        epilog = 'Text at the bottom of help')
    parser.add_argument('-c','--code',default=1024)
    parser.add_argument('-d','--depth',default=8)
    parser.add_argument('-l','--length',default=1024)
    parser.add_argument('-b','--batch',default=4)
    parser.add_argument('-m','--model_dim',default=128)
    parser.add_argument('-ml','--model_enc_layers',default=6)
    parser.add_argument('-ch','--chunk',default=128)
    parser.add_argument('-lr','--initial_lr',default=3e-4)
    parser.add_argument('-ck','--checkpoint_steps',default=1000)
    parser.add_argument('-w','--warmup_steps',default=5000)
    parser.add_argument('-s','--save_path',default='./models')
    args = parser.parse_args()
    assert args.length*args.depth % args.chunk == 0

    # model
    
    datalist = [torch.randint(0,args.code,(4,args.depth,args.length)) for _ in range(10)]
  

    model = MegaLayer(dim=args.model_dim,attn_dim_qk=args.model_dim//2,attn_dim_value=args.model_dim*2,laplacian_attn_fn=True,causal=False,chunk_size=args.chunk)
    
    print([model(torch.rand(4,512,128,dtype=torch.float32) * _) for _ in range(1,11)])
