import os
import numpy as np
import torch
import glob
from pathlib import Path
from torch.utils.data import Dataset
from numpy.random import default_rng

def getData(path,length,fraction=1.0):
    path = path
    l = length
    wildpath = os.path.join(path, '*.npy')
    paths = sorted(glob.glob(wildpath))
    paths = paths[:int(len(paths)*fraction)]
    i = -1
    prev = ''
    dic = []
    t = []
    for j,p in enumerate(paths):
        s = Path(p).stem[:-4]
        if prev != s:
            i+=1
            prev=s
            if len(t)>0:
                combined = torch.cat(t,dim=-1)
                ll = list(combined.split(length,-1))
                if ll[-1].size(-1) < length:
                    ll[-1] = combined[:,-length:]
                dic += ll
                del t
                t=[]
        ten = torch.from_numpy(np.load(p)).squeeze()
        t.append(ten)
        print(str(j)+'/'+str(len(paths)))
    if len(t)>0:
        combined = torch.cat(t,dim=-1)
        ll = list(combined.split(length,-1))
        if ll[-1].size(-1) < length:
            ll[-1] = combined[:,-length:]
        dic += ll
        del t
    return dic

def augment_data(length, codebook_dim=8):
    def to_return(data_list):
        l = []
        for data in data_list:
            data = data[:codebook_dim,:]
            i = -1 if data.size(-1)%length == 0 else None
            l += data.split(length,dim=-1)[:i]
        return l
    return to_return

class RepeatingSampler():
    def __init__(self, data_source, batch_size, shuffle=False) -> None:
        self.data_source = data_source
        self.b = batch_size
        self.rng = default_rng() if shuffle else None
    def __iter__(self):
        tmp = np.repeat(np.arange(len(self.data_source)),self.b).reshape((-1,self.b))
        if self.rng is not None:
            self.rng.shuffle(tmp,axis=0)
        for i in range(len(self.data_source)):
            yield tmp[i,:]
    def __len__(self) -> int:
        return len(self.data_source)

class EnCodecData(Dataset):
    def __init__(self, data_list, transforms=None):
        self.dic = data_list
        if transforms is not None:
            self.dic = transforms(data_list)
    def __len__(self):
        return len(self.dic)

    def __getitem__(self, idx):
        return self.dic[idx]
