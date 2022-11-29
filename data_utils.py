import os
import numpy as np
import torch
import glob
from pathlib import Path
from torch.utils.data import Dataset

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

class EnCodecData(Dataset):
    def __init__(self, data_list):
        self.dic = data_list


    def __len__(self):
        return len(self.dic)

    def __getitem__(self, idx):
        return self.dic[idx]
