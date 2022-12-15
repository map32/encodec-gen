from model import LitAutoEncoder
import torch
import argparse
import gc,math,numpy
from encodec import EncodecModel
from einops import rearrange
import torchaudio

def noise(x,step,maxsteps,mask_id):
    frac = math.cos((step/maxsteps)*math.pi/2)
    print(frac)
    nmask = torch.rand_like(x,dtype=torch.float32) < frac
    return nmask * mask_id + ~nmask * x

def getCodes(path):
    n = numpy.load(path)
    print(n.shape)
    #print(n)
    return torch.tensor(n[:,:4,:512])

def save_audio(wav, path, sample_rate = 24000, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog = 'encodec-gen',
        description = 'What the program does',
        epilog = 'Text at the bottom of help')
    
    parser.add_argument('filepath')
    parser.add_argument('-s','--step',default=50)
    parser.add_argument('-o','--output',default='result.wav')
    parser.add_argument('-c','--condition',default = '')
    args = parser.parse_args()
    codebook = 1024
    depth=4
    mask_id = depth*codebook
    length=512

    # model

    m = LitAutoEncoder.load_from_checkpoint(args.filepath)
    m.eval()
    maxsteps = int(args.step)

    x = torch.ones(1,depth*length) * mask_id
    if args.condition != '':
        xx = getCodes(args.condition)
        print(xx)
        xx = m.denoiser.preprocess(xx)
        print(xx)
        x[:,:256*4] = xx[:,:256*4]
    for i in range(maxsteps):
        i += 1
        x_prob = m(x)
        x_prob[:,::4,1024:] = -1e9
        x_prob[:,1::4,:1024] = -1e9
        x_prob[:,1::4,2048:] = -1e9
        x_prob[:,2::4,:2048] = -1e9
        x_prob[:,2::4,3072:] = -1e9
        x_prob[:,3::4,:3072] = -1e9
        #print(x_prob)
        x = m.sample(x_prob)
        print(x)
        x = noise(x,i,maxsteps,mask_id)
        print(x)
        if args.condition != '':
            x[:,:256*4] = xx[:,:256*4]
            print(x)
    print(x.shape)
    print(x)

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(3.0)
    s2 = x % codebook
    print(s2)
    s2 = rearrange(s2,'b (l q) -> b q l',q=depth)
    print(s2.shape,s2)
    l = [(s2,None)]
    wav = model.decode(l).squeeze(0)
    print(wav.shape,wav)
    save_audio(wav,args.output)

