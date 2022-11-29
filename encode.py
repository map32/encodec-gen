import glob
import torch
from torch import nn
from encodec import EncodecModel
from encodec.utils import convert_audio
import numpy as np
import torchaudio
import subprocess
from pydub import AudioSegment
print(subprocess.check_output(['ffmpeg', '-version']))

from pathlib import Path
paths = glob.glob('/content/drive/MyDrive/wave/*')
paths_ = [Path(p).stem[:-4] for p in glob.glob('/content/drive/MyDrive/wave_encodec/*0000.npy')]
def filt(item):
  return Path(item).stem not in paths_
paths = list(filter(filt,paths))


# Instantiate a pretrained EnCodec model
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
for i,p in enumerate(paths):
  try:
    print(str(i)+'/'+str(len(paths)))
    print(p)
    audio = AudioSegment.from_file(p,format=Path(p).suffix[1:])
    audio = audio.set_frame_rate(24000)
    audio = audio.set_channels(1)
    samples = audio.get_array_of_samples()
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples.typecode).max
    print(fp_arr.shape)
    sr = audio.frame_rate
    print(sr,len(fp_arr))
    # Extract discrete codes from EnCodec
    print(list(range(0,len(fp_arr),24000*30)))
    waves = [fp_arr[st:st+24000*30] for st in range(0,len(fp_arr),24000*30)]
    print(waves)
    for n,wav in enumerate(waves):
      w = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0)
      encoded_frames = model.encode(w)
      codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
      filename = Path(p).stem
      with open('./drive/MyDrive/wave_encodec/'+filename+'{:04d}'.format(n)+'.npy', 'wb') as f:
        np.save(f,codes.numpy())
  except Exception as e:
    print(str(e)+', continuing')

  #print('exception occurred reading '+p+', '+str(e))
  #i += 1
    