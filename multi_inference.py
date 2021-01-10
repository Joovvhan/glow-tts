import matplotlib.pyplot as plt

import sys
sys.path.append('./waveglow/')

import librosa
import numpy as np
import os
import glob
import json

import scipy.io.wavfile as wavfile

import torch
from text import text_to_sequence, cmudict
from text.symbols import symbols
import commons
import attentions
import modules
import models
import utils

def main(tst_stn, model_dir, save_name):

  # load waveglow
  waveglow_path = './waveglow/waveglow_256channels_universal_v5.pt'
  # waveglow_path = './waveglow/waveglow_114000'
  # waveglow_path = './waveglow/waveglow_248000'
  waveglow = torch.load(waveglow_path)['model']
  waveglow = waveglow.remove_weightnorm(waveglow)
  _ = waveglow.cuda().eval()
  from apex import amp
  waveglow, _ = amp.initialize(waveglow, [], opt_level="O3") # Try if you want to boost up synthesis speed.

  ###############################################################################

  model_dir = model_dir
  # model_dir = "./logs/grapheme"
  hps = utils.get_hparams_from_dir(model_dir)
  model = models.FlowGenerator(
      len(symbols),
      out_channels=hps.data.n_mel_channels,
      **hps.model).to("cuda")

  checkpoint_path = utils.latest_checkpoint_path(model_dir)
  utils.load_checkpoint(checkpoint_path, model)
  model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
  _ = model.eval()

  # cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)
  cmu_dict = None

  ###############################################################################

  # tst_stn = " Glow TTS is really awesome ! " # Adding spaces at the beginning and the end of utterance improves quality
  # sequence = np.array(text_to_sequence(tst_stn, ['english_cleaners'], cmu_dict))[None, :]
  sequence = np.array(text_to_sequence(tst_stn, ['korean_cleaners'], cmu_dict))[None, :]

  with open('g2p_result.txt', 'a') as f:
    f.write("".join([symbols[c] for c in sequence[0]]))
    f.write(" | " + save_name)
    f.write("\n")
  
  print("".join([symbols[c] for c in sequence[0]]))
  x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
  x_tst_lengths = torch.tensor([x_tst.shape[1]]).cuda()

  ###############################################################################

  with torch.no_grad():
    noise_scale = .333
    # noise_scale = .5
    length_scale = 1.0
    # length_scale = 0.75
    (y_gen_tst, *r), attn_gen, *_ = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)
    
    try:
      audio = waveglow.infer(y_gen_tst.half(), sigma=.666)
    except:
      audio = waveglow.infer(y_gen_tst, sigma=.666)

  audio = audio.cpu().numpy()[0]
  audio *= 2 ** 15
  audio = audio.astype(np.int16)

  wavfile.write(f'{save_name}', hps.data.sampling_rate, audio)

if __name__ == "__main__":

  model_path = sys.argv[1]
  script_file = sys.argv[2]

  with open('g2p_result.txt', 'w') as f:
    f.write('')

  with open(script_file, 'r') as file:
    for line in file:
      sentence, save_name = line.split('|')
      main(sentence, model_path, save_name.strip())  
