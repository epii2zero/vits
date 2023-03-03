import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import json
import math
import yaml
import argparse
import torch
import soundfile
import psutil
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import soundfile as sf


def get_text(text, hps):
  text_norm = text_to_sequence(text, hps.data.text_cleaners)
  if hps.data.add_blank:
    text_norm = commons.intersperse(text_norm, 0)
  text_norm = torch.LongTensor(text_norm)
  return text_norm


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./inference/configure.yaml",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--mode', type=str, default="normal",
                      help='mode for inference')
  args = parser.parse_args()
  config_path = args.config
  # model_type = 'LJspeech'
  model_type = 'VCTK'
  with open(config_path, "r") as f:
    config = yaml.safe_load(f)
  hps_path = config[model_type]['hparams_path']
  hps = utils.get_hparams_from_file(hps_path)

  ckp_path = config[model_type]['pretrained_path']
  infer = Inference(hps, ckp_path, args.mode)

  input_path = config[model_type]['input_path']
  with open(input_path, "r") as f:
    input_config = yaml.safe_load(f)

  infer.tts(input_config, model_type)


class Inference():
  def __init__(self, hps, ckp_path, mode):
    self.hps = hps
    self._model_prep(ckp_path)
    self.mode = mode

  def _model_prep(self, ckp_path):
    self.net_g = SynthesizerTrn(
      len(symbols),
      self.hps.data.filter_length // 2 + 1,
      self.hps.train.segment_size // self.hps.data.hop_length,
      n_speakers=self.hps.data.n_speakers,
      **self.hps.model).cuda()
    self.net_g.eval()
    _ = utils.load_checkpoint(ckp_path, self.net_g, None)

  def _report(self, message: str = 'DEBUG') -> None:
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20  # Bytes to MB
    print(f"[{message}] memory usage: {rss: 10.5f} MB\n")

  def tts(self, input_config, model_type):
    for name in input_config.keys():
      task = input_config[name]
      text = task['text']
      length_scale = task['length_scale']
      if self.hps.data.n_speakers > 0:
        sid = task['sid']
        sid = torch.LongTensor([sid]).cuda()
      else:
        sid = None
      if 'noise_scale' in task:
        noise_scale = task['noise_scale']
      else:
        noise_scale = .667
      
      stn_tst = get_text(text, self.hps)
      with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        o, attn, y_mask, (z, z_p, m_p, logs_p) = self.net_g.infer(
            x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
            noise_scale_w=0.8, length_scale=length_scale
            )
        audio = o[0, 0].data.cpu().float().numpy()
      
      

      path_name = f'{name}_{model_type}'
      self._attention_map(attn[0,0].transpose(0,1).data.cpu().numpy(), path_name=path_name)
      sf.write(f'./inference/result/{path_name}.wav',
               audio, self.hps.data.sampling_rate, 'PCM_16')

  def _attention_map(self, attn, path_name, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = plt.imshow(attn, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
      xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    save_dir = './inference/figures'
    title = path_name
    plt.savefig(f'{save_dir}/{title}.png')
    plt.close()

  def _latent_plot(self, latent):
    pass
if __name__ == "__main__":
  main()
