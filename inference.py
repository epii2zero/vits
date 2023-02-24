import matplotlib.pyplot as plt

import os
import json
import math
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
    parser.add_argument('-c', '--config', type=str, default="./inference/configure.json",
                        help='JSON file for configuration')
    parser.add_argument('-m', '--mode', type=str, help='mode for inference')
    args = parser.parse_args()
    config_path = args.config
    with open(config_path, "r") as f:
        data = f.read()
        config = json.loads(data)
    hps_path = config['LJspeech']['hparams_path']
    hps = utils.get_hparams_from_file(hps_path)
    ckp_path = config['LJspeech']['pretrained_path']
    infer = Inference(hps, ckp_path)
    infer.tts('Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.')


class Inference():
    def __init__(self, hps, ckp_path):
        self.hps = hps
        self._model_prep(ckp_path)

    def _model_prep(self, ckp_path):
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model).cuda()
        self.net_g.eval()
        _ = utils.load_checkpoint(ckp_path, self.net_g, None)
    
    def _report(self, message: str = 'DEBUG') -> None:
        # current process RAM usage
        p = psutil.Process()
        rss = p.memory_info().rss / 2 ** 20 # Bytes to MB
        print(f"[{message}] memory usage: {rss: 10.5f} MB\n")


    def tts(self, text):
        stn_tst = get_text(text, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            audio = self.net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        sf.write('./inference/result/VITS_tts.wav', audio, self.hps.data.sampling_rate, 'PCM_16')



if __name__ == "__main__":
    main()