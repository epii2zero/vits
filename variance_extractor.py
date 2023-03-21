import os
import random
import json
import yaml
import argparse

# import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio

"""
  First version which ignore normalization technique
"""



class Preprocessor:
  def __init__(self, config):
    self.config = config
    self.in_dir = config["path"]["wav_path"]
    self.out_dir = config["path"]["preprocessed_path"]
    # self.val_size = config["preprocessing"]["val_size"]
    self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    self.hop_length = config["preprocessing"]["stft"]["hop_length"]

    assert config["preprocessing"]["pitch"]["feature"] in [
      "phoneme_level",
      "frame_level",
    ]
    assert config["preprocessing"]["energy"]["feature"] in [
      "phoneme_level",
      "frame_level",
    ]
    self.pitch_phoneme_averaging = (
      config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
    )
    self.energy_phoneme_averaging = (
      config["preprocessing"]["energy"]["feature"] == "phoneme_level"
    )

    self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
    self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

    self.STFT = Audio.stft.TacotronSTFT(
      config["preprocessing"]["stft"]["filter_length"],
      config["preprocessing"]["stft"]["hop_length"],
      config["preprocessing"]["stft"]["win_length"],
      config["preprocessing"]["mel"]["n_mel_channels"],
      config["preprocessing"]["audio"]["sampling_rate"],
      config["preprocessing"]["mel"]["mel_fmin"],
      config["preprocessing"]["mel"]["mel_fmax"],
    )

  def build_from_path(self):
    os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
    os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
    os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
    os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

    print("Processing Data ...")
    out = list()
    n_frames = 0
    pitch_scaler = StandardScaler()
    energy_scaler = StandardScaler()

    # Compute pitch, energy, duration, and mel-spectrogram
    for i, file in enumerate(tqdm(os.listdir(self.in_dir))):
      if ".wav" not in file:
        continue

      basename = file.split(".")[0]
      ret = self._process_utterance(basename)

    print("Computing statistic quantities ...")
    # Perform normalization if necessary
    if self.pitch_normalization:
      pitch_mean = pitch_scaler.mean_[0]
      pitch_std = pitch_scaler.scale_[0]
    else:
      # A numerical trick to avoid normalization...
      pitch_mean = 0
      pitch_std = 1
    if self.energy_normalization:
      energy_mean = energy_scaler.mean_[0]
      energy_std = energy_scaler.scale_[0]
    else:
      energy_mean = 0
      energy_std = 1

    pitch_min, pitch_max = self.normalize(
      os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
    )
    energy_min, energy_max = self.normalize(
      os.path.join(self.out_dir, "energy"), energy_mean, energy_std
    )

    # # Save files
    # with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
    # 	f.write(json.dumps(speakers))

    with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
      stats = {
        "pitch": [
          float(pitch_min),
          float(pitch_max),
          float(pitch_mean),
          float(pitch_std),
        ],
        "energy": [
          float(energy_min),
          float(energy_max),
          float(energy_mean),
          float(energy_std),
        ],
      }
      f.write(json.dumps(stats))

    print(
      "Total time: {} hours".format(
        n_frames * self.hop_length / self.sampling_rate / 3600
      )
    )

    return out

  def _process_utterance(self, basename):
    wav_path = os.path.join(self.in_dir, f"{basename}.wav")

    # Read and trim wav files
    wav, _ = librosa.load(wav_path)
    wav = wav.astype(np.float32)

    # Compute fundamental frequency
    pitch, t = pw.dio(
      wav.astype(np.float64),
      self.sampling_rate,
      frame_period=self.hop_length / self.sampling_rate * 1000,
    )
    pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
    
    if np.sum(pitch != 0) <= 1:
      return None
    
    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)

    # Save files
    pitch_filename = "pitch-{}.npy".format(basename)
    np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

    energy_filename = "energy-{}.npy".format(basename)
    np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

    mel_filename = "mel-{}.npy".format(basename)
    np.save(
      os.path.join(self.out_dir, "mel", mel_filename),
      mel_spectrogram.T,
    )

    return (
      self.remove_outlier(pitch),
      self.remove_outlier(energy),
      mel_spectrogram.shape[1],
    )

  def remove_outlier(self, values):
    values = np.array(values)
    p25 = np.percentile(values, 25)
    p75 = np.percentile(values, 75)
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)
    normal_indices = np.logical_and(values > lower, values < upper)

    return values[normal_indices]

  def normalize(self, in_dir, mean, std):
    max_value = np.finfo(np.float64).min
    min_value = np.finfo(np.float64).max
    for filename in os.listdir(in_dir):
      filename = os.path.join(in_dir, filename)
      values = (np.load(filename) - mean) / std
      np.save(filename, values)

      max_value = max(max_value, max(values))
      min_value = min(min_value, min(values))

    return min_value, max_value

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, default="./configs/variance_extractor.yaml",
                      help="path to preprocess.yaml")
  args = parser.parse_args()
  config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
  
  preprocessor = Preprocessor(config)
  preprocessor.build_from_path()