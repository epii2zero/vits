import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa

dirs = {
  "dataset": "./DUMMY1",
  "preprocessed": "preprocessed_data/LJSpeech_paper",
  "save": "./temp"
}
samples = {"0003": "LJ001-0003"}

class CheckData():
  def __init__(self):
    self.datum = {}
    for sample, name in samples.items():
      wav_path = f'{dirs["dataset"]}/{name}.wav'
      wav, sr = sf.read(wav_path)
      
      pitch_path = f'{dirs["preprocessed"]}/pitch/pitch-{name}.npy'
      pitch = np.load(pitch_path)

      energy_path = f'{dirs["preprocessed"]}/energy/energy-{name}.npy'
      energy = np.load(energy_path)

      mel_path = f'{dirs["preprocessed"]}/mel/mel-{name}.npy'
      mel = np.load(mel_path)

      self.datum[name] = (wav, pitch, energy, mel)
      
  def analysis(self):
    for name, data in self.datum.items():
      wav, pitch, energy, mel = data
      print(energy.size)
      print(pitch.size)
      print(mel.size)
      plt.figure(figsize=(10, 10))
      plt.subplot(2,2,1)
      plt.plot(pitch)
      plt.subplot(2,2,2)
      plt.plot(energy)
      plt.subplot(2,2,3)
      librosa.display.specshow(mel, sr=22050)
      plt.savefig(f'{dirs["save"]}/{name}.png')
      exit()


def main():
  checkdata = CheckData()
  checkdata.analysis()

if __name__ == '__main__':
  main()