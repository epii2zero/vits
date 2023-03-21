import os
import json
import random
import text
from utils import load_filepaths_and_text
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import soundfile as sf
from text import text_to_sequence, cleaned_text_to_sequence

class SLRdata:
  def __init__(self, dataset_dir):
    self.dataset_dir = dataset_dir
    self.shuffle = True
    self.filelist = os.listdir(self.dataset_dir)
    self.speaker_ids = self._speaker_analysis()
    self.corpus = {'train': [], 'dev': [], 'test': []}
    self.filelist_name = 'slr109_audio_sid_text'

  def _speaker_analysis(self):
    speakers = []
    for file in self.filelist:
      filename, ext = os.path.splitext(file)
      if ext == '.json':
        speaker, _, quality, group = filename.split('_')
        if speaker not in speakers:
          speakers.append(speaker)
    speaker_ids = {speaker: str(id) for id, speaker in enumerate(speakers)}
    return speaker_ids

  def _process(self):
    for file in self.filelist:
      filename, ext = os.path.splitext(file)
      if ext != '.json':
        continue
      speaker, _, quality, group = filename.split('_')
      with open(os.path.join(self.dataset_dir, file), 'r') as f:
        for line in f:
          data = json.loads(line)
          audio_path = os.path.join(self.dataset_dir, data['audio_filepath'])
          text_norm = data['text_normalized']
          duration = data['duration']
          # remove too short data
          if duration < 0.4 or duration > 7.0:
            continue
          self.corpus[group].append([audio_path, self.speaker_ids[speaker], text_norm])
    while True:
      check = input("Type 'yes' to start processing:\n>>> ")
      if check == 'yes':
        break
    for group in self.corpus.keys():
      random.shuffle(self.corpus[group])
      filelist_path = f'./filelists/{self.filelist_name}_{group}_filelist.txt'
      f1 = open(filelist_path, "w", encoding="utf-8")
      f2 = open(f'{filelist_path}.cleaned', "w", encoding="utf-8")
      original_text_list = []
      for line in self.corpus[group]:
        f1.write('|'.join(line) + '\n')
        original_text_list.append(line[2])
      cleaned_text_list = text._clean_text(original_text_list, ["english_cleaners2_multi"])
      
      for i, line in enumerate(self.corpus[group]):
        try:
          cleaned_text_to_sequence(cleaned_text_list[i])
        except:
          error_text = '|'.join([line[0], line[1], cleaned_text_list[i]])
          print(f"Remove:\n {error_text}")
          continue
        f2.write('|'.join([line[0], line[1], cleaned_text_list[i]]) + '\n')

      f1.close()
      f2.close()

  def _analysis(self):
    stat_corpus = {'train': [], 'dev': [], 'test': []}
    for file in self.filelist:
      filename, ext = os.path.splitext(file)
      if ext != '.json':
        continue
      speaker, _, quality, group = filename.split('_')
      with open(os.path.join(self.dataset_dir, file), 'r') as f:
        for line in f:
          data = json.loads(line)
          audio_path = os.path.join(self.dataset_dir, data['audio_filepath'])
          text_norm = data['text_normalized']
          duration = data['duration']
          stat_corpus[group].append([audio_path, self.speaker_ids[speaker], text_norm, duration])
      
    for group in stat_corpus.keys():
      print(f'{group}')
      total_len = len(stat_corpus[group])
      print(f'length: {total_len}')
      durations = []
      for i in range(total_len):
        durations.append(stat_corpus[group][i][3])
      print(f'max: {max(durations)}, min: {min(durations)}')

      plt.figure(figsize=(10,10))
      plt.hist(durations, bins=20)
      plt.savefig(f'temp/SLR_{group}.png')

  def run(self):
    task = input("Type task:\n>>> ")
    if task == 'process':
      self._process()
    elif task == 'analysis':
      self._analysis()


class LJSpeech:
  def __init__(self, dataset_dir):
    self.dataset_dir = dataset_dir

  def _analysis(self):
    corpus = {'duration': []}
    filelist = os.listdir(self.dataset_dir)
    for file in filelist:
      filename, ext = os.path.splitext(file)
      if ext != '.wav':
        continue
      data, sr = sf.read(os.path.join(self.dataset_dir, file))
      duration = len(data)/sr
      corpus['duration'].append(duration)
    

    plt.figure(figsize=(10,10))
    plt.hist(corpus['duration'], bins=20)
    plt.savefig(f'temp/LJSpeech_duration.png')

  def run(self):
    task = input("Type task:\n>>> ")
    if task == 'process':
      self._process()
    elif task == 'analysis':
      self._analysis()


class Filelist_checker:
  def __init__(self, filelist_dir):
    self.filelist_dir = filelist_dir

  def _check_symbols(self):
    filelist = os.listdir(self.filelist_dir)
    for file in filelist:
      filename, ext = os.path.splitext(file)
      if ext != '.cleaned':
        continue
      with open(os.path.join(self.filelist_dir, file), 'r') as f:
        lines = f.readlines()
      for line in lines:
        data = line.strip().split('|')
        text = data[-1]
        try:
          cleaned_text_to_sequence(text)
          # print('success')
        except Exception as e:
          print('Failed')
          # print(text)
          print(f'File: {file}\n line: {line}')
          print(Exception, e)




  def run(self):
    self._check_symbols()



# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument("--out_extension", default="cleaned")
#   parser.add_argument("--text_index", default=1, type=int)
#   parser.add_argument("--filelists", nargs="+", default=["filelists/ljs_audio_text_val_filelist.txt", "filelists/ljs_audio_text_test_filelist.txt"])
#   parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

#   args = parser.parse_args()
    

#   
#       f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])






def main():
  LJSpeech_datset_dir = "./DUMMY1"
  SLR_dataset_dir = "./DUMMY3"
  fileilst_dir = './filelists' 
  instance = SLRdata(SLR_dataset_dir)
  task = input("Type dataset:\n>>> ")
  if task == 'LJSpeech':
    instance = LJSpeech(LJSpeech_datset_dir)
    instance.run()
  elif task == 'SLR109':
    instance = SLRdata(SLR_dataset_dir)
    instance.run()
  elif task == 'Check':
    instance = Filelist_checker(fileilst_dir)
    instance.run()

  


if __name__ == '__main__':
  main()