from torch.utils.data import Dataset

import os 
import json
import h5py
import pandas as pd
from numpy import random
import numpy as np
from tqdm import tqdm
import time
import torch
import copy

from separator.audio.ffmpeg import FFMPEGProcessAudioAdapter
from separator.utils.stft import STFT

def save(data, file):
    with open(file, 'w') as f:
        json.dump(data, f)

from config_files.General_config import General_config
from config_files.General_config import Train_config
from config_files.Model_config import Model_config

instrument_list = Train_config["instrument_list"]

class loader_STFT_musdb(Dataset):
    def __init__(self):
    
        self.instrument_list = copy.deepcopy(instrument_list)
        self.instrument_list.append("mixture")
        self.audio_root = Train_config['audio_root_extra']
        self.cache_root = Train_config["cache_root_extra"]
        self.cache_list_path = os.path.join(self.cache_root,'cache_list.json')
        self.spectrogram_loader_path = os.path.join(self.cache_root,'spectrogram_loader.json')
        self.sample_rate = Model_config["sample_rate"]
        self.total_length = 0
        self.channels = Model_config["channels"]
        self.segment_length = Model_config["segment_length"]
        self.frequency_bins = Model_config["frequency_bins"]
        self.adapter = FFMPEGProcessAudioAdapter()
        frame_length = Model_config["frame_length"]
        frame_step = Model_config["frame_step"]
        self.stft = STFT(frame_length, frame_step)
        self.cache_list = self.create_cache_list()
        with open(self.spectrogram_loader_path) as f:
            self.spectrogram_loader = json.load(f)
        self.spectrogram_loader_copy = copy.deepcopy(self.spectrogram_loader)


    # Create a json file with all the paths
    def create_cache_list(self): 
        cache_list_path = self.cache_list_path

        if os.path.exists(cache_list_path) == False:
            os.system("touch " + cache_list_path)
            with open(cache_list_path,'w') as f:
                default = {}
                for i in self.instrument_list:
                    default[i] = {}
                json.dump(default, f)

        with open(cache_list_path) as f:
            cache_list = json.load(f)

        for folder_name in os.listdir(self.audio_root): 
            if folder_name in self.instrument_list:
                folder_path = os.path.join(self.audio_root, folder_name)
                if os.path.isdir(folder_path): 
                    for file_name in os.listdir(folder_path):
                        if ".mp3" in file_name:
                            file_path = os.path.join(folder_path,file_name)
                            if file_path not in cache_list[folder_name]:
                                cache_list[folder_name][file_path] = 0

        save(cache_list,cache_list_path)
        self.generate_stft(cache_list)
        return cache_list

    def generate_stft(self,cache_list):

        spectrogram_loader_path = self.spectrogram_loader_path
        if os.path.exists(spectrogram_loader_path) == False:
            os.system("touch " + spectrogram_loader_path)
            with open(spectrogram_loader_path,'w') as f:
                default = {}
                for i in self.instrument_list:
                    default[i] = {}
                default["length"] = 0 
                json.dump(default, f)

        with open(spectrogram_loader_path) as f:
            spectrogram_loader = json.load(f)

        cache_list_path = self.cache_list_path
        for i in self.instrument_list:
            if os.path.exists(os.path.join(self.cache_root,i)) == False:
                os.system("mkdir " + os.path.join(self.cache_root,i))

        for instrument_type in cache_list:
            print("converting " + str(instrument_type) + " to spectrogram")
            for file_path in tqdm(cache_list[instrument_type]):
                # when a mp3 file is not cached
                if cache_list[instrument_type][file_path] == 0:
                    # try:
                    # print(file_path)
                    if  "._" not in file_path:
                        wave = self.adapter.load(path=file_path,sample_rate=self.sample_rate, channels=self.channels)
                        # remove first 5 seconds
                        if len(wave) > 5*44100:
                            wave = wave[5*44100:-1]
                            spectrogram = self.stft.stft(wave)
                            spectrogram = np.abs(spectrogram)
                            spectrogram = spectrogram[:,0:self.frequency_bins,:]
                            start = spectrogram.shape[0]//4
                            end = spectrogram.shape[0]*3//4
                            spectrogram= spectrogram[start:end,0:self.frequency_bins,:]
                            stft_path = file_path.replace(self.audio_root,self.cache_root)
                            spectrogram_length = spectrogram.shape[0]
                            stft_size = spectrogram_length//self.segment_length
                            spectrogram_loader[instrument_type][stft_path] = stft_size
                            if instrument_type == "vocals":
                                spectrogram_loader["length"] = spectrogram_loader["length"] + stft_size
                            stft_cache = h5py.File(stft_path, 'w')
                            stft_cache['spectrogram'] = spectrogram
                            stft_cache.flush()
                            stft_cache.close()
                            cache_list[instrument_type][file_path] = 1
                            save(cache_list,cache_list_path)
                            save(spectrogram_loader,spectrogram_loader_path)
                        # except:
                        #     print("ffmpeg warnings")

    # 总共多少组
    def __len__(self):
        if Train_config["train_length"] != None:
            return Train_config["train_length"]
        else:
            return self.total_length

    def __getitem__(self, idx):
        spectrograms = dict()
        self.zero_flag = 3
        # print("next")
        for instrument_type in self.instrument_list:
            if self.spectrogram_loader_copy[instrument_type] == {}:
                self.spectrogram_loader_copy[instrument_type] = copy.deepcopy(self.spectrogram_loader[instrument_type])

            try:
                for stft_paths in  self.spectrogram_loader_copy[instrument_type]:
                    stft_size = self.spectrogram_loader_copy[instrument_type][stft_paths]
                    if stft_size > 0:
                        file = h5py.File(stft_paths, 'r')
                        subspectrogram = file["spectrogram"]
                        subspectrogram = subspectrogram[(stft_size-1)*self.segment_length:(stft_size)*self.segment_length,:,:]
                        # transpose to (1,512,1537)
                        subspectrogram = np.transpose(subspectrogram[:, :], axes=(2, 0, 1))
                        
                        # randomly set certain instruments to zero

                        if "instrument_type" == "drums":
                            if random.random() <= 0.35:
                                subspectrogram = np.zeros(subspectrogram.shape)
                                self.zero_flag = self.zero_flag -1 
                        elif "instrument_type" == "vocals":
                            if random.random() <= 0.25:
                                subspectrogram = np.zeros(subspectrogram.shape)
                                self.zero_flag = self.zero_flag -1 
                        elif "instrument_type" == "bass":
                            if random.random() <= 0.35:
                                subspectrogram = np.zeros(subspectrogram.shape)
                                self.zero_flag = self.zero_flag -1 
                        elif "instrument_type" == "other":
                            if random.random() <= 0.05 and self.zero_flag != 0:
                                subspectrogram = np.zeros(subspectrogram.shape)
    
                        if stft_size <= 1:
                            self.spectrogram_loader_copy[instrument_type].pop(stft_paths)
                        else:
                            self.spectrogram_loader_copy[instrument_type][stft_paths] = stft_size - 1
                        spectrograms[instrument_type] = torch.from_numpy(subspectrogram)
                        file.close()
                        break
                    else:
                        self.spectrogram_loader_copy[instrument_type].pop(stft_paths)

            # sometimes songs have zero frame. The following part solves the exception.
            # You can also do it in stft generator
            except:
                
                if self.spectrogram_loader_copy[instrument_type] == {}:
                    self.spectrogram_loader_copy[instrument_type] = self.spectrogram_loader[instrument_type].copy()

                for stft_paths in  self.spectrogram_loader_copy[instrument_type]:
                    stft_size = self.spectrogram_loader_copy[instrument_type][stft_paths]
                    if stft_size > 0:
                        file = h5py.File(stft_paths, 'r')
                        subspectrogram = file["spectrogram"]
                        subspectrogram = subspectrogram[(stft_size-1)*self.segment_length:(stft_size)*self.segment_length,:,:]
                        # transpose to (1,512,1537)
                        subspectrogram = np.transpose(subspectrogram[:, :], axes=(2, 0, 1))

                        if stft_size <= 1:
                            self.spectrogram_loader_copy[instrument_type].pop(stft_paths)
                        else:
                            self.spectrogram_loader_copy[instrument_type][stft_paths] = stft_size - 1
                        spectrograms[instrument_type] = torch.from_numpy(subspectrogram)
                        file.close()
                        break

        # add to form mix
        for instrument_type in self.instrument_list:
            if "mixture" not in spectrograms:
                spectrograms["mixture"] = torch.from_numpy(np.zeros(spectrograms[instrument_type].shape))
            spectrograms["mixture"] = spectrograms["mixture"] + spectrograms[instrument_type]

        return spectrograms
    
