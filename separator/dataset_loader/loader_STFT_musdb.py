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
        self.audio_root = Train_config['audio_root_musdb_train']
        self.cache_root = Train_config["cache_root_musdb_train_F"]
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


    def create_cache_list(self): 
        cache_list_path = self.cache_list_path

        if os.path.exists(cache_list_path) == False:
            os.system("touch " + cache_list_path)
            # print(cache_list_path)
            with open(cache_list_path,'w') as f:
                default = {}
                json.dump(default, f)

        with open(cache_list_path) as f:
            cache_list = json.load(f)

        for song_name in os.listdir(self.audio_root): 
            song_path = os.path.join(self.audio_root, song_name)
            if os.path.isdir(song_path) == True:
                if song_name not in cache_list:
                    cache_list[song_name] = {}
                for instrument_type in self.instrument_list:
                    instrument_path = os.path.join(song_path, instrument_type)
                    instrument_path = instrument_path + ".wav"
                    if instrument_path not in cache_list[song_name]:
                        cache_list[song_name][instrument_path] = 0
        save(cache_list,cache_list_path)

        self.generate_stft(cache_list)
        return cache_list

    def generate_stft(self,cache_list):

        spectrogram_loader_path = self.spectrogram_loader_path
        if os.path.exists(spectrogram_loader_path) == False:
            os.system("touch " + spectrogram_loader_path)
            with open(spectrogram_loader_path,'w') as f:
                default = {}
                default["length"] = 0 
                json.dump(default, f)

        with open(spectrogram_loader_path) as f:
            spectrogram_loader = json.load(f)
            if "instrument_list" not in spectrogram_loader:
                spectrogram_loader["instrument_list"] = instrument_list

        cache_list_path = self.cache_list_path

        with open(cache_list_path) as f:
            cache_list = json.load(f)

        print("converting musdb to STFT")
        
        for song_name in tqdm(cache_list):
            
            if (song_name not in spectrogram_loader):
                spectrogram_loader[song_name] = {}
                if os.path.exists(os.path.join(self.cache_root,song_name)) == False:
                    song_name_format = song_name.replace(" ", "\ ")
                    song_name_format = song_name_format.replace("'", "\\" +"'")
                    song_name_format = song_name_format.replace("&", "\\" +"&")
                    song_name_format = song_name_format.replace("(", "\\" +"(")
                    song_name_format = song_name_format.replace(")", "\\" +")")
                    os.system("mkdir " + os.path.join(self.cache_root,song_name_format))

                for instrument_path in cache_list[song_name]:
                    if cache_list[song_name][instrument_path] == 0:
                        stft_path = instrument_path.replace(self.audio_root,self.cache_root)
                        wave = self.adapter.load(path=instrument_path,sample_rate=self.sample_rate, channels=self.channels)
                        spectrogram = self.stft.stft(wave)
                        spectrogram = np.abs(spectrogram)
                        spectrogram = spectrogram[:,0:self.frequency_bins,:]
                        spectrogram_length = spectrogram.shape[0]
                        stft_size = spectrogram_length//self.segment_length
                        spectrogram_loader[song_name][stft_path] = stft_size
                        if "vocals" in stft_path:
                            spectrogram_loader["length"] = spectrogram_loader["length"] + stft_size
                        stft_cache = h5py.File(stft_path, 'w')
                        stft_cache['spectrogram'] = spectrogram
                        stft_cache.flush()
                        stft_cache.close()
                        cache_list[song_name][instrument_path] = 1
                        save(cache_list,cache_list_path)
                        save(spectrogram_loader,spectrogram_loader_path)

            elif spectrogram_loader["instrument_list"] != instrument_list:
                for instrument_path in cache_list[song_name]:
                    if cache_list[song_name][instrument_path] == 0:
                        stft_path = instrument_path.replace(self.audio_root,self.cache_root)
                        wave = self.adapter.load(path=instrument_path,sample_rate=self.sample_rate, channels=self.channels)
                        spectrogram = self.stft.stft(wave)
                        spectrogram = np.abs(spectrogram)
                        spectrogram = spectrogram[:,0:self.frequency_bins,:]
                        spectrogram_length = spectrogram.shape[0]
                        stft_size = spectrogram_length//self.segment_length
                        spectrogram_loader[song_name][stft_path] = stft_size
                        if "vocals" in stft_path:
                            spectrogram_loader["length"] = spectrogram_loader["length"] + stft_size
                        stft_cache = h5py.File(stft_path, 'w')
                        stft_cache['spectrogram'] = spectrogram
                        stft_cache.flush()
                        stft_cache.close()
                        cache_list[song_name][instrument_path] = 1
                        save(cache_list,cache_list_path)
                        save(spectrogram_loader,spectrogram_loader_path)
        
        spectrogram_loader["instrument_list"] = instrument_list
        self.total_length = spectrogram_loader["length"]

# 总共多少组
    def __len__(self):
        if Train_config["train_length"] != None:
            return Train_config["train_length"]
        else:
            return self.total_length

    def __getitem__(self, idx):
        spectrograms = dict()
        if self.spectrogram_loader_copy == {}:
            self.spectrogram_loader_copy = copy.deepcopy(self.spectrogram_loader)
        try:
            for song_name in self.spectrogram_loader_copy:
                if song_name != "length" and song_name != "instrument_list":
                    for stft_path in  self.spectrogram_loader_copy[song_name]:
                        stft_size = self.spectrogram_loader_copy[song_name][stft_path]
                        if stft_size > 0:
                            file = h5py.File(stft_path, 'r')
                            subspectrogram = file["spectrogram"]
                            subspectrogram = subspectrogram[(stft_size-1)*self.segment_length:(stft_size)*self.segment_length,:,:]
                            # transpose to (1,512,1536)
                            subspectrogram = np.transpose(subspectrogram[:, :], axes=(2, 0, 1))
                        
                            spectrograms[stft_path.split(".")[-2].split("/")[-1]] = subspectrogram

                            self.spectrogram_loader_copy[song_name][stft_path] = stft_size - 1
                            file.close()
                        else:
                            self.spectrogram_loader_copy.pop(song_name)
                    break

                # sometimes songs have zero frame. The following part solves the exception.
                # You can also do it in stft generator
        except:
            if self.spectrogram_loader_copy == {}:
                self.spectrogram_loader_copy = copy.deepcopy(self.spectrogram_loader)
            for song_name in self.spectrogram_loader_copy:
                if song_name != "length" and song_name != "instrument_list":
                    for stft_path in  self.spectrogram_loader_copy[song_name]:
                        # print(self.spectrogram_loader_copy[song_name])
                        stft_size = self.spectrogram_loader_copy[song_name][stft_path]
                        if stft_size > 0:
                            file = h5py.File(stft_path, 'r')
                            subspectrogram = file["spectrogram"]
                            subspectrogram = subspectrogram[(stft_size-1)*self.segment_length:(stft_size)*self.segment_length,:,:]
                            # transpose to (1,512,1536)
                            subspectrogram = np.transpose(subspectrogram[:, :], axes=(2, 0, 1))
                            spectrograms[stft_path.split(".")[-2].split("/")[-1]] = subspectrogram
                            self.spectrogram_loader_copy[song_name][stft_path] = stft_size - 1
                            file.close()
                        else:
                            self.spectrogram_loader_copy.pop(song_name)
                    break

        return spectrograms