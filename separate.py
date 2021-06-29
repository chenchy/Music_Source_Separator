from separator.audio.ffmpeg import FFMPEGProcessAudioAdapter
from separator.utils.stft import STFT
import numpy as np
import torch
import time
import copy
import matplotlib.pyplot as plt

from config_files.General_config import General_config
from config_files.General_config import Separate_config
from config_files.General_config import Eval_config
from config_files.Model_config import Model_config

from separator.utils.create_models import create_model

MODEL = General_config["model"]
torch.set_flush_denormal = True

if  "eval" in General_config["phase"]:
    instrument_list = Eval_config["instrument_list"]

elif "separate" in General_config["phase"]:
    instrument_list = Separate_config["instrument_list"]


def F_preprocess(wave):
    stft = STFT(Model_config["frame_length"],Model_config["frame_step"])
    spectrogram = stft.stft(wave)
    if len(spectrogram) == 1:
        reps = [2] + [1] * (len(spectrogram.shape) - 1)
        spectrogram = np.tile(spectrogram, reps=reps)
    else:
        spectrogram = spectrogram[...,:2]
    # reshape to input
    split_size = Model_config["segment_length"]
    padding = split_size - np.mod(spectrogram.shape[0], split_size)
    pad_width = [[0, 0]] * len(spectrogram.shape)
    pad_width[0] = [0, padding]
    spectrogram = np.pad(spectrogram, pad_width)
    split_num = spectrogram.shape[0] // split_size
    spectrogram = spectrogram.reshape((split_num, split_size) + spectrogram.shape[1:])
    raw_spectrogram = spectrogram
    spectrogram = np.abs(spectrogram)
    spectrogram = spectrogram[:, :, :Model_config["frequency_bins"], :]
    spectrogram = np.ascontiguousarray(np.transpose(spectrogram, axes=[0, 3, 1, 2]))
    spectrogram = torch.from_numpy(spectrogram)
    return spectrogram, raw_spectrogram


def F_separate(model,wave):
    stft = STFT(Model_config["frame_length"],Model_config["frame_step"])
    spectrogram, raw_spectrogram = F_preprocess(wave)
    
    with torch.no_grad():
        if  "eval" in General_config["phase"]:
            masks = model(spectrogram.to(Eval_config["device"]))
        elif "separate" in General_config["phase"]:
            masks = model(spectrogram.to(Separate_config["device"]))
    outputs = {}
    
    for instrument in instrument_list:
        mask = masks[instrument]
        value = torch.zeros(mask.shape[:-1], dtype=mask.dtype, device=mask.device).unsqueeze(dim=-1)
        row = Model_config["frame_length"] // 2 + 1 - Model_config["frequency_bins"]
        value = torch.repeat_interleave(value, repeats=row, dim=-1)
        mask = torch.cat((mask, value), dim=-1)
        mask = mask.permute(dims=[0, 2, 3, 1])
        mask = mask.reshape(shape=((-1, ) + mask.shape[2:]))
        mask = mask.to("cpu")
        mask = mask.numpy()
        raw_spectrogram = np.reshape(raw_spectrogram,mask.shape)  
        output = mask * raw_spectrogram
        output = stft.istft(output,len(wave))
        outputs[instrument] = output
    return outputs

def T_separate(model,wave):
    wave = torch.from_numpy(wave)
    if General_config["phase"] == "separate":
        wave = wave.to(Separate_config["device"])
    elif General_config["phase"] == "eval":
        wave = wave.to(Eval_config["device"])
    wave = np.transpose(wave)
    wave = wave[:,:]
    wave = wave.unsqueeze(0)
    with torch.no_grad():
        output = model(wave.float())
    outputs = {}
    for instrument in instrument_list:
        outputs[instrument] = output[instrument].numpy()
    return outputs

def main():
    audio = FFMPEGProcessAudioAdapter()
    wave = audio.load(path= Separate_config['audio_path'],sample_rate=Separate_config["sample_rate"], channels = Separate_config["channels"])
    pretrain_model = Separate_config["pretrain_model"]    
    model = create_model("separate")
    model.to(Separate_config["device"])
    model.load_state_dict(torch.load(pretrain_model, map_location=Separate_config["device"]))
    model.eval()

    if Model_config["input_type"] == "F":
        results = F_separate(model,wave)

    if Model_config["input_type"] == "T":
        results = T_separate(model,wave)

    for key, value in results.items():
        output_path = Separate_config["output_path"] + "/separated_%s.wav"%(key)
        audio.save(output_path, value, Separate_config["sample_rate"], Separate_config["channels"], 'wav','128k')

if __name__ == '__main__':
    main()

