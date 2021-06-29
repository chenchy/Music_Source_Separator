import musdb
import museval
from glob import glob
import os
import json
import numpy as np
import torch
from tqdm import tqdm
import mir_eval

from separator.audio.ffmpeg import FFMPEGProcessAudioAdapter
from separate import F_separate, T_separate
from separator.utils.create_models import create_model


from config_files.General_config import General_config
from config_files.General_config import Eval_config
from config_files.Model_config import Model_config

input_type = Model_config["input_type"]
eval_dataset =  Eval_config["eval_dataset"]
instrument_list = Eval_config["instrument_list"]

# def si_snr(source,estimate_source,eps = 1e-5):
#     B,T = source.shape[0],source.shape[1]
#     source_energy = torch.sum(source ** 2,dim = 1).view(B, 1)  # B , 1
#     print(source_energy.shape)

#     print(source.T.shape)
#     dot = torch.matmul(estimate_source, source.T)  # B , B
#     s_target = torch.matmul(dot, source) / (source_energy + eps)  # B , T
#     e_noise = estimate_source - source
#     snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps) + eps)  # B , 1
#     lo = 0 - torch.mean(snr)

def compute_musdb_metrics():
    SDR = {}
    for key in Eval_config["instrument_list"]:
        SDR[key] = 0
    audio = FFMPEGProcessAudioAdapter()
    pretrain_model = Eval_config["pretrain_model"]    
    model = create_model("test")
    model.to(Eval_config["device"])
    model.load_state_dict(torch.load(pretrain_model, map_location=Eval_config["device"]))
    model.eval()

    songs = glob(os.path.join(Eval_config["audio_root_musdb_test"],'*/mixture.wav'))
    
    for song in tqdm(songs[0:Eval_config["eval_length"]]):
        wave = audio.load(song,sample_rate=Eval_config["sample_rate"], channels = Eval_config["channels"])
        if Model_config["input_type"] == "F":
            results = F_separate(model,wave)
        if Model_config["input_type"] == "T":
            results = T_separate(model,wave)

        foldername = os.path.basename(os.path.dirname(song))
        for key, value in results.items():
            track = song.replace("mixture",key)
            wave = audio.load(track,sample_rate=Eval_config["sample_rate"], channels = Eval_config["channels"])
            wave = wave[500:-500]
            value = value[500:-500]
            (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(wave.T,value.T)
            SDR[key] = SDR[key] + sdr
            print(sdr)
    for key in SDR:
        SDR[key] = SDR[key]/Eval_config["eval_length"]
    return SDR


def main():
    SDR = compute_musdb_metrics()
    print(SDR)
    log_file_path = os.path.join(Eval_config["save_eval_path"], "eval" + General_config["task_name"]  + ".txt")
    os.system("touch " + log_file_path)
    log_file = open(log_file_path, "w")

    for key in SDR:
        log_file.write(key + ": " + str(SDR[key])+"\n")
    
if __name__ == '__main__':
    main()