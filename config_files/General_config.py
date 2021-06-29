import os

General_config = {
    'HOME' : '/data/user_yz/MSS/Music-Source-Separator/',
    'phase' : ['eval'],
    'model' : "Spleeter",
    'task_name' : "1_track",
}

HOME = General_config["HOME"]

Train_config = {
    'instrument_list': ['vocals','drums','bass','other'],
    'instrument_weight': [1,1,1,1],
    'train_dataset' : ["musdb"],   #available: "musdb", "extra"
    'log_path' : os.path.join(HOME, "log_files"),  #path of log files
    'audio_root_extra': os.path.join(HOME, "dataset/extra"), #path of extra dataset
    'audio_root_musdb_train': os.path.join(HOME, "dataset/musdb18/train"),  #path of musdb dataset train folder
    'cache_root_extra': os.path.join(HOME, "data_files/cache_extra"),  #path to save extra dataset STFT cache
    'cache_root_musdb_train_F': os.path.join(HOME, "data_files/cache_musdb/train"),  #path to save musdb dataset STFT cache
    'cache_root_musdb_train_T': os.path.join(HOME, "data_files/cache_musdb_T/train"),  #path to save musdb dataset time domain cache
    'save_directory': os.path.join(HOME, 'saved_model'),   #path to save models
    'pretrained_model': None, #path to pretrained model
    'save_interval': 20,     #interval betwee saving models
    'deterministic' : True,   #reproduce result
    'distrib' : False,       #TODO: in progress
    'train_length': None #train samples: if none, use the full dataset
}

Separate_config = {
    "audio_path": "/Users/Rain/Desktop/out2.wav",
    "sample_rate":44100, 
    "channels":1,
    # "output_path": os.path.join(HOME, "output_files"),
    "output_path": "/Users/Rain/Desktop/Music-Source-Separator/output_files",
    'pretrain_model': os.path.join(HOME, '/Users/Rain/Desktop/Music-Source-Separator/saved_model/_epoch_20_1.19.pth'),
    'instrument_list': ['vocals'],
    "device": "cpu"
}

Eval_config = {
    "eval_dataset" : ["musdb"],
    "metrics" : ['SDR'],
    'instrument_list': ['vocals','drums','bass','other'],
    'audio_root_musdb_test': os.path.join(HOME, "dataset/musdb18/train"),
    'pretrain_model': os.path.join(HOME, '/data/user_yz/MSS/Music-Source-Separator/saved_model/1_track/_sepoch_22_1.18.pth'),
    "device": "cuda",
    "folder_name": "spleeter_1_tracks",
    "sample_rate":44100, 
    "channels":1,
    "eval_length" : 2,
    "save_eval_path":  os.path.join(HOME, "log_files")

}
