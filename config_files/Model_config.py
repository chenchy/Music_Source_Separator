import os
from config_files.General_config import General_config

if General_config["model"] == "Spleeter":

    Model_config = {
        'loss_type' : "smooth_l1",
        'max_epoch': 300,
        'lr': 1e-3,
        'milestones': [75, 150,225,300],
        'gamma': 0.1,
        'weight_decay': 5e-4,
        'thread_num': 4,
        'accumulation_steps' : 1,
        'FP' : "32",
        
        'batch_size': 4,
        'sample_rate': 44100,
        'channels': 1,
        'frame_length': 4096,
        'frame_step': 1024,
        'segment_length': 512,
        'frequency_bins':1536,
        
        'activition':'sigmoid',
        'input_type': "F",
        'extra_parameter': {}
    }


if General_config["model"] == "D3Net":
    Model_config = {
        'loss_type' : "smooth_l1",
        'max_epoch': 400,
        'lr': 1e-3,
        'milestones': [100, 200,300,400],
        'gamma': 0.1,
        'weight_decay': 5e-4,
        'thread_num': 4,
        'accumulation_steps' : 1,
        'FP' : "16",
        
        'batch_size': 1,
        'sample_rate': 44100,
        'channels': 1,
        'frame_length': 4096,
        'frame_step': 1024,
        'segment_length': 512,
        'frequency_bins':1536,
        'frequency_bins_low':256,

        "in_channels" : 1, 
        "num_features" : {'low': 32, 'high': 8, 'full': 8}, 
        "growth_rate" : {'low': [3, 4, 5, 4, 3], 'high': [3, 4, 3], 'full': [3, 4, 3]},
        "bottleneck_channels" : 8,

        "kernel_size" : {'low': (3, 3), 'high': (3, 3), 'full': (3, 3)},
        "scale" : {'low': (2,2), 'high': (2,2), 'full': (2,2)},
        "depth" : {'low': [4, 4, 4, 3, 3], 'high': [4, 4, 3], 'full': [4, 4, 3]},
        "num_d3blocks" : {'low': 5, 'high': 3, 'full': 3},
        "num_d2blocks" : {'low': [2, 2, 2, 2, 2], 'high': [2, 2, 2], 'full': [2, 2, 2]},

        "kernel_size_d2block" : (3, 3),
        "growth_rate_d2block" : 1,
        "depth_d2block" : 2,
        "kernel_size_gated" : (3, 3),

        'activition':'sigmoid',
        'input_type': "F",
        'extra_parameter': {}
    }

elif General_config["model"] == "ConvTasNet":
    Model_config = {
        'loss_type' : "si_snr",
        'max_epoch': 400,
        'lr': 1e-3,
        'milestones': [100, 200,300,400],
        'gamma': 0.1,
        'weight_decay': 5e-4,
        'thread_num': 4,
        'accumulation_steps' : 1,
        'FP' : "32",
        
        'batch_size': 1,
        'sample_rate': 22050,
        'channels': 1,
        "frame_length": 7 * 22050,

        "batch_size": 1,
        "C": 1,
        "T": 64,
        "L":24, 
        "stride" : 12,
        "N" : 512,
        "H" : 512,
        "B" : 128,
        "Sc" : 128,
        "P" : 5,
        "R" :3,
        "X" :8,
        "sep_norm" : True,
        "enc_bases": 'trainable',
        "dec_bases" :'trainable',
        "enc_nonlinear" : None,
        "causal" : False,
        
        'activition':'sigmoid',
        'input_type': "T",
        'extra_parameter': {}
    }
