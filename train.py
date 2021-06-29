from torch.optim.lr_scheduler import MultiStepLR
from torch.cuda.amp import autocast as autocast
from torch.utils import data

import os
import numpy as np
import torch
from tqdm import tqdm
import random
import time

from config_files.General_config import General_config
from config_files.General_config import Train_config
from config_files.Model_config import Model_config

from separator.utils.create_models import create_model
from separator.utils.multi_loss import MultiLoss
from separator.utils.meter import AverageMeter

# default parameter
LAST_PATH = "first"

# GENERAL_SPECIFICATION
HOME  = General_config["HOME"]
MODEL = General_config["model"]
instrument_list =  Train_config["instrument_list"]
instrument_weight =  Train_config["instrument_weight"]
train_name = General_config["task_name"]

# TRAIN_SPECIFICATION
train_dataset =  Train_config["train_dataset"]
save_directory =  Train_config["save_directory"]
pretrained_model =  Train_config["pretrained_model"]
save_interval = Train_config["save_interval"]
deterministic = Train_config["deterministic"]
log_folder_path = Train_config["log_path"]


# MODEL_SPECIFICATION
accumulation_steps = Model_config["accumulation_steps"]
max_epoch = Model_config['max_epoch']
lr = Model_config['lr']
milestones = Model_config['milestones']
gamma = Model_config["gamma"]
weight_decay =  Model_config["weight_decay"]
thread_num = Model_config["thread_num"]
FP = Model_config["FP"]
batch_size = Model_config["batch_size"]
input_type = Model_config["input_type"]
loss_type = Model_config["loss_type"]

scaler = torch.cuda.amp.GradScaler()

def reproducible():
    seed = 2021
    import torch.backends.cudnn
    torch.manual_seed(seed)
    if deterministic == True:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
   
def get_data_loader():
    train_dataloader_extra = None
    train_dataloader_musdb = None

    if "extra" in train_dataset:
        if input_type == "F":
            from separator.dataset_loader.loader_STFT_extra import loader_STFT_extra
            train_dataset_extra = loader_STFT_extra()
            train_dataloader_extra = data.DataLoader(train_dataset_extra, batch_size=batch_size, shuffle=True,num_workers=thread_num, drop_last=True, pin_memory=True) 
    
    if "musdb" in train_dataset: 
        if input_type == "F":
            from separator.dataset_loader.loader_STFT_musdb import loader_STFT_musdb
            train_dataset_musdb = loader_STFT_musdb()
            train_dataloader_musdb = data.DataLoader(train_dataset_musdb, batch_size=batch_size, shuffle=True,num_workers=thread_num, drop_last=True, pin_memory=True)
        else:
            from separator.dataset_loader.loader_TD_musdb import loader_TD_musdb
            train_dataset_musdb = loader_TD_musdb()
            train_dataloader_musdb = data.DataLoader(train_dataset_musdb, batch_size=batch_size, shuffle=True,num_workers=thread_num, drop_last=True, pin_memory=True)

    return train_dataloader_extra,train_dataloader_musdb
    
def train_one_epoch(model, device, loader, optimizer, criterion):
    i = 0
    global scaler
    model.train()
    model.cuda()
    meters = dict()
    meters['loss'] = AverageMeter()
    meters.update({key: AverageMeter() for key in instrument_list})

    for spectrograms in tqdm(loader):
    
        for key in instrument_list:
            spectrograms[key] = spectrograms[key].to(device)
        spectrograms["mixture"] = spectrograms["mixture"].to(device)
             
        if FP == "16":
            with autocast():
                predict = model(spectrograms["mixture"].float())
                loss, sub_loss = criterion(predict, spectrograms)
        else:
            
            predict = model(spectrograms["mixture"].float())
            loss, sub_loss = criterion(predict, spectrograms)
                
        loss = loss/accumulation_steps

        if FP == "16":
            if((i+1)%accumulation_steps)==0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                i = 0
            else:
                scaler.scale(loss).backward()
                i = i+1
        else:
            if((i+1)%accumulation_steps)==0:
                loss.backward()
                optimizer.step() 
                optimizer.zero_grad() 
                i = 0
            else:
                loss.backward()
                i = i+1
            
        meters['loss'].update(loss.item(), batch_size)
        for key in sub_loss:
            meters[key].update(sub_loss[key].item(), batch_size)

    return meters


def train(model, device):
    log_name = train_name + "_" + str(time.time()) + "_log.txt"
    log_file_path = os.path.join(log_folder_path, log_name)
    os.system("touch " + log_file_path)
    log_file = open(log_file_path, "w")
    global LAST_PATH

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    train_folder = os.path.join(save_directory, train_name)
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if FP == "16":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,eps = 1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,eps = 1e-8)
    criterion = MultiLoss(instrument_list, instrument_weight,loss_type)
    scheduler = MultiStepLR(optimizer, milestones, gamma)
    train_loader,train_loader_musdb = get_data_loader()
    
    for i in range(1,max_epoch):
        if "extra" in train_dataset:
            start = time.time()
            # Train with custom dataset (TF domain)
            t_loss = train_one_epoch(model, device, train_loader_extra, optimizer, criterion)
            end = time.time()
            scheduler.step()
            msg = ''
            for key, value in t_loss.items():
                value = value.result()
                msg += f'{key}:{value:.4f}\t'
            msg += f'On extra: time:{(end - start):.1f}\tepoch:{i}'
            print(msg)
            log_file.write(msg+"\n")
            loss_str = "{:.2f}".format(t_loss['loss'].result())
    
        if "musdb" in train_dataset:
            start = time.time()
            t_loss = train_one_epoch(model, device, train_loader_musdb, optimizer, criterion)
            end = time.time()
            scheduler.step()
            msg = ''
            for key, value in t_loss.items():
                value = value.result()*accumulation_steps
                msg += f'{key}:{value:.4f}\t'
            msg += f'On musdb data: time:{(end - start):.1f}\tepoch:{i}'
            print(msg)
            log_file.write(msg+"\n")
            loss_str = "{:.2f}".format(t_loss['loss'].result())

        if i%save_interval == 0 and i >= save_interval:
            save_path = os.path.join(train_folder, 'epoch_' + str(i) + '_' + loss_str  + '.pth')
            torch.save(model.state_dict(), save_path)
            model.phase = 'train'

        else:
            save_path = os.path.join(train_folder, 'epoch_' + str(i) + '_' + loss_str  + '.pth')
            torch.save(model.state_dict(), save_path)
            model.phase = 'train'
            if LAST_PATH != "first":
                os.remove(LAST_PATH)
            LAST_PATH = save_path


def main():
    
    reproducible()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    model = create_model("train")

    if pretrained_model is not None:
        model.load_state_dict(torch.load(pretrained_model, map_location=device))
        print("pretrained model loaded: " + pretrained_model)

    train(model, device)

if __name__ == '__main__':
    main()
