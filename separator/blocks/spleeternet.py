from torch import nn
import torch
from .unet import UNet
import numpy as np



def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    

class SpleeterNet(nn.Module):
    def __init__(self, instruments, output_mask_logit, phase='train'):
        super(SpleeterNet, self).__init__()
        for instrument in instruments:
            sub_model = UNet()
            self.__setattr__(instrument, sub_model)

        self.instruments = instruments
        self.output_mask_logit = output_mask_logit
        self.phase = phase

        self.apply(_weights_init)

    def forward(self, x):
        result = {}

        outputs = []
        for instrument in self.instruments:
            output = self.__getattr__(instrument)(x)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=-1)

        if self.output_mask_logit:
            mask = torch.softmax(outputs, dim=-1)
        else:
            mask = torch.sigmoid(outputs)

        for idx, instrument in enumerate(self.instruments):
            if self.phase == 'train':
                try:
                    result[instrument] = mask[..., idx] * x
                except:
                    mask = mask[:,:,:,0:len(x[0][0][0]),:]
                    x = x[:,:,:,0:len(mask[0][0][0])]
                    result[instrument] = mask[..., idx] * x
            else:
                result[instrument] = mask[..., idx]
        return result
