import torch
from torch import nn
from .utils.keras_layer import Conv2dKeras


def _get_conv_activation_layer(name):
    if name == 'ReLU':
        return nn.ReLU()
    elif name == 'ELU':
        return nn.ELU(alpha=1)
    else:
        return nn.LeakyReLU(negative_slope=0.2)


def _get_deconv_activation_layer(name):
    if name == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=0.2)
    elif name == 'ELU':
        return nn.ELU(alpha=1)
    else:
        return nn.ReLU()


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn):
        super(BasicConv2d, self).__init__()
        kernel_size = 5
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation_fn = act_fn
        # 方便后续层可以直接获取输出通道数
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        y = self.bn(x)
        y = self.activation_fn(y)
        return x, y


class BasicDeconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn, with_dropout):
        super(BasicDeconv2d, self).__init__()
        kernel_size = 5
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                        stride=2, padding=2, output_padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation_fn = act_fn
        if with_dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = nn.Sequential()
        # 方便后续层可以直接获取输出通道数
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.activation_fn(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        # (-1, 2, 512, 1024)
        super(UNet, self).__init__()
        in_channels, out_channels = 1,1
        mask_channels = 1
        conv_activation = 'ELU'
        deconv_activation = 'ELU'

        setting = [
            # c, dropout
            [16, False],
            [32, False],
            [64, False],
            [128, True],
            [256, True],
            [512, True]
        ]

        # 下采样部分
        conv_layers = []
        conv_act_fn = _get_conv_activation_layer(conv_activation)
        for c, _ in setting:
            layer = BasicConv2d(in_channels, c, conv_act_fn)
            conv_layers.append(layer)
            in_channels = c
        self.conv_layers = nn.ModuleList(conv_layers)

        # 上采样部分
        deconv_layers = []
        deconv_act_fn = _get_deconv_activation_layer(deconv_activation)
        # ::-1 inverses setting, and we only want 256 to 16
        for c, d in setting[::-1][1:]:
            layer = BasicDeconv2d(in_channels, c, deconv_act_fn, d)
            deconv_layers.append(layer)
            # # 因为会 cat 通道
            in_channels = c * 2

        self.deconv_layers = nn.ModuleList(deconv_layers)
        # mask 部分
        self.mask_layer = BasicDeconv2d(in_channels, mask_channels, deconv_act_fn, False)
        in_channels = mask_channels

        # 输出部分
        self.last_layer = Conv2dKeras(in_channels, out_channels, 4, dilation=2,stride = 1)



    def forward(self, x):
        conv_layers = []
        
        # input
        y = x

        # convolution step 
        for layer in self.conv_layers:
            x, y = layer(y)
            conv_layers.append((x, y))

        # bottom
        y = x

        # deconvolution
        for layer, (x, _) in zip(self.deconv_layers, conv_layers[::-1][1:]):
            y = layer(y)            
            # y = y[:,:]

            # Unet concatenation
            # x is original raw input
            # y is the output from deconvlution
            try:
                y = torch.cat((x, y), dim=1)
            except:
                y = y[:,:,:,0:len(x[0][0][0])]
                x = x[:,:,:,0:len(y[0][0][0])]
                y = torch.cat((x, y), dim=1)

        mask = self.mask_layer(y)
        output = self.last_layer(mask)
        return output
