import torch
import torch.nn as nn
import os
import sys
import torch.nn.functional as F
from conv_TasNet.models.DCCRN.conv_stft import ConvSTFT, ConviSTFT 

from conv_TasNet.models.DCCRN.complexnn import ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm



class DCCRN(nn.Module):
    def __init__(self,
                 rnn_layer=2, rnn_hidden=256,
                 win_len=400, hop_len=100, fft_len=512, win_type='hanning',
                 use_clstm=True, use_cbn=False, masking_mode='E',
                 kernel_size=5, kernel_num=(32, 64, 128, 256, 256, 256)
                 ):
        super(DCCRN, self).__init__()
        self.rnn_layer = rnn_layer
        self.rnn_hidden = rnn_hidden

        self.win_len = win_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.win_type = win_type

        self.use_clstm = True
        self.use_cbn = use_cbn
        self.masking_mode = masking_mode

        self.kernel_size = kernel_size
        self.kernel_num = (2,) + kernel_num

        self.stft = ConvSTFT(self.win_len, self.hop_len, self.fft_len, self.win_type, 'complex', fix=True)
        self.istft = ConviSTFT(self.win_len, self.hop_len, self.fft_len, self.win_type, 'complex', fix=True)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                        self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

# LSTM
        if self.use_clstm:
            rnns = []
            for idx in range(rnn_layer):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim * self.kernel_num[-1] if idx == 0 else self.rnn_hidden,
                        hidden_size=self.rnn_hidden,
                        batch_first=False,
                        projection_dim=hidden_dim * self.kernel_num[-1] if idx == rnn_layer - 1 else None
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_hidden,
                num_layers=2,
                dropout=0.0,
                batch_first=False
            )
            self.transform = nn.Linear(self.rnn_hidden, hidden_dim * self.kernel_num[-1])
# 
        
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            self.kernel_num[idx - 1]),
                        nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        )
                    )
                )
    # LSTM
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()
    # 

    def forward(self, x):
        x = x.to(torch.float32)
        stft = self.stft(x)
        # print("stft:", stft.size())
        real = stft[:, :self.fft_len // 2 + 1]
        imag = stft[:, self.fft_len // 2 + 1:]
        # print("real imag:", real.size(), imag.size())
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        spec_phase = torch.atan2(imag, real)
        spec_complex = torch.stack([real, imag], dim=1)[:, :, 1:]  # B,2,256
        # print("spec", spec_mags.size(), spec_phase.size(), spec_complex.size())
        out = spec_complex
        
        # print("1")

        encoder_out = []
        for idx, encoder in enumerate(self.encoder):
            out = encoder(out)
            # print("encoder out:", out.size())
            encoder_out.append(out)
            # print("2")
        B, C, D, T = out.size()
        out = out.permute(3, 0, 1, 2)
        # LSTM
        
        if self.use_clstm:
            out = out.to(torch.float32)
            r_rnn_in = out[:, :, :C // 2]
            i_rnn_in = out[:, :, C // 2:]
            r_rnn_in = torch.reshape(r_rnn_in, [T, B, C // 2 * D])
            i_rnn_in = torch.reshape(i_rnn_in, [T, B, C // 2 * D])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(r_rnn_in, [T, B, C // 2, D])
            i_rnn_in = torch.reshape(i_rnn_in, [T, B, C // 2, D])
            out = torch.cat([r_rnn_in, i_rnn_in], 2)
            out = out.to(torch.float16)

        else:
            out = out.to(torch.float32)
            out = torch.reshape(out, [T, B, C * D])
            out, _ = self.enhance(out)
            out = self.transform(out)
            out = torch.reshape(out, [T, B, C, D])
            out = out.to(torch.float16)
            # 
        # print("3")
        out = out.permute(1, 2, 3, 0)
        for idx in range(len(self.decoder)):
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
            out = out[..., 1:]
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])
        if self.masking_mode == 'E':
            mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
            real_phase = mask_real / (mask_mags + 1e-8)
            imag_phase = mask_imag / (mask_mags + 1e-8)
            mask_phase = torch.atan2(
                imag_phase,
                real_phase
            )
            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags * spec_mags
            est_phase = spec_phase + mask_phase
            real = est_mags * torch.cos(est_phase)
            imag = est_mags * torch.sin(est_phase)
        elif self.masking_mode == 'C':
            real = real * mask_real - imag * mask_imag
            imag = real * mask_imag + imag * mask_real
        elif self.masking_mode == 'R':
            real = real * mask_real
            imag = imag * mask_imag

        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)
        out_wav = torch.squeeze(out_wav, 1)
        out_wav = out_wav.clamp_(-1, 1)
        out_wav = out_wav.unsqueeze(1)
        return out_wav

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose1d):
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
    def __init__(self, instruments, output_mask_logit, phase='train', keras=True):
        super(SpleeterNet, self).__init__()
        for instrument in instruments:
            sub_model = DCCRN()
            self.__setattr__(instrument, sub_model)

        self.instruments = instruments
        self.output_mask_logit = True
        self.phase = phase

        self.apply(_weights_init)

    def forward(self, x):
        result = {}

        outputs = []
        for instrument in self.instruments:
            output = self.__getattr__(instrument)(x)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=-1)

        # if self.output_mask_logit:
        #     mask = torch.softmax(outputs, dim=-1)
        # else:
        #     mask = torch.sigmoid(outputs)

        for idx, instrument in enumerate(self.instruments):
            result[instrument] = outputs[..., idx]
            # if self.phase == 'train':
            #     # try:
            #     result[instrument] = outputs[..., idx]
            #     # except:
            #     #     # mask = mask[:,:,:,0:len(x[0][0][0]),:]
            #     #     # x = x[:,:,:,0:len(mask[0][0][0])]
            #     #     result[instrument] = mask[..., idx] * x
            # else:
            #     result[instrument] = mask[..., idx]
        return result




# Print Parameters
from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
if __name__ == '__main__':
    model = SpleeterNet(["vocals","drums","bass","other"], True, phase='train', keras=False)
    # model = DCCRN()
    model.eval()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    count_parameters(model)