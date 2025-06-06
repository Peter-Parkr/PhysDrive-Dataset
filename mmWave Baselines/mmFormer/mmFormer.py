import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
# -*- coding: UTF-8 -*-
import sys
from torchvision import models
from mmFormer.basic_modules import *
from torch.nn import InstanceNorm2d, BatchNorm2d
import mmFormer.transformer as My_TF    # add positional encoding at the attention operator v

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, mlp_ratio=4., nhead=16, num_layers=12):
        super().__init__()
        dim_feedforward = int(d_model * mlp_ratio)
        encoder_layer = My_TF.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0, activation='gelu',
                                                      batch_first=True, norm_first=True)
        self.tf_encoder = My_TF.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = torch.squeeze(x, dim=2)
        x = torch.transpose(x, -2, -1)  # [B, seq, #features]
        x = self.tf_encoder(x)
        x = torch.transpose(x, -2, -1)  # [B, #features, seq]
        x = torch.unsqueeze(x, dim=2)
        return x


class Up_Conv(nn.Module):
    def __init__(self, C, kernel_size, stride, output_padding, F_norm=BatchNorm2d):
        super().__init__()
        if not isinstance(kernel_size, int):
            padding = [(kernel_size[i] - stride[i] + output_padding[i]
                        ) // 2 for i in range(len(kernel_size))]
        else:
            padding = (kernel_size - stride + output_padding) // 2
        self.Up = nn.Sequential(
            nn.ConvTranspose2d(C, C//2, kernel_size, stride,
                               padding, output_padding, bias=False),
            F_norm(C//2),
            nn.LeakyReLU(0.3, inplace=True)
        )

    def forward(self, x, r):
        return torch.cat((self.Up(x), r), 1)

class UNet_Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, F_norm=BatchNorm2d, dropout=0.0, last_conv=False):
        super().__init__()
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        if last_conv == False:
            self.layer = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride=1, padding=padding, bias=False),
                # nn.BatchNorm2d(C_out),
                # SwitchNorm2d(C_out),
                F_norm(C_out),
                # F_norm,
                nn.Dropout(dropout),
                nn.LeakyReLU(0.3, inplace=True),

                nn.Conv2d(C_out, C_out, kernel_size,
                          stride=1, padding=padding, bias=False),
                # nn.BatchNorm2d(C_out),
                # SwitchNorm2d(C_out),
                F_norm(C_out),
                # F_norm,
                nn.Dropout(dropout),
                nn.LeakyReLU(0.3, inplace=True)
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride=1, padding=padding, bias=False),
                F_norm(C_out),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv2d(C_out, C_out, kernel_size,
                          stride=1, padding=padding, bias=False),

            )

    def forward(self, x):
        return self.layer(x)

class Down_Conv(nn.Module):
    def __init__(self, C, kernel_size, stride, F_norm=BatchNorm2d):
        super().__init__()
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, kernel_size, stride=stride, padding=padding, bias=False),
            F_norm(C),
            nn.LeakyReLU(0.3, inplace=True)
        )

    def forward(self, x):
        return self.Down(x)

class Rangebin_Linear(nn.Module):
    def __init__(self, num_bins=4, hidden_dim=16):
        super().__init__()
        self.num_bins = num_bins
        self.rangebin_weight = nn.Sequential(
            nn.Linear(num_bins, hidden_dim, bias=False),
            nn.LeakyReLU(0.3, inplace=False),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        
    def forward(self, x):
        return self.rangebin_weight(x)
        
        


class UNet_Transformer_BatchNorm_Attn_Cut_KD_MyTF_Multi_Expert(nn.Module):
    def __init__(self, num_range_bins=8, num_doppler=8):
        super().__init__()
      


        self.iq_conv = nn.Sequential(
            Conv(2, 32, 1, 1),
            BatchNorm2d(32),
            nn.LeakyReLU(0.3)
        )  # [B, 32, 1, 4096]

        self.doppler_layer = nn.Sequential(
                            nn.Linear(num_doppler, num_doppler*4),
                            nn.LeakyReLU(0.3),
                            nn.Linear(num_doppler*4, 1),
                            nn.LeakyReLU(0.3),
                        )
        self.rangebin_weight0 = Rangebin_Linear(num_range_bins, num_range_bins*4)
        self.C1 = UNet_Conv(32, 64, (1, 11), F_norm= BatchNorm2d) 
        self.rangebin_weight1 = Rangebin_Linear(num_range_bins, num_range_bins*4)
        self.D1 = Down_Conv(64, (1, 11), (1, 2), F_norm=BatchNorm2d)  
        self.C2 = UNet_Conv(64, 128, (1, 11), F_norm=BatchNorm2d)  
        self.rangebin_weight2 = Rangebin_Linear(num_range_bins, num_range_bins*4)
        self.D2 = Down_Conv(128, (1, 11), (1, 2), F_norm=BatchNorm2d) 
        self.C3 = UNet_Conv(128, 256, (1, 11), F_norm=BatchNorm2d)
        self.rangebin_weight3 = Rangebin_Linear(num_range_bins, num_range_bins*4)
        self.D3 = Down_Conv(256, (1, 11), (1, 2), F_norm=BatchNorm2d)  
        self.C4 = UNet_Conv(256, 512, (1, 11), F_norm=BatchNorm2d)
        self.transformer = TransformerEncoder(
            d_model=256, mlp_ratio=4., nhead=8, num_layers=12)  

        self.U1 = Up_Conv(256, (1, 11), (1, 2), (0, 1), F_norm=BatchNorm2d)  
        self.C5 = UNet_Conv(256, 128, (1, 11), F_norm=BatchNorm2d)  
        self.U2 = Up_Conv(128, (1, 11), (1, 2), (0, 1), F_norm=BatchNorm2d)  
        self.C6 = UNet_Conv(128, 64, (1, 11), F_norm=BatchNorm2d)  
        self.U3 = Up_Conv(64, (1, 11), (1, 2), (0, 1), F_norm=BatchNorm2d)  
        self.C7 = UNet_Conv(64, 32, (1, 11), F_norm=BatchNorm2d)  


        self.ecg = nn.Sequential(
            nn.Conv1d(32, 64, 11, 1, 5, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3),
            nn.Conv1d(64, 32, 11, 1, 5, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3),
            nn.Conv1d(32, 16, 11, 1, 5, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3),
            nn.Conv1d(16, 1, 11, 1, 5, bias=False)
        )
        self.resp = nn.Sequential(
            nn.Conv1d(32, 64, 11, 1, 5, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.3),
            nn.Conv1d(64, 32, 11, 1, 5, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3),
            nn.Conv1d(32, 16, 11, 1, 5, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3),
            nn.Conv1d(16, 1, 11, 1, 5, bias=False)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.hr = nn.Linear(32, 1)
        self.rr = nn.Linear(32, 1)



    def forward(self, x):
        ### x: [B, 200, 2, 8, 16, 8]
        ### x: [B, num_frames, Real/Imag, num_doppler, num_angles, num_rangebins]
        # Calculate energy for each angle in the input
        # x shape: [B, num_frames, Real/Imag, num_doppler, num_angles, num_rangebins]
        
        # Calculate energy (sum of squares of real and imaginary parts)
        energy = x[:, :, 0, :, :, :]**2 + x[:, :, 1, :, :, :]**2  # [B, num_frames, num_doppler, num_angles, num_rangebins]
        
        # Sum across all dimensions except batch and angles
        energy_per_angle = energy.sum(dim=(1, 2, 4))  # [B, num_angles]
        
        # Find the angle index with the highest energy for each sample in the batch
        _, max_energy_angle_idx = torch.max(energy_per_angle, dim=1)  # [B]
        
        # Create a batch indexer
        batch_size = x.shape[0]
        
        # Extract data for each sample using its specific angle index
        x_selected = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3], x.shape[5], device=x.device)
        for i in range(batch_size):
            x_selected[i] = x[i, :, :, :, max_energy_angle_idx[i], :]
        
        # [B, num_frames, Real/Imag, num_doppler, num_rangebins]
        x = x_selected

        x = x.permute(0, 1, 2, 4, 3)
        x = self.doppler_layer(x).squeeze(dim=-1) # [B, num_frames, Real/Imag, num_rangebins]
        x = x.permute(0, 2, 3, 1) # [B, Real/Imag, num_rangebins, num_frames]
        
        x = self.iq_conv(x)  # [B, 32, num_rangebins, 200] 
        attn_cut0 = self.rangebin_weight0(x.transpose(-1, -2)).transpose(-1, -2)
        DS1 = self.D1(self.C1(x))  # [B, 64, num_rangebins, 100]
        attn_cut1 = self.rangebin_weight1(DS1.transpose(-1, -2)).transpose(-1, -2)
        DS2 = self.D2(self.C2(DS1))  # [B, 128, num_rangebins, 50]
        attn_cut2 = self.rangebin_weight2(DS2.transpose(-1, -2)).transpose(-1, -2)
        DS3 = self.D3(self.C3(DS2))  # [B, 256, num_rangebins, 25]
        attn_cut3 = self.rangebin_weight3(DS3.transpose(-1, -2)).transpose(-1, -2)
        
        DS3 = self.transformer(attn_cut3)

        UP1 = self.C5(self.U1(DS3, attn_cut2))  # [B, 128, 1, 50]

        UP2 = self.C6(self.U2(UP1, attn_cut1))  # [B, 64, 1, 100]
        UP3 = self.C7(self.U3(UP2, attn_cut0))  # [B, 32, 1, 200]
        UP3 = torch.squeeze(UP3, dim=2)
        
        ecg = self.ecg(UP3)
        resp = self.resp(UP3)
        hr = self.hr(self.pool(UP3).squeeze(-1))
        rr = self.rr(self.pool(UP3).squeeze(-1))
        return ecg, resp, hr, rr


if __name__ == '__main__':
    x = torch.randn(10, 200, 2, 8, 16, 8)
    model = UNet_Transformer_BatchNorm_Attn_Cut_KD_MyTF_Multi_Expert(num_range_bins=8, num_doppler=8)
    output = model(x)
    print(output.shape)
