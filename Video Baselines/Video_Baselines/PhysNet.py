""" PhysNet
We repulicate the net pipeline of the orginal paper, but set the input as diffnormalized data.
orginal source:
Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks
British Machine Vision Conference (BMVC)} 2019,
By Zitong Yu, 2019/05/05
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019
"""

import math
import pdb

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.pool = nn.AdaptiveAvgPool3d((frames // 4, 1, 1))
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        self.hr_head = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(frames // 4, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=(1, 1, 1))
        )

        self.rr_head = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(frames // 4, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=(1, 1, 1))
        )

        self.spo_head = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(frames // 4, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=(1, 1, 1))
        )

        self.upsample_resp = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2_resp = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        # self.bn = nn.BatchNorm3d(3)

        self.ConvBlock10_resp = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def count_parameters(self, grad):
        return sum(p.numel() for p in self.parameters() if p.requires_grad == grad)

    def calculate_training_parameter_ratio(self):
        trainable_param_num = self.count_parameters(True)
        other_param_num = self.count_parameters(False)
        print("Non-trainable parameters:", other_param_num)
        print("Trainable parameters (M):", trainable_param_num / (1024 ** 2))

        ratio = trainable_param_num / (other_param_num + trainable_param_num)
        # final_ratio = (ratio / (1 - ratio))
        print("Ratio:", ratio)

        return ratio

    def diff_normalize_data(self, data):
        """Calculate discrete difference in video data along the time-axis and normalize by its standard deviation."""
        # Calculate the discrete difference along the time-axis
        diff_data = torch.diff(data, dim=1)

        # Normalize by the sum of consecutive frames
        sum_data = data[:, :-1] + data[:, 1:] + 1e-7
        diffnormalized_data = diff_data / sum_data

        # Normalize by the standard deviation
        std_dev = torch.std(diffnormalized_data, unbiased=False)
        diffnormalized_data = diffnormalized_data / std_dev

        # Add padding of zeros to match the original temporal dimension
        padding = torch.zeros((data.shape[0], 1, *data.shape[2:]), dtype=torch.float32, device=data.device,
                              requires_grad=True)
        diffnormalized_data = torch.cat((diffnormalized_data, padding), dim=1)

        # Handle NaN values
        diffnormalized_data[torch.isnan(diffnormalized_data)] = 0

        return diffnormalized_data

    def forward(self, x):  # Batch_size*[3, T, 128,128]
        # x = x.permute(0, 3, 4, 1, 2)  # from [Heith, With, T, 3] to [T, 3, Heith, With]
        x = x.permute(0, 4, 3, 1, 2)
        # x = self.diff_normalize_data(x)
        # x = torch.diff(x, dim=2)
        # x = self.bn(x)
        # x = x.permute(0, 2, 1, 3, 4)  # [3, T, Heith, With]
        # x_visual = x
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        # x [32, T/2, 32,32]    Temporal halve
        x = self.MaxpoolSpaTem(x_visual6464)

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        em = self.ConvBlock9(x)  # x [64, T/4, 8, 8]

        # HR
        x = self.pool(em)  # x [64, T/4, 1, 1]
        hr = self.hr_head(x)
        #
        # # RR
        # x = self.pool(em)  # x [64, T/4, 1, 1]
        # rr = self.rr_head(x)
        #
        # # spo
        # x = self.pool(em)  # x [64, T/4, 1, 1]
        # spo = self.spo_head(x)

        # BVP
        x = self.upsample(em)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]
        # x [64, T, 1,1]    -->  groundtruth left and right - 7
        x = self.poolspa(x)
        bvp = self.ConvBlock10(x)  # x [1, T, 1,1]
        bvp = bvp.view(-1, length)

        # # Resp
        # x = self.upsample_resp(em)  # x [64, T/2, 8, 8]
        # x = self.upsample2_resp(x)  # x [64, T, 8, 8]
        # # x [64, T, 1,1]    -->  groundtruth left and right - 7
        # x = self.poolspa(x)
        # resp = self.ConvBlock10_resp(x)  # x [1, T, 1,1]
        # resp = resp.view(-1, length)

        # return bvp, hr.view(-1, 1), spo.view(-1, 1), rr.view(-1, 1), resp  # x_visual, x_visual3232, x_visual1616
        return bvp, hr.view(-1, 1), None, None, None
