import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
# -*- coding: UTF-8 -*-
import sys
from torchvision import models
from torch.nn import InstanceNorm2d, BatchNorm2d
from typing import Any
from torch import Tensor
from torch.nn.functional import mse_loss

def kl_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()

def alignment_loss(real_mu, real_logvar, imag_mu, imag_logvar):
    return mse_loss(real_mu, imag_mu, reduction='mean') + mse_loss(real_logvar, imag_logvar, reduction='mean')

class Conv1dBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, is_padding=1, dilation=1):
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [dilation*(i - 1) // 2 for i in kernel_size]
            else:
                padding = dilation*(kernel_size - 1) // 2
        else:
            padding = 0
        super(Conv1dBNReLU, self).__init__(OrderedDict([
            ('conv1d', nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                                                groups=groups, dilation=dilation)),
            ('bn', nn.BatchNorm1d(out_channels)),
            ('leakyrelu', nn.LeakyReLU(0.3, inplace=True))                        
        ]))

class ConvTrans1dBNReLu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, groups=1, dialation=1):
        # output_padding=1 and padding= 1 to maintain the shape
        super(ConvTrans1dBNReLu, self).__init__(OrderedDict([
            ('convtrans1d', nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, 
                                                output_padding, groups, dilation=dialation)),
            ('bn', nn.BatchNorm1d(out_channels)),
            ('LeakyReLu', nn.LeakyReLU(0.3, inplace=True))                        
        ]))


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        pool_kernel_size=None,
    ):
        super().__init__()
        self.convs = nn.ModuleList()

        # conv 1x1
        self.conv1 = Conv1dBNReLU(in_channels, out_channels, kernel_size=1, stride=1, is_padding=True)

        # atrous convs
        self.aconv1 = Conv1dBNReLU(in_channels, out_channels, kernel_size=3, stride=1, is_padding=True, dilation=2)

        self.aconv2 = Conv1dBNReLU(in_channels, out_channels, kernel_size=3, stride=1, is_padding=True, dilation=4)

        self.aconv3 = Conv1dBNReLU(in_channels, out_channels, kernel_size=3, stride=1, is_padding=True, dilation=6)

        # pooling
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel_size, stride=1, padding=pool_kernel_size//2)

        self.conv = Conv1dBNReLU(out_channels*5, out_channels, kernel_size=1, stride=1, is_padding=True)

    def forward(self, x):
        # conv 1x1
        x1 = self.conv1(x)

        # atrous convs
        x2 = self.aconv1(x)
        x3 = self.aconv2(x)
        x4 = self.aconv3(x)

        # pooling
        x5 = self.pool(x)

        # concatenate
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.conv(out)

        return out
        
    
class SeparableConv1d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            is_padding: bool = True,
    ) -> None:
        super(SeparableConv1d, self).__init__()
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        self.conv1 = nn.Conv1d(in_channels, in_channels, groups=in_channels, bias=False, 
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, 
                                   padding=0, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.3, inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.relu(out)
        return out
    

class Encoder(nn.Module):
    def __init__(self, num_ranges=8):
        super().__init__()
        self.conv0 = SeparableConv1d(num_ranges, 16, kernel_size=3, stride=1, is_padding=True)
        self.conv1 = SeparableConv1d(16, 32, kernel_size=5, stride=1, is_padding=True)
        self.conv2 = SeparableConv1d(32, 64, kernel_size=9, stride=1, is_padding=True)
        self.conv3 = SeparableConv1d(64, 128, kernel_size=17, stride=1, is_padding=True)
        self.conv4 = SeparableConv1d(128, 256, kernel_size=33, stride=1, is_padding=True)
        self.conv5 = SeparableConv1d(256, 512, kernel_size=65, stride=1, is_padding=True)

        self.aspp = ASPP(512, 512, pool_kernel_size=21)
        agg_dim = 16+128+512
        self.conv_mu = Conv1dBNReLU(agg_dim, agg_dim, kernel_size=1, stride=1, is_padding=True)
        self.conv_logvar = Conv1dBNReLU(agg_dim, agg_dim, kernel_size=1, stride=1, is_padding=True)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, if_train=True):
        # x: [B, num_rangebins=8, num_frames=200]
        x0 = self.conv0(x) # [B, 16, 200]
        x1 = self.conv1(x0) # [B, 32, 200]
        x2 = self.conv2(x1) # [B, 64, 200]
        x3 = self.conv3(x2) # [B, 128, 200]
        x4 = self.conv4(x3) # [B, 256, 200]
        x5 = self.conv5(x4) # [B, 512, 200]

        x6 = self.aspp(x5) # [B, 512, 200]

        out = torch.cat((x0, x3, x6), dim=1) # [B, 16+128+512, 200]
        mu = self.conv_mu(out) # [B, 16+128+512, 200]
        logvar = self.conv_logvar(out)
        if if_train:
            out = self.sample(mu, logvar)
        else:
            out = mu

        return out, mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvTrans1dBNReLu(in_channels=16+128+512, out_channels=512,
                                       kernel_size=65, stride=1, padding=32, output_padding=0)
        
        self.conv2 = ConvTrans1dBNReLu(in_channels=512, out_channels=256,
                                       kernel_size=33, stride=1, padding=16, output_padding=0)
        self.conv3 = ConvTrans1dBNReLu(in_channels=256, out_channels=128,
                                       kernel_size=17, stride=1, padding=8, output_padding=0)
        self.conv4 = ConvTrans1dBNReLu(in_channels=128, out_channels=64,
                                       kernel_size=9, stride=1, padding=4, output_padding=0)
        self.conv5 = ConvTrans1dBNReLu(in_channels=64, out_channels=32,
                                       kernel_size=5, stride=1, padding=2, output_padding=0)
        self.conv6 = ConvTrans1dBNReLu(in_channels=32, out_channels=32,
                                       kernel_size=3, stride=1, padding=1, output_padding=0)
        
        self.ecg = nn.Sequential(
            nn.Conv1d(32, 32, 11, 1, 5, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.3),
            nn.Conv1d(32, 16, 11, 1, 5, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.3),
            nn.Conv1d(16, 1, 11, 1, 5, bias=False)
        )

        self.resp = nn.Sequential(
            nn.Conv1d(32, 32, 11, 1, 5, bias=False),
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
        ### x: [B, 16+128+512, 200]
        x = self.conv1(x) # [B, 512, 200]
        x = self.conv2(x) # [B, 256, 200]
        x = self.conv3(x) # [B, 128, 200]
        x = self.conv4(x) # [B, 64, 200]
        x = self.conv5(x) # [B, 32, 200]
        x = self.conv6(x) # [B, 32, 200]

        ecg = self.ecg(x)
        resp = self.resp(x)
        hr = self.hr(self.pool(x).squeeze(-1))
        rr = self.rr(self.pool(x).squeeze(-1))

        return ecg, resp, hr, rr



class IQ_MVED(nn.Module):
    def __init__(self, num_doppler=8):
        super().__init__()
        self.doppler_layer = nn.Sequential(
                            nn.Linear(num_doppler, num_doppler*4),
                            nn.LeakyReLU(0.3),
                            nn.Linear(num_doppler*4, 1),
                            nn.LeakyReLU(0.3),
                        )
        
        self.encoder = Encoder(num_ranges=8)
        self.decoder = Decoder()


    def forward(self, x, if_train=True):
        ## x: [B, 200, 2, 8, 16, 8]
        ### x: [B, num_frames, Real/Imag, num_doppler, num_angles, num_rangebins]


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

        ### aggregate the doppler dimension
        x = x.permute(0, 1, 2, 4, 3)
        x = self.doppler_layer(x).squeeze(dim=-1) # [B, num_frames, Real/Imag, num_rangebins]
        x = x.permute(0, 2, 3, 1) # [B, Real/Imag, num_rangebins, num_frames]

        # Split the real and imaginary parts
        ### x_real: [B, num_rangebins=8, num_frames=200]
        ### x_imag: [B, num_rangebins=8, num_frames=200]
        x_real = x[:, 0, :, :]
        x_imag = x[:, 1, :, :]

        real_feats, real_mu, real_var = self.encoder(x_real, if_train=if_train)
        imag_feats, imag_mu, imag_var = self.encoder(x_imag, if_train=if_train)

        feats = real_feats + imag_feats

        ecg, resp, hr, rr = self.decoder(feats)

        dis_loss = kl_loss(real_mu, real_var) + kl_loss(imag_mu, imag_var)

        align_loss = alignment_loss(real_mu, real_var, imag_mu, imag_var)

        return ecg, resp, hr, rr, dis_loss, align_loss




if __name__ == "__main__":
    x = torch.randn(10, 200, 2, 8, 16, 8)
    if_train = False
    model = IQ_MVED(num_doppler=8)
    output = model(x, if_train=if_train)
    for i in output:
        print(i.shape)