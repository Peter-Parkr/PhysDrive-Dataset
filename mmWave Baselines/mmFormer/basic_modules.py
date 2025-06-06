import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict


class Conv2dBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, is_padding=1):
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(Conv2dBN, self).__init__(OrderedDict([
            ('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=padding, groups=groups, bias=True)),
            ('bn', nn.BatchNorm2d(out_channels))
        ]))

class Conv2dBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, is_padding=1):
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(Conv2dBNReLU, self).__init__(OrderedDict([
            ('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('leakyrelu', nn.LeakyReLU(0.3, inplace=True))   
        ]))

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, is_padding=1):
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(Conv2dReLU, self).__init__(OrderedDict([
            ('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('leakyrelu', nn.LeakyReLU(0.3, inplace=True))   
        ]))

class Conv2dINReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, is_padding=1):
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(Conv2dINReLU, self).__init__(OrderedDict([
            ('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('in', nn.InstanceNorm2d(out_channels)),
            ('leakyrelu', nn.LeakyReLU(0.3, inplace=True))   
        ]))

class ConvTrans1dBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, groups=1, dialation=1):
        # output_padding=1 and padding= 1 to maintain the shape
        super(ConvTrans1dBN, self).__init__(OrderedDict([
            ('convtrans1d', nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, 
                                                output_padding, groups, dilation=dialation)),
            ('bn', nn.BatchNorm1d(out_channels))                        
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

class ConvTrans2dBNReLu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, groups=1, dialation=1):
        # output_padding=1 and padding= 1 to maintain the shape
        super(ConvTrans2dBNReLu, self).__init__(OrderedDict([
            ('convtrans2d', nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, 
                                                output_padding, groups, dilation=dialation)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('LeakyReLu', nn.LeakyReLU(0.3, inplace=True))                        
        ]))

class Conv1dBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, is_padding=1):
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(Conv1dBNReLU, self).__init__(OrderedDict([
            ('conv1d', nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, 
                                                groups)),
            ('bn', nn.BatchNorm1d(out_channels)),
            ('leakyrelu', nn.LeakyReLU(0.3, inplace=True))                        
        ]))

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, use_1x1conv=True):
        super(ResBlock, self).__init__()
        self.in_ch = int(in_channels)
        self.out_ch = int(out_channels)
        self.k = kernel_size
        self.stride = stride
        self.padding = int(padding)

        self.conv1 = Conv2dBN(self.in_ch, self.out_ch, self.k, 1, is_padding=self.padding)
        self.conv2 = Conv2dBN(self.out_ch, self.out_ch, self.k, self.stride, is_padding=self.padding)

        if use_1x1conv:
            self.conv3 = Conv2dBN(self.in_ch, self.out_ch, 1, self.stride, is_padding=self.padding)
        else:
            self.conv3 = None
    def forward(self, x):
        # x1 = self.conv2(F.relu(self.conv1(x)))
        x1 = self.conv2(F.relu(self.conv1(x)))
        if self.conv3:
            x = self.conv3(x)
        out = x+x1
        return out

class ResBlockBNReLu(nn.Sequential):
    def __init__(self, in_channels, out_channels,  kernel_size, stride, padding, use_1x1conv=True):
        super(ResBlockBNReLu, self).__init__(OrderedDict([
            ('conv2d', ResBlock(in_channels, out_channels, kernel_size, stride,
                                padding=padding, use_1x1conv=use_1x1conv)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('LeakyReLu', nn.LeakyReLU(0.3, inplace=True))    
        ]))

class ConvAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.query = Conv2dBNReLU(in_ch, out_ch, kernel_size=(1, 5))
        self.key = Conv2dBNReLU(in_ch, out_ch, kernel_size=(1, 5))
        self.value = Conv2dBNReLU(in_ch, out_ch, kernel_size=(1, 5))
        self.lepe = Conv2dBNReLU(in_ch, out_ch, kernel_size=(1, 5))
    def forward(self, x):
        q = self.query(x)   # [B, c, 1, 1024]
        k = self.key(x)     # # [B, c, 1, 1024]
        attn_matrix = q.transpose(-2, -1) @ k
        attn_matrix = nn.functional.softmax(attn_matrix, dim=-1, dtype=attn_matrix.dtype)
        v = self.value(x)
        pe = self.lepe(x)
        x = (v @ attn_matrix) + pe
        return x


class Conv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, is_padding=1):
        if is_padding:
            if not isinstance(kernel_size, int):
                padding = [(i - 1) // 2 for i in kernel_size]
            else:
                padding = (kernel_size - 1) // 2
        else:
            padding = 0
        super(Conv, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False))
        ]))