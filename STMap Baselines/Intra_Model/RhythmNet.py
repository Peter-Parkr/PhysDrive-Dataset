import torch
import torch.nn.functional as F
from torch import nn
import sys
from torchvision import models
import numpy as np
import utils
import basic_module
# from ZipLora import *

np.set_printoptions(threshold=np.inf)
sys.path.append('../..')
args = utils.get_args()

class BasicBlock(nn.Module):
    def __init__(self, inplanes, out_planes, stride=2, downsample=1, Res=0, islast=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )
        if downsample == 1:
            self.down = nn.Sequential(
                nn.Conv2d(inplanes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        self.downsample = downsample
        self.Res = Res
        self.islast = islast

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.Res == 1:
            if self.downsample == 1:
                x = self.down(x)
            out += x
        if self.islast:
            return out
        else:
            return F.relu(out)

class RhythmNet(nn.Module):
    def __init__(self, pretrain='resnet18'):
        super(RhythmNet, self).__init__()
        if pretrain == 'resnet18':
            resnet = models.resnet18(pretrained=False)
            resnet.load_state_dict(torch.load('./pre_encoder/resnet18-5c106cde.pth'))
            modules = list(resnet.children())[:-1]
            self.resnet18 = nn.Sequential(*modules)
            # The resnet average pool layer before fc
            # self.avgpool = nn.AvgPool2d((10, 1))
            self.resnet_linear = nn.Linear(512, 1000)
            self.hr_fc_regression = nn.Linear(1000, 1)
            self.spo2_fc_regression = nn.Linear(1000, 1)
            self.rr_fc_regression = nn.Linear(1000, 1)
            self.gru_fc_out = nn.Linear(1000, 1)
            self.rnn = nn.GRU(input_size=1000, hidden_size=1000, num_layers=1)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.hr = nn.Linear(512, 1)
            self.spo = nn.Linear(512, 1)
            self.rf = nn.Linear(512, 1)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4



    def forward(self, input):
        hr_batched_output_per_clip = []
        spo2_batched_output_per_clip = []
        rr_batched_output_per_clip = []
        gru_input_per_clip = []
        hr_per_clip = []

        # Need to have so as to reflect a batch_size = 1 // if batched then comment out
        st_maps = input.unsqueeze(0)
        for t in range(st_maps.size(1)):
            # with torch.no_grad():
            x = self.resnet18(st_maps[:, t, :, :, :])
            # collapse dimensions to BSx512 (resnet o/p)
            x = x.view(x.size(0), -1)
            # output dim: BSx1 and Squeeze sequence length after completing GRU step
            em = self.resnet_linear(x)


            # Final regression layer for CNN features -> HR (per clip)
            hr = self.hr_fc_regression(em)
            # normalize HR by frame-rate: 25.0 for VIPL
            hr = hr * 30
            hr_batched_output_per_clip.append(hr.squeeze(0))
            # input should be (seq_len, batch, input_size)
            spo2 = self.spo2_fc_regression(em)
            # normalize HR by frame-rate: 25.0 for VIPL
            spo2 = spo2 * 30
            spo2_batched_output_per_clip.append(spo2.squeeze(0))

            rr = self.rr_fc_regression(em)
            # normalize HR by frame-rate: 25.0 for VIPL
            rr = rr * 30
            rr_batched_output_per_clip.append(rr.squeeze(0))

        # the features extracted from the backbone CNN are fed to a one-layer GRU structure.
        HR = torch.stack(hr_batched_output_per_clip, dim=0)
        SPO = torch.stack(spo2_batched_output_per_clip, dim=0)
        RF = torch.stack(rr_batched_output_per_clip, dim=0)




        return None, HR, SPO, None, RF