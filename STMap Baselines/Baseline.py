# -*- coding: UTF-8 -*-
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


class BaseNet_CNN(nn.Module):
    def __init__(self, pretrain='resnet18'):
        super(BaseNet_CNN, self).__init__()
        if pretrain == 'resnet18':
            resnet = models.resnet18(pretrained=False)
            resnet.load_state_dict(torch.load('./pre_encoder/resnet18-5c106cde.pth'))

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.hr = nn.Linear(512, 1)
            self.spo = nn.Linear(512, 1)
            self.rf = nn.Linear(512, 1)

            #self.mixstyle = basic_module.MixStyle(p=0.5, alpha=0.1)

            self.up1_bvp = nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(512, 256, [2, 1], downsample=1),
            )
            self.up2_bvp = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(256, 64, [1, 1], downsample=1),
            )
            self.up3_bvp = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(64, 32, [2, 1], downsample=1),
            )
            self.up4_bvp = nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(32, 1, [1, 1], downsample=1, islast=True),
            )

            self.up1_resp = nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(512, 256, [2, 1], downsample=1),
            )
            self.up2_resp = nn.Sequential(
                nn.ConvTranspose2d(256, 256, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(256, 64, [1, 1], downsample=1),
            )
            self.up3_resp = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(64, 32, [2, 1], downsample=1),
            )
            self.up4_resp = nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(32, 1, [1, 1], downsample=1, islast=True),
            )

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

    def get_av(self, x):
        av = torch.mean(torch.mean(x, dim=-1), dim=-1)
        min, _ = torch.min(av, dim=1, keepdim=True)
        max, _ = torch.max(av, dim=1, keepdim=True)
        av = torch.mul((av - min), ((max - min).pow(-1)))
        return av

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


    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        em = self.layer4(x)

        HR = self.hr(self.avgpool(em).view(x.size(0), -1))
        SPO = self.spo(self.avgpool(em).view(x.size(0), -1))
        RF = self.rf(self.avgpool(em).view(x.size(0), -1))
        # For Sig
        x = self.up1_bvp(em)
        x = self.up2_bvp(x)
        x = self.up3_bvp(x)
        Sig = self.up4_bvp(x).squeeze(dim=1)

        # For Resp
        x = self.up1_resp(em)
        x = self.up2_resp(x)
        x = self.up3_resp(x)
        Resp = self.up4_resp(x).squeeze(dim=1)

        #domain_label = self.dis(self.avgpool(em).view(x.size(0), -1))

        return Sig, HR, SPO, Resp, RF#, domain_label #self.avgpool(em).view(x.size(0), -1)  # , domain_label

