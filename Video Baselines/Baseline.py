# -*- coding: UTF-8 -*-
import torch
import sys
from torchvision import models
import numpy as np
import utils
from basic_module import GRL
import models_vit
# from ZipLora import *
import torch.nn as nn
import torch.nn.functional as F

np.set_printoptions(threshold=np.inf)
sys.path.append('../..')
args = utils.get_args()


class Discriminator(nn.Module):
    def __init__(self, max_iter, domain_num=5):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, domain_num)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL(max_iter)

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out


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
            self.resnet = models.resnet18(pretrained=False)
            self.resnet.load_state_dict(torch.load('./pre_encoder/resnet18-5c106cde.pth'))

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.hr = nn.Linear(512, 1)
            self.spo = nn.Linear(512, 1)
            self.rf = nn.Linear(512, 1)

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
            '''

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
            )'''
            # self.dis = Discriminator(4000)

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
        # input = self.LowRankDecomposition(input)
        x = self.resnet.conv1(input)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x = self.resnet.layer1(x)
        # av1 = self.get_av(x)
        x = self.resnet.layer2(x)
        # av2 = self.get_av(x)
        x = self.resnet.layer3(x)
        # av3 = self.get_av(x)
        em = self.resnet.layer4(x)
        # av4 = self.get_av(em)

        # av = torch.cat([av1, av2, av3, av4], dim=1)

        HR = self.hr(self.avgpool(em).view(x.size(0), -1))
        SPO = self.spo(self.avgpool(em).view(x.size(0), -1))
        RF = self.rf(self.avgpool(em).view(x.size(0), -1))
        # For Sig
        x = self.up1_bvp(em)
        x = self.up2_bvp(x)
        x = self.up3_bvp(x)
        Sig = self.up4_bvp(x).squeeze(dim=1)
        '''
        x = self.up1_resp(em)
        x = self.up2_resp(x)
        x = self.up3_resp(x)
        Resp = self.up4_resp(x).squeeze(dim=1)'''

        # domain_label = self.dis(self.avgpool(em).view(x.size(0), -1))

        return Sig, HR, SPO, RF#, Resp#, self.avgpool(em).view(x.size(0), -1)  # , domain_label


class BaseNet_ViT(nn.Module):
    def __init__(self, pretrain='vit-base', gamma=8, lora_alpha=16):
        super(BaseNet_ViT, self).__init__()
        if pretrain == 'vit-base':
            self.vit = models_vit.__dict__['vit_base_patch16'](
                num_classes=1000,
                drop_path_rate=0.1,
                #global_pool=True,
                in_chans=3
            )
            self.vit.load_state_dict(torch.load('./pre_encoder/jx_vit_base_p16_224-80ecf9dd.pth'))
            del self.vit.head
            self.fc_norm = nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.hr = nn.Linear(768, 1)
            self.spo = nn.Linear(768, 1)
            self.rf = nn.Linear(768, 1)
            self.bvp = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(inplace=True),
                nn.Linear(768, 256))
            self.resp = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(inplace=True),
                nn.Linear(768, 256))

            #self.dis = Discriminator(4000)

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
        x = self.vit.forward_pos(input)
        for blk in self.vit.blocks:
            x = blk(x)
        x = x[:, 0:, :].mean(dim=1)  # global pool without cls token
        em = self.fc_norm(x)

        HR = self.hr(em)
        SPO = self.spo(em)
        RF = self.rf(em)
        # For Sig
        Sig = self.bvp(em)
        Resp = self.resp(em)

        #domain_label = self.dis(em)

        return Sig, HR, SPO, RF, Resp#, None#domain_label
