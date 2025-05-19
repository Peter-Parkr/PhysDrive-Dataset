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
        # print(em.shape)
        # print(self.avgpool(em).shape)
        # ([100, 512, 4, 16])

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

        return None, HR, SPO, None, RF#, domain_label #self.avgpool(em).view(x.size(0), -1)  # , domain_label


# class BVPNet(nn.Module):
#     def __init__(self, frames=256,fw=48):
#         super(BVPNet, self).__init__()
#         #add double conv
#         self.inc = nn.Sequential(
#             nn.Conv2d(3, fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(fw),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(fw, fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(fw),
#             nn.ReLU(inplace=True),
#         )
#
#         self.down1 = nn.Sequential(
#             nn.Conv2d(fw, 2*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(2*fw),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(2*fw, 2*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(2*fw),
#             nn.ReLU(inplace=True),
#         )
#
#         self.down2 = nn.Sequential(
#             nn.Conv2d(2*fw, 4*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(4*fw),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(4*fw, 4*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(4*fw),
#             nn.ReLU(inplace=True),
#         )
#
#         self.down3 = nn.Sequential(
#             nn.Conv2d(4*fw, 8*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(8*fw),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(8*fw, 8*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(8*fw),
#             nn.ReLU(inplace=True),
#         )
#
#         self.down4 = nn.Sequential(
#             nn.Conv2d(8*fw, 8*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(8*fw),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(8*fw, 8*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(8*fw),
#             nn.ReLU(inplace=True),
#         )
#
#         self.up1 = nn.Sequential(
#             nn.Conv2d(8*fw, 4*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(4*fw),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(4*fw, 4*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(4*fw),
#             nn.ReLU(inplace=True),
#         )
#
#
#         self.up2 = nn.Sequential(
#             nn.Conv2d(4*fw, 2*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(2*fw),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(2*fw, 2*fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(2*fw),
#             nn.ReLU(inplace=True),
#         )
#
#         self.up3 = nn.Sequential(
#             nn.Conv2d(2*fw, fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(fw),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(fw, fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(fw),
#             nn.ReLU(inplace=True),
#         )
#
#         self.up4 = nn.Sequential(
#             nn.Conv2d(fw, fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(fw),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(fw, fw, [3,3],stride=1, padding=1,padding_mode="replicate"),
#             nn.BatchNorm2d(fw),
#             nn.ReLU(inplace=True),
#         )
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         self.outc = nn.Sequential(
#             nn.Conv2d(fw,3, [3,3],stride=1, padding=1,padding_mode="replicate"),
#         )
#
#         self.upsample = nn.Upsample(scale_factor=(2,2),mode="nearest")
#         self.avgpool = nn.AvgPool2d((2, 2), stride=(2, 2))
#
#         self.avgpool_output = nn.AdaptiveAvgPool2d((1, 1))
#         self.hr = nn.Linear(256, 1)
#         self.spo = nn.Linear(256, 1)
#         self.rf = nn.Linear(256, 1)
#
#     def forward(self, x):
#         # 64,256,3
#         # print(x.shape)
#         x = self.inc(x)
#         x = self.avgpool(x)
#         # 32,128,48
#         x = self.down1(x)
#         x = self.avgpool(x)
#         # 16,64,96
#         x = self.down2(x)
#         x = self.avgpool(x)
#         # 8,32,192
#         x = self.down3(x)
#         x = self.avgpool(x)
#         # 4,16,384
#         feat = self.down4(x)
#         # 4,16,384
#
#         x = self.upsample(feat)
#         x = self.up1(x)
#         # 8,32,192
#         x = self.upsample(x)
#         x = self.up2(x)
#         # 16,64,96
#         x = self.upsample(x)
#         x = self.up3(x)
#         # 32,128,48
#         x = self.upsample(x)
#         x = self.up4(x)
#         # 64,256,48  ([100, 512, 4, 16])
#         em = self.outc(x)
#
#         em = em.permute(0, 3, 1, 2)
#         # print(em.shape)
#         # 64,256,3
#         HR = self.hr(self.avgpool_output(em).view(em.size(0), -1))
#         SPO = self.spo(self.avgpool_output(em).view(em.size(0), -1))
#         RF = self.rf(self.avgpool_output(em).view(em.size(0), -1))
#
#         return None, HR, SPO, None, RF

class DualConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DualConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class BVPNet(nn.Module):
    def __init__(self, input_channels=3):
        super(BVPNet, self).__init__()

        # inc layer
        self.inc = nn.Sequential(
            nn.Conv2d(input_channels, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            DualConv(48, 48)
        )

        # down layers
        self.down1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            DualConv(48, 96)
        )

        self.down2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            DualConv(96, 192)
        )

        self.down3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            DualConv(192, 384)
        )

        # up layers
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DualConv(384, 192)
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DualConv(192, 96)
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DualConv(96, 48)
        )

        # outc layer
        self.outc = nn.Conv2d(48, 1, kernel_size=3, padding=1)

        self.avgpool_output = nn.AdaptiveAvgPool2d((1, 1))
        self.hr = nn.Linear(256, 1)
        self.spo = nn.Linear(256, 1)
        self.rf = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.up1(x4)
        x6 = self.up2(x5)
        x7 = self.up3(x6)

        em = self.outc(x7)

        em = em.permute(0, 3, 1, 2)
        # print(em.shape)
        # 64,256,3
        sig = torch.rand(em.size(0), 1, 256).to(em.device)
        HR = self.hr(self.avgpool_output(em).view(em.size(0), -1))
        SPO = self.spo(self.avgpool_output(em).view(em.size(0), -1))
        RF = self.rf(self.avgpool_output(em).view(em.size(0), -1))
        return sig, HR, SPO, sig, RF