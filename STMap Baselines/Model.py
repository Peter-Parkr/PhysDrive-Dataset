from basic_module import *
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
import utils
from torchvision import models
from Baseline import BaseNet_CNN, BasicBlock
import numpy as np
import copy

np.set_printoptions(threshold=np.inf)
sys.path.append('..')


class My_model(nn.Module):
    def __init__(self, people_num=1):
        super(My_model, self).__init__()
        resnet = models.resnet18(pretrained=False)
        resnet.load_state_dict(torch.load('./pre_encoder/resnet18-5c106cde.pth'))

        self.gate_hr = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(256),
            nn.Sigmoid()
        )
        self.gate_rr = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(256),
            nn.Sigmoid()
        )
        self.gate_spo = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(256),
            nn.Sigmoid()
        )
        # self.gate_P = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.Dropout2d(0.5),
        #     nn.BatchNorm2d(256),
        #     nn.Sigmoid()
        # )

        '''
        self.part_hr = nn.Sequential(
            BasicBlock(256, 512, 2, downsample=1),
            BasicBlock(512, 512, 1, downsample=0),
        )  # resnet.layer4
        self.part_rr = nn.Sequential(
            BasicBlock(256, 512, 2, downsample=1),
            BasicBlock(512, 512, 1, downsample=0),
        )
        self.part_spo = nn.Sequential(
            BasicBlock(256, 512, 2, downsample=1),
            BasicBlock(512, 512, 1, downsample=0),
        )'''

        self.part_hr = copy.deepcopy(resnet.layer4)
        self.part_rr = copy.deepcopy(resnet.layer4)
        self.part_spo = copy.deepcopy(resnet.layer4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.hr = nn.Linear(512, 1)
        self.spo = nn.Linear(512, 1)
        self.rf = nn.Linear(512, 1)

        #self.P = nn.Linear(512, people_num)

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
        # self.dis = Discriminator(4000, domain_num=people_num)
        #self.offset = Offset_Learner(in_feat=256, domain_num=people_num)

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
        self.l1_IN = nn.InstanceNorm2d(128)
        self.layer2 = resnet.layer2
        self.l2_IN = nn.InstanceNorm2d(256)
        self.layer3 = resnet.layer3
        self.l3_IN = nn.InstanceNorm2d(256)


        # self.part_P = copy.deepcopy(resnet.layer4)

        # self.mixstyle = MixStyle(p=0.5, alpha=0.1)

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

    def get_av(self, x):

        av = torch.mean(torch.mean(x, dim=-1), dim=-1)
        min, _ = torch.min(av, dim=1, keepdim=True)
        max, _ = torch.max(av, dim=1, keepdim=True)
        av = torch.mul((av - min), ((max - min).pow(-1)))
        mean = torch.mean(av, dim=-1, keepdim=True)
        # standard deviation
        std = torch.std(av, dim=-1, keepdim=True)
        # av = (x - av) / std
        av = (av - mean) / std

        # x = x.reshape(x.shape[0], -1)
        # av = torch.mean(torch.mean(x, dim=-1, keepdim=True), dim=-2, keepdim=True)
        # av = torch.mean(x, dim=-1, keepdim=True)
        # standard deviation
        # std = torch.std(x, dim=-1, keepdim=True)
        # av = (x - av) / std
        # av = (x - av) / std
        # min, _ = torch.min(av, dim=-1, keepdim=True)
        # max, _ = torch.max(av, dim=-1, keepdim=True)
        # av = torch.mul((av - min), ((max - min).pow(-1)))
        return av  # av.reshape(x.shape[0], x.shape[1], -1)

    def forward(self, input, withIN=False):
        B = input.size(0)
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        if withIN:
            x = self.l1_IN(x)
        feat_av1 = self.get_av(x)
        # x = self.mixstyle(x)
        x = self.layer2(x)
        if withIN:
            x = self.l2_IN(x)
        feat_av2 = self.get_av(x)
        # x = self.mixstyle(x)
        x = self.layer3(x)
        if withIN:
            x = self.l3_IN(x)
        feat_av3 = self.get_av(x)

        # dis_invariant = self.dis(torch.cat([feat_av1, feat_av2, feat_av3],dim=-1))

        # em_P0 = torch.mul(2 * self.gate_P(x), x)
        # # em_P = self.P_encode(self.avgpool(em_P0).reshape(B, -1))
        # # dis_invariant = self.P_decode(em_P)
        #
        # em_P = self.part_P(em_P0)
        # dis_invariant = self.P(self.avgpool(em_P).reshape(B, -1))

        #dis_invariant = self.offset(feat_av3)

        em_hr0 = torch.mul(2 * self.gate_hr(x), x)
        em_rr0 = torch.mul(2 * self.gate_rr(x), x)
        em_spo0 = torch.mul(2 * self.gate_spo(x), x)

        em_hr = self.part_hr(em_hr0)
        em_rr = self.part_rr(em_rr0)
        em_spo = self.part_spo(em_spo0)

        HR = self.hr(self.avgpool(em_hr).reshape(B, -1))
        RF = self.rf(self.avgpool(em_rr).reshape(B, -1))
        SPO = self.spo(self.avgpool(em_spo).reshape(B, -1))

        x = self.up1_bvp(em_hr)
        x = self.up2_bvp(x)
        x = self.up3_bvp(x)
        Sig = self.up4_bvp(x).squeeze(dim=1)

        return (Sig, HR, SPO, RF)
            #     , [feat_av1, feat_av2, feat_av3], dis_invariant, [
            # self.avgpool(em_hr0).reshape(B, -1),
            # self.avgpool(em_rr0).reshape(B, -1),
            # self.avgpool(em_spo0).reshape(B, -1),
            # self.offset.get_offset(feat_av3)])


def freeze_model_BN(model):
    """Freezes all weights of the model."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.training = False
    return model



def freeze_model(model, freeze=True):
    """Freezes all weights of the model."""
    if freeze:
        # First freeze/ unfreeze all model weights
        for n, p in model.named_parameters():
            # if 'bvp' not in n and 'hr' not in n and 'spo' not in n and 'rr' not in n:
            if 'bvp' not in n and 'hr' not in n and 'spo' not in n and 'rr' not in n:
                p.requires_grad = False
            elif 'part' in n or 'gate' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

        for n, p in model.named_parameters():
            if "bn" in n:
                p.requires_grad = True
    else:
        # Unfreeze
        for n, p in model.named_parameters():
            p.requires_grad = True
    return model

# import scipy.io as scio
# import matplotlib.pyplot as plt
# import numpy as np
# bvp = scio.loadmat('RESP_Filt')['RESP']
# bvp = np.array(bvp.astype('float32')).reshape(-1)
# bvp = bvp[:256]
# bvp = (bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
# bvp = bvp.astype('float32')
#
# plt.plot(bvp)
# plt.show()

