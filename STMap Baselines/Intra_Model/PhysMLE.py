# -*- coding: UTF-8 -*-
import torch
import sys
from torchvision import models
import numpy as np
import utils


np.set_printoptions(threshold=np.inf)
sys.path.append('..')
args = utils.get_args()

import torch.nn as nn
from collections import OrderedDict
import math
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    Base lora class
    """

    def __init__(
            self,
            r,
            lora_alpha,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        # Mark the weight as unmerged
        self.merged = False

    def reset_parameters(self):
        raise NotImplementedError

    def train(self, mode: bool = True):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError


class LoRALinear(LoRALayer):
    def __init__(self, r, lora_alpha, linear_layer):
        """
        LoRA class for nn.Linear class
        :param r: low rank dimension
        :param lora_alpha: scaling factor
        :param linear_layer: target nn.Linear layer for applying Lora
        """
        super().__init__(r, lora_alpha)
        self.linear = linear_layer

        in_features = self.linear.in_features
        out_features = self.linear.out_features

        # Lora configuration
        self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.linear.weight.new_zeros((out_features, r)))

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def get_task_weights(self):
        return (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling

    def train(self, mode: bool = True):
        self.linear.train(mode)

    def eval(self):
        self.linear.eval()

    def forward(self, x):
        result = F.linear(
            input=x,
            weight=(self.lora_B @ self.lora_A) * self.scaling, bias=self.linear.bias
        )
        return result


class LoraConv2d(LoRALayer):
    def __init__(self, r, lora_alpha, conv_layer):
        super().__init__(r, lora_alpha)

        self.conv = conv_layer

        in_channels = self.conv.in_channels
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]

        self.lora_A = nn.Parameter(
            self.conv.weight.new_zeros((kernel_size * r, in_channels * kernel_size))
        )
        self.lora_B = nn.Parameter(
            self.conv.weight.new_zeros((out_channels * kernel_size, kernel_size * r))
        )

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        self.conv.train(mode)

    def eval(self):
        self.conv.eval()

    def get_task_weights(self):
        return (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling

    def forward(self, x):
        result = F.conv2d(
            x,
            (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
            self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups
        )
        return result


class MLoraConv2d(LoRALayer):
    def __init__(self, r, lora_alpha, conv_layer, num_E=3):
        """
        LoRA class for nn.Conv2d class
        """
        super().__init__(r, lora_alpha)
        self.conv = conv_layer

        self.num_E = num_E

        self.experts = []
        self.gates = []

        for _ in range(self.num_E):
            self.experts.append(LoraConv2d(r, lora_alpha, conv_layer))

            self.gates.append(nn.Sequential(
                nn.Conv2d(self.conv.out_channels, self.conv.out_channels, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.BatchNorm2d(self.conv.out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.conv.out_channels, self.conv.out_channels, kernel_size=1, stride=1, padding=0,
                          bias=False),
                nn.Dropout2d(0.1),
                nn.BatchNorm2d(self.conv.out_channels),
                nn.Sigmoid()
            ))
            # self.merges.append(nn.Parameter(
            # self.conv.weight.new_ones(self.conv.weight.shape)
            # ))
        self.experts = nn.ModuleList(self.experts)
        self.gates = nn.ModuleList(self.gates)
        # self.merges = nn.ParameterList(self.merges)

    def get_query(self, x):
        self.query = x

    def self_gate(self, x):
        return x * torch.sigmoid(x)

    def get_task_weights(self):
        return [self.experts[i].get_task_weights() for i in range(self.num_E)]

    def train(self, mode: bool = True):
        self.conv.train(mode)

    def eval(self):
        self.conv.eval()

    def forward(self, x):

        '''results = []
        for i in range(self.num_E):
            results.append(self.experts[i](x).unsqueeze(-2))
        results = torch.cat(results, dim=-2)
        self.query = self.query.unsqueeze(-1)
        att = torch.softmax(torch.matmul(results, self.query), dim=-1).permute(0, 1, 2, 4, 3)
        results = torch.matmul(att, results)
        return results.squeeze()
        '''

        results = self.conv(x)
        results += torch.mul(2 * self.gates[0](self.query), self.experts[0](x))
        # results = self.experts[0](x)
        for i in range(1, self.num_E):
            results += torch.mul(2 * self.gates[i](self.query), self.experts[i](x))
            # results += self.experts[i](x)

        return results


class MLoraLinear(LoRALayer):
    def __init__(self, r, lora_alpha, linear_layer, num_E=3):
        """
        LoRA class for nn.Conv2d class
        """
        super().__init__(r, lora_alpha)
        self.linear = linear_layer

        self.num_E = num_E

        self.experts = []
        self.gates = []

        for _ in range(self.num_E):
            self.experts.append(LoRALinear(r, lora_alpha, linear_layer))

            self.gates.append(nn.Sequential(
                #nn.Linear(768, self.linear.out_features),
                #nn.LayerNorm(self.linear.out_features),
                #nn.ReLU(inplace=True),
                nn.Linear(768, self.linear.out_features),
                nn.LayerNorm(self.linear.out_features),
                nn.Dropout(0.1),
                nn.Sigmoid()
            ))
            # self.merges.append(nn.Parameter(
            # self.conv.weight.new_ones(self.conv.weight.shape)
            # ))
        self.experts = nn.ModuleList(self.experts)
        self.gates = nn.ModuleList(self.gates)
        # self.merges = nn.ParameterList(self.merges)

    def get_query(self, x):
        self.query = x

    def self_gate(self, x):
        return x * torch.sigmoid(x)

    def get_task_weights(self):
        return [self.experts[i].get_task_weights() for i in range(self.num_E)]

    def train(self, mode: bool = True):
        self.linear.train(mode)

    def eval(self):
        self.linear.eval()

    def forward(self, x):
        results = self.linear(x)
        results += torch.mul(2 * self.gates[0](self.query), self.experts[0](x))
        for i in range(1, self.num_E):
            results += torch.mul(2 * self.gates[i](self.query), self.experts[i](x))
        return results


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
    def __init__(self, pretrain='resnet18', gamma=8, lora_alpha=16):
        super(BaseNet_CNN, self).__init__()
        if pretrain == 'resnet18':
            self.resnet = models.resnet18(pretrained=False)
            self.resnet.load_state_dict(torch.load('./pre_encoder/resnet18-5c106cde.pth'))

            self.add_adapter(MLoraConv2d, gamma=gamma, lora_alpha=lora_alpha)
            self.freeze_model(True)

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

            self.gate_hr = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Dropout2d(0.1),
                nn.BatchNorm2d(512),
                nn.Sigmoid()
            )

            self.gate_spo = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Dropout2d(0.1),
                nn.BatchNorm2d(512),
                nn.Sigmoid()
            )

            self.gate_rf = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Dropout2d(0.1),
                nn.BatchNorm2d(512),
                nn.Sigmoid()
            )


        elif pretrain == 'resnet50':
            self.resnet = models.resnet50(pretrained=False)
            self.resnet.load_state_dict(torch.load('./pre_encoder/resnet50-19c8e357.pth'))

            self.add_adapter(LoraConv2d, gamma=gamma, lora_alpha=lora_alpha)
            self.freeze_model(True)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.hr = nn.Linear(2048, 1)
            self.spo = nn.Linear(2048, 1)
            # For Sig
            # 模仿以下的结构，将[batch_size, 2048, 4, 16]的特征图转换为[batch_size, 1, 256, 1]的特征图
            self.up1_bvp = nn.Sequential(
                nn.ConvTranspose2d(2048, 2048, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(2048, 512, [2, 1], downsample=1),
            )
            self.up2_bvp = nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(512, 128, [1, 1], downsample=1),
            )
            self.up3_bvp = nn.Sequential(
                nn.ConvTranspose2d(128, 128, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(128, 32, [2, 1], downsample=1),
            )
            self.up4_bvp = nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size=[1, 2], stride=[1, 2]),
                BasicBlock(32, 1, [1, 1], downsample=1),
            )

    def add_adapter(self, adapter_class, gamma=8, lora_alpha=16):
        # Add adapter for resnet blocks
        target_layers = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ]

        for layer in target_layers:
            for bottleneck_layer in layer:
                for cv in ["conv1", "conv2", "conv3"]:
                    if hasattr(bottleneck_layer, cv) and getattr(bottleneck_layer, cv) is not None:
                        target_conv = getattr(bottleneck_layer, cv)
                        adapter = adapter_class(
                            r=gamma,
                            lora_alpha=lora_alpha,
                            conv_layer=target_conv,
                            num_E=3
                        )
                        setattr(bottleneck_layer, cv, adapter)

    def query(self, x):
        # Add adapter for resnet blocks
        target_layers = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ]

        for layer in target_layers:
            for bottleneck_layer in layer:
                if getattr(bottleneck_layer, 'downsample') is not None:
                    x = bottleneck_layer.downsample(x)
                for cv in ["conv1", "conv2", "conv3"]:
                    if hasattr(bottleneck_layer, cv) and getattr(bottleneck_layer, cv) is not None:
                        target_conv = getattr(bottleneck_layer, cv)
                        target_conv.get_query(x)
        return x

    def freeze_model(self, freeze=True):
        """Freezes all weights of the model."""
        if freeze:
            # First freeze/ unfreeze all model weights
            for n, p in self.named_parameters():
                if 'lora_' not in n and 'merge_' not in n and 'downsample' not in n and 'experts' not in n and 'gate' not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

            for n, p in self.named_parameters():
                if 'bias' in n:
                    if "fc" not in n:
                        p.requires_grad = True
                elif "bn" in n:
                    p.requires_grad = True
        else:
            # Unfreeze
            for n, p in self.named_parameters():
                p.requires_grad = True
        self.resnet.conv1.requires_grad = True
        self.model_frozen = freeze

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
        print("Non-trainable parameters (M):", other_param_num/ (1024 ** 2))
        print("Trainable parameters (M):", trainable_param_num / (1024 ** 2))

        ratio = trainable_param_num / (other_param_num + trainable_param_num)
        # final_ratio = (ratio / (1 - ratio))
        print("Ratio:", ratio)

        return ratio

    def get_task_weights(self):
        w = []

        target_layers = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ]
        for layer in target_layers:
            for bottleneck_layer in layer:
                for cv in ["conv1", "conv2", "conv3"]:
                    if hasattr(bottleneck_layer, cv) and getattr(bottleneck_layer, cv) is not None:
                        target_conv = getattr(bottleneck_layer, cv)
                        w_now = target_conv.get_task_weights()
                        w.append(w_now)
        return w

    def predict_spo(self, feature):
        return self.spo(feature)


    def forward(self, input):
        # input, loss = self.LowRankDecomposition(input)
        x = self.resnet.conv1(input)
        #loss = self.LowRankDecomposition(x)
        feat = [x]
        query = self.query(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.layer1(x)
        feat.append(x)
        x = self.resnet.layer2(x)
        feat.append(x)
        x = self.resnet.layer3(x)
        feat.append(x)
        em = self.resnet.layer4(x)
        feat.append(em)

        query_hr = 2 * self.gate_hr(query)
        em_hr = torch.mul(query_hr, em)
        HR = self.hr(self.avgpool(em_hr).view(x.size(0), -1))

        query_spo = 2 * self.gate_spo(query)
        em_spo = torch.mul(query_spo, em)
        SPO = self.spo(self.avgpool(em_spo).view(x.size(0), -1))

        query_rf = 2 * self.gate_rf(query)
        em_rf = torch.mul(query_rf, em)
        RF = self.rf(self.avgpool(em_rf).view(x.size(0), -1))

        # For Sig
        x = self.up1_bvp(em_hr)
        x = self.up2_bvp(x)
        x = self.up3_bvp(x)
        Sig = self.up4_bvp(x).squeeze(dim=1)

        return Sig, HR, SPO, Sig, RF
        # return Sig, HR, SPO, SPO, em
