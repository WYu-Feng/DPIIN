import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
from utils import tensor_correlation, standard_scale, norm
from torchvision import models
from collections import OrderedDict
from torch.autograd import Variable
import torchvision.transforms as transforms
import math
from timm.models.layers import trunc_normal_


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out


class ASF(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch):
        super().__init__()
        self.N = 3
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch1 + in_ch2 + in_ch3, in_ch1 + in_ch2 + in_ch3, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(in_ch1 + in_ch2 + in_ch3, affine=False),
            nn.ReLU(True),
            nn.Conv2d(in_ch1 + in_ch2 + in_ch3, out_ch, kernel_size=1, padding=0)
        )

    def forward(self, feat):
        feat = self.enc(feat)
        return feat


class FH1(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch=256):
        super().__init__()

        self.clusterer1 = nn.Identity()
        self.clusterer2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.clusterer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fus = ASF(in_ch1, in_ch2, in_ch3, out_ch=out_ch)

    def forward(self, feat1, feat2, feat3):
        code1 = self.clusterer1(feat1)
        code2 = self.clusterer2(feat2)
        code3 = self.clusterer3(feat3)

        fused = self.fus(torch.cat((code1, code2, code3), dim=1))
        return fused


class FH2(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch=128):
        super().__init__()

        self.clusterer1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.clusterer2 = nn.Identity()
        self.clusterer3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fus = ASF(in_ch1, in_ch2, in_ch3, out_ch=out_ch)

    def forward(self, feat1, feat2, feat3):
        code1 = self.clusterer1(feat1)
        code2 = self.clusterer2(feat2)
        code3 = self.clusterer3(feat3)

        fused = self.fus(torch.cat((code1, code2, code3), dim=1))
        return fused


class FH3(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3, out_ch=64):
        super().__init__()

        self.clusterer1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.clusterer2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.clusterer3 = nn.Identity()

        self.fus = ASF(in_ch1, in_ch2, in_ch3, out_ch=out_ch)

    def forward(self, feat1, feat2, feat3):
        code1 = self.clusterer1(feat1)
        code2 = self.clusterer2(feat2)
        code3 = self.clusterer3(feat3)
        fused = self.fus(torch.cat((code1, code2, code3), dim=1))
        return fused


class GateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        super(GateConv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.gate_conv = nn.ConvTranspose2d(in_channels, out_channels * 2,
                                                kernel_size=kernel_size,
                                                stride=stride, padding=padding)
        else:
            self.gate_conv = nn.Conv2d(in_channels, out_channels * 2,
                                       kernel_size=kernel_size,
                                       stride=stride, padding=padding)

    def forward(self, x):
        x = self.gate_conv(x)
        (x, g) = torch.split(x, self.out_channels, dim=1)
        return x * torch.sigmoid(g)


class BAM(nn.Module):
    def __init__(self, map_dim=256, im_in_dim1=256, im_out_dim=256, use_concat=True, apa_idx=1):
        super(LAGI, self).__init__()

        self.apa_idx = apa_idx
        self.use_concat = use_concat

        # high-level (SPADE)
        self.conv_high1 = nn.Conv2d(map_dim + im_in_dim1, map_dim, 3, dilation=2, padding=2 * (3 // 2))
        self.spaed_block = SPADE(im_dim1, map_dim)
        self.conv_high2 = nn.Sequential(
            nn.Conv2d(in_channels=im_dim1, out_channels=im_out_dim, kernel_size=3, padding=1),
            nn.ReLU())

        # low-level (GC)
        self.conv_low1 = nn.Conv2d(in_channels=map_dim + im_in_dim1, out_channels=1, kernel_size=3, padding=3 // 2)
        self.conv_low2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            GateConv(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.low_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=im_in_dim1 + 12, out_channels=im_out_dim, kernel_size=3, padding=3 // 2),
            nn.BatchNorm2d(im_out_dim),
            nn.ReLU())

        # out
        self.out_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=im_out_dim + im_out_dim, out_channels=im_out_dim, kernel_size=3, padding=3 // 2),
            nn.InstanceNorm2d(im_out_dim, track_running_stats=False),
            nn.ReLU())

        self.out_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=im_out_dim, out_channels=im_out_dim, kernel_size=3, padding=3 // 2),
            nn.InstanceNorm2d(im_out_dim, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=im_out_dim, out_channels=im_out_dim, kernel_size=3, padding=3 // 2),
            nn.InstanceNorm2d(im_out_dim, track_running_stats=False),
            nn.ReLU()
        )

        if apa_idx == 1:
            self.apa = APA1(3, 3, 3, out_ch=3)
        elif apa_idx == 2:
            self.apa = APA2(3, 3, 3, out_ch=3)
        elif apa_idx == 3:
            self.apa = APA3(3, 3, 3, out_ch=3)

    def forward(self, x1, x2, rep_c_list):
        # information fusion
        map = self.apa(rep_c_list[2], rep_c_list[1], rep_c_list[0])

        # high-level information
        high_level = self.high_conv1(torch.cat((map, x1), dim=1))
        high_up_im = self.spaed_block(x1, high_level)
        high_up_im = self.high_conv2(high_up_im)

        # low-level information
        low_level = self.conv_low1(torch.cat((map, x1), dim=1))
        low_up_im = self.low_conv1(low_level)
        low_up_im = self.low_conv3(torch.cat((low_up_im, x1), dim=1))

        # out
        out = self.out_conv1(torch.cat((high_up_im, low_up_im), dim=1)) + x2
        out = self.out_conv2(out)
        return out


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

class Cl_Block(nn.Module):
    def __init__(self, in_ch, out_ch=3):
        super().__init__()

        self.clusterer = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, (1, 1)))

        self.nonlinear_clusterer = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, in_ch, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_ch, out_ch, (1, 1)))

    def forward(self, image_feat):
        code = self.clusterer(image_feat)
        code += self.nonlinear_clusterer(image_feat)
        return code