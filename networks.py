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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def Upsample(dim):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, dim//2, 3, padding=1)
    )

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)
    
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


class APA(nn.Module):
    def __init__(self, in_fea_dim = 64, in_rep_dim = [152, 76, 76], resolution = 152, out_dim = 64):
        super(APA, self).__init__()
        self.resolution = resolution

        self.conv11 = nn.Conv2d(in_rep_dim[0], in_fea_dim, 3, padding=3//2)
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_fea_dim + in_rep_dim[0], in_fea_dim, 1),
            nn.GELU()
        )
        self.conv13 = nn.Conv2d(in_fea_dim, in_fea_dim, 3, padding=3 // 2)

        self.conv21 = nn.Conv2d(in_rep_dim[1], in_fea_dim, 3, padding=3//2)
        self.conv22 = nn.Sequential(
            nn.Conv2d(in_fea_dim + in_rep_dim[1], in_fea_dim, 1),
            nn.GELU()
        )
        self.conv23 = nn.Conv2d(in_fea_dim, in_fea_dim, 3, padding=3 // 2)

        self.conv31 = nn.Conv2d(in_rep_dim[2], in_fea_dim, 3, padding=3//2)
        self.conv32 = nn.Sequential(
            nn.Conv2d(in_fea_dim + in_rep_dim[2], in_fea_dim, 1),
            nn.GELU()
        )
        self.conv33 = nn.Conv2d(in_fea_dim, in_fea_dim, 3, padding=3 // 2)

        self.out_conv = nn.Conv2d(in_fea_dim + in_fea_dim + in_fea_dim, out_dim, 1)

    def forward(self, x, rep_c_list):
        p1 = F.interpolate(rep_c_list[0], size = self.resolution, mode='bilinear', align_corners=False)
        p2 = F.interpolate(rep_c_list[1], size=self.resolution, mode='bilinear', align_corners=False)
        p3 = F.interpolate(rep_c_list[2], size=self.resolution, mode='bilinear', align_corners=False)

        p1_in = self.conv11(p1)
        x1 = self.conv12(torch.cat((p1, x), dim = 1))
        x1 = x1 * p1_in
        x1 = self.conv13(x1) + p1_in

        p2_in = self.conv21(p2)
        x2 = self.conv22(torch.cat((p2, x), dim = 1))
        x2 = x2 * p2_in
        x2 = self.conv23(x2) + p2_in

        p3_in = self.conv31(p3)
        x3 = self.conv32(torch.cat((p3, x), dim = 1))
        x3 = x3 * p3_in
        x3 = self.conv33(x3) + p3_in

        out = self.out_conv(torch.cat((x1, x2, x3), dim = 1))
        return out

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
    def __init__(self, x1_dim = 64, x2_dim = 64, prior_dim = 152, out_dim = 64, use_concat = True):
        super(BAM, self).__init__()

        self.use_concat = use_concat

        self.conv1 = nn.Conv2d(x1_dim, x1_dim, 3, padding=3//2)
        self.conv2 = nn.Conv2d(prior_dim, prior_dim, 3, padding=3 // 2)

        self.spaed_conv = nn.Conv2d(prior_dim + x1_dim, x1_dim, 3, padding=3//2)
        self.spaed_block = SPADE(x1_dim, prior_dim)

        self.gated_block = nn.Sequential(
            nn.Conv2d(prior_dim + x1_dim, x1_dim, 3, padding=3//2),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            GateConv(in_channels = x1_dim, out_channels = x1_dim, kernel_size=3, stride=1, padding=0)
        )

        # out
        self.reduce_chan = nn.Conv2d(x1_dim + x1_dim + x2_dim, x1_dim, kernel_size=1, bias=False)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels = x1_dim, out_channels = out_dim, kernel_size=3, padding=3 // 2),
            nn.InstanceNorm2d(out_dim, track_running_stats=False),
            nn.ReLU()
        )

    def forward(self, x1, x2, priors):
        prior_ident = priors
        priors = self.conv2(priors)
        x1 = self.conv1(x1)

        spaed_out = self.spaed_block(self.spaed_conv(torch.cat((x1, priors), dim = 1)), priors)
        gated_out = self.gated_block(torch.cat((x1, priors), dim = 1)))

        if self.use_concat:
            out = self.reduce_chan(torch.cat((spaed_out, gated_out, x2), dim = 1)))
        else:
            out = spaed_out + gated_out + x2

        out = self.out_conv(out)
        return out

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
