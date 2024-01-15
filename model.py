import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.spatial_transform import LearnableSpatialTransformWrapper
from modules.networks import Bottleneck, Cl_Block


class MAPs_R(nn.Module):
    def __init__(self, layers=(3, 4, 23, 3), output_dim=1024, heads=64 * 32 // 64, input_resolution=512, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # self.begin = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=False)

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down = nn.Upsample(scale_factor=1 / 2, mode='nearest')

        self.res_b1 = Cl_Block(152)
        self.res_b2 = Cl_Block(76)
        self.res_b3 = Cl_Block(76)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def get_fea_list(self, masked):
        rep_list = list()
        with torch.no_grad():
            x = masked.type(self.conv1.weight.dtype)
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            rep_list.append(x)

            x = self.avgpool(x)
            x = self.layer1(x)
            rep_list.append(x)

            x = self.layer2(x)
            rep_list.append(x)
        return rep_list

    def get_se_list(self, rep_list):
        rep_c_list = list()
        rep_c_list.append(self.res_b1(rep_list[0]))
        rep_c_list.append(self.res_b2(rep_list[1]))
        rep_c_list.append(self.res_b3(rep_list[2]))
        return rep_c_list

    def forward(self, x):
        return x


class MAPs_IN(nn.Module):
    def __init__(self, input_nc = 4, output_nc = 3, ngf=64):
        super().__init__()
        
        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=input_nc, out_channels=ngf, kernel_size=7, padding=0),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True)
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True)
        )

        blocks = []
        for i in range(8):
            cur_resblock = Bottleneck(ngf*4, ngf*4)
            blocks.append(cur_resblock)
        self.middle = nn.Sequential(*blocks)

        self.up = nn.Upsample(scale_factor=2)
        self.decoder1 = BAM(se_dim = 3, im_dim1 = ngf*4, im_dim2 = ngf*4, im_out_dim = ngf*4, use_concat = True, se_idx = 1)  # in = 32, out = 64
        self.decoder2 = BAM(se_dim = 3, im_dim1 = ngf*4, im_dim2 = ngf*2, im_out_dim = ngf*2, use_concat = True, se_idx = 2)  # in = 64, out = 128
        self.decoder3 = BAM(se_dim = 3, im_dim1 = ngf*2, im_dim2 = ngf, im_out_dim = ngf, use_concat = True, se_idx = 3)  # in = 128, out = 256

        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=output_nc, kernel_size=7, padding=0)
        )

    def forward(self, inputs, rep_c_list):
        inputs_list = []
        inputs = self.encoder1(inputs)
        inputs_list.append(inputs)

        inputs = self.encoder2(inputs)
        inputs_list.append(inputs)

        inputs = self.encoder3(inputs)
        x = inputs

        for i in range(8):
            x = self.middle[i](x)

        x = self.decoder1(x1=x, x2=inputs, rep_c_list=rep_c_list)

        x = self.up(x)
        x = self.decoder2(x1=x, x2=inputs_list[-1], rep_c_list=rep_c_list)

        x = self.up(x)
        x = self.decoder3(x1=x, x2=inputs_list[-2], rep_c_list=rep_c_list)

        x = self.out(x)
        x = (torch.tanh(x) + 1) / 2
        return x