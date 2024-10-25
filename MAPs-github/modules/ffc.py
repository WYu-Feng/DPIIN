import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.spatial_transform import LearnableSpatialTransformWrapper
from modules.MAP_networks import ResBlock, Upsample, Downsample
from modules.MAP_networks import BAM, APA
from modules.rn import RN_B
from modules.MAP_networks import ResnetBlock_Spade, RDB, ResnetBlock, ResnetBlock_withshort, ResnetBlock_Spade_MaskG

class MAP_R(nn.Module):
    def __init__(self, input_channels=3, residual_blocks=8):
        super(MAP_R, self).__init__()

        # Encoder
        self.encoder_prePad = nn.ReflectionPad2d(3)
        self.encoder_conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7, padding=0)
        self.encoder_in1 = RN_B(feature_channels=64)
        self.encoder_relu1 = nn.ReLU(True)
        self.encoder_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.encoder_in2 = RN_B(feature_channels=128)
        self.encoder_relu2 = nn.ReLU(True)
        self.encoder_conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.encoder_in3 = RN_B(feature_channels=256)
        self.encoder_relu3 = nn.ReLU(True)

        # Middle
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock_Spade(256, layout_dim=256, dilation=2, use_spectral_norm=False)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        # Encoder semantic branch
        self.encoder_prePad_sm = nn.ReflectionPad2d(3)
        self.encoder_conv1_sm = nn.Conv2d(in_channels=input_channels+1, out_channels=64, kernel_size=7, padding=0)
        self.encoder_relu1_sm = nn.ReLU()
        self.encoder_conv2_sm = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.encoder_relu2_sm = nn.ReLU()
        self.encoder_conv3_sm = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.encoder_relu3_sm = nn.ReLU()
        self.encoder_conv4_sm = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.encoder_relu4_sm = nn.ReLU()

        self.encoder_sm_out = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # branch for Asl feature recon
        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(256, 32, 6)])
        for _ in range(15):
            self.rdbs.append(RDB(256, 32, 6))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(256 * 16, 256, kernel_size=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=3 // 2),
        )
        self.feature_recon = nn.Sequential(
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
        )
        self.feature_mapping = nn.Conv2d(in_channels=256, out_channels=152, kernel_size=1, stride=1, padding=0)
        self.feature_recon_decoder = nn.Sequential(
            nn.Conv2d(256, 128*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            ResnetBlock_withshort(128+256, 128, dilation=1, use_spectral_norm=False),
            nn.Conv2d(128, 64*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            ResnetBlock_withshort(64+128, 64, dilation=1, use_spectral_norm=False),
        )
        self.feature_mapping_128 = nn.Conv2d(in_channels=128, out_channels=76, kernel_size=1, stride=1, padding=0)
        self.feature_mapping_256 = nn.Conv2d(in_channels=64, out_channels=76, kernel_size=1, stride=1, padding=0)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ResnetBlock_Spade_MaskG(f_in=256, f_out=128, layout_dim=128),
            nn.Upsample(scale_factor=2),
            ResnetBlock_Spade_MaskG(f_in=128, f_out=64, layout_dim=64),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=input_channels, kernel_size=7, padding=0)

        )

    def encoder(self, x, mask):
        x = self.encoder_prePad(x)

        x = self.encoder_conv1(x)
        x = self.encoder_in1(x, mask)
        x = self.encoder_relu2(x)

        x = self.encoder_conv2(x)
        x = self.encoder_in2(x, mask)
        x = self.encoder_relu2(x)

        x = self.encoder_conv3(x)
        x = self.encoder_in3(x, mask)
        x = self.encoder_relu3(x)
        return x

    def encoder_sm(self, x):
        x = self.encoder_prePad_sm(x)

        x = self.encoder_conv1_sm(x)
        x = self.encoder_relu2_sm(x)

        x = self.encoder_conv2_sm(x)
        x_256 = self.encoder_relu2_sm(x)

        x = self.encoder_conv3_sm(x_256)
        x_128 = self.encoder_relu3_sm(x)

        x = self.encoder_conv4_sm(x_128)
        x = self.encoder_relu4_sm(x)
        return x, x_128, x_256

    def forward(self, x, mask, masked_512, mask_512):
        x_input = (x * (1 - mask)) + mask
        # input mask: 1 for hole, 0 for valid
        x = self.encoder(x_input, mask)

        # perform feature recon
        x_sm, x_sm_128, x_sm_256 = self.encoder_sm(torch.cat((masked_512, mask_512), 1))
        x_sm_skip = self.encoder_sm_out(x_sm)
        local_features = []
        for i in range(16):
            x_sm_skip = self.rdbs[i](x_sm_skip)
            local_features.append(x_sm_skip)

        x_sm = self.gff(torch.cat(local_features, 1)) + x_sm
        layout_64 = self.feature_recon(x_sm)
        feature_recon_64 = self.feature_mapping(layout_64)
        for i in range(len(self.feature_recon_decoder)):
            sub_block = self.feature_recon_decoder[i]
            if i == 0:
                layout_128 = sub_block(layout_64)
            elif i == 3:
                layout_128 = sub_block(torch.cat((x_sm_128, layout_128), 1))
                layout_forrecon = layout_128
            elif i ==7:
                layout_256 = sub_block(torch.cat((x_sm_256, layout_128), 1))
            else:
                layout_128 = sub_block(layout_128)
        feature_recon_128 = self.feature_mapping_128(layout_forrecon)
        feature_recon_256 = self.feature_mapping_256(layout_256)

        return [feature_recon_256, feature_recon_128, feature_recon_64]

class MAPs_IN(nn.Module):
    def __init__(self, input_nc = 4, output_nc = 3, ngf=64, prior_dim = 10):
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


        ### resnet blocks
        blocks = []
        n_blocks = 4
        for _ in range(n_blocks):
            cur_resblock = ResBlock(in_channels = ngf*4, out_channels = ngf*4)
            blocks.append(cur_resblock)
        self.middle = nn.Sequential(*blocks)


        self.decoder1 = BAM(x_dim = ngf*4, prior_dim = prior_dim, out_dim = ngf*4, use_concat = True)  # in = 32, out = 64
        self.up1 = Upsample(dim = ngf*4)

        self.decoder2 = BAM(x_dim = ngf*2, prior_dim = prior_dim, out_dim = ngf*2, use_concat = True)  # in = 64, out = 128
        self.up2 = Upsample(dim = ngf*2)

        self.decoder3 = BAM(x_dim = ngf, prior_dim = prior_dim, out_dim = ngf, use_concat = True)  # in = 128, out = 256

        self.apa1 = APA(in_fea_dim = ngf*4, in_rep_dim = [76, 76, 152], resolution = 64, out_dim = prior_dim)
        self.apa2 = APA(in_fea_dim = ngf*2, in_rep_dim = [76, 76, 152], resolution = 128, out_dim = prior_dim)
        self.apa3 = APA(in_fea_dim = ngf,   in_rep_dim = [76, 76, 152], resolution = 256, out_dim = prior_dim)

        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=output_nc, kernel_size=7, padding=0)
        )

    def forward(self, inputs, rep_c_list):
        F_list = list()
        P_list = list()

        inputs_list = list()
        inputs = self.encoder1(inputs)
        inputs_list.append(inputs)

        inputs = self.encoder2(inputs)
        inputs_list.append(inputs)

        inputs = self.encoder3(inputs)
        inputs_list.append(inputs)

        x = inputs
        for i in range(4):
            x = self.middle[i](x)

        p1 = self.apa1(x, rep_c_list)
        F_list.append(x)
        P_list.append(p1)
        x = x + inputs_list[-1]
        x = self.decoder1(x=x, priors = p1)
        
        x = self.up1(x)
        p2 = self.apa2(x, rep_c_list)
        F_list.append(x)
        P_list.append(p2)
        x = x + inputs_list[-2]
        x = self.decoder2(x=x, priors = p2)

        x = self.up2(x)
        p3 = self.apa3(x, rep_c_list)
        F_list.append(x)
        P_list.append(p3)
        x = x + inputs_list[-3]
        x = self.decoder3(x=x, priors = p3)

        x = self.out(x)
        x = (torch.tanh(x) + 1) / 2
        return x, F_list, P_list
