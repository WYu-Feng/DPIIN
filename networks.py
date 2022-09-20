import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import backbone
from modules import SEU, SER, SE_Block, get_pad, ResnetBlock, RN_B
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
from utils import tensor_correlation, standard_scale, norm
from torchvision import models

    
class STIIN(nn.Module):
    def __init__(self, args):
        super(STIIN, self).__init__()
        self.out_dim = args.out_dim
        self.cnum = 64  # Minimum mediate feature dimension 
        self.T_middle_num = 3
        self.S_middle_num = 3
        self.temperature = args.temperature
        self.K = args.K_train
        
        self.adversarial_loss = AdversarialLoss('nsgan')
        
        self.classifier256 = nn.Linear(self.cnum, self.K, bias=False)
        self.classifier128 = nn.Linear(2 * self.cnum, self.K, bias=False)
        self.classifier64 = nn.Linear(4 * self.cnum, self.K, bias=False)
        
        # Fixed classification header parameters
        self.classifier64.weight.requires_grad  = False
        self.classifier128.weight.requires_grad = False
        self.classifier256.weight.requires_grad = False
    
        self.SE_encoder = nn.Sequential(
            SE_Block(3 + self.K + 1, self.cnum, kernel_size = 5, stride = 1),
            SE_Block(self.cnum, 2 * self.cnum, kernel_size = 3, stride = 2),
            SE_Block(2 * self.cnum, 4 * self.cnum, kernel_size = 3, stride = 2))
        
        ## input 4x256x256
        self.T_encoder1 = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(in_channels=4, out_channels=self.cnum, kernel_size=7, padding=0))
        self.T_encoder1_rn = RN_B(feature_channels=self.cnum)
        self.T_encoder1_relu = nn.ReLU(True)
        
        self.T_encoder2 = nn.Conv2d(in_channels=self.cnum, out_channels=2 * self.cnum, kernel_size=4, stride=2, padding=1)   
        self.T_encoder2_rn = RN_B(feature_channels=2 * self.cnum)
        self.T_encoder2_relu = nn.ReLU(True)
        
        self.T_encoder3 = nn.Conv2d(in_channels=2 * self.cnum, out_channels=4 * self.cnum, kernel_size=4, stride=2, padding=1)
        self.T_encoder3_rn = RN_B(feature_channels=4 * self.cnum)
        self.T_encoder3_relu = nn.ReLU(True)
        
        self.T_Rse = nn.Sequential(
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
            ResnetBlock(256, dilation=1, use_spectral_norm=False))
        
        #### Bottleneck layers ####
        ## Visual Bottleneck Layers
        for i in range(1, self.T_middle_num + 1):
            name = 'T_middle_{:d}'.format(i)
            setattr(self, name, IMU(f_in = 4 * self.cnum, layout_dim = self.K, f_out = 4 * self.cnum)) # 256, 64

        ## Semantic Bottleneck Layers
        for i in range(1, self.S_middle_num + 1):
            name = 'S_middle_{:d}'.format(i)
            setattr(self, name, SEU(4 * self.cnum, 4 * self.cnum, 4 * self.cnum)) # 256, 64
        
        ## upsample for 128 x 128
        self.up = nn.Upsample(scale_factor=2)     
        self.T_res_decoder2 = nn.Sequential(
            nn.Conv2d(self.K + 4 * self.cnum + 2 * self.cnum, 4 * self.cnum, kernel_size=1, stride=1, padding=get_pad(128, 1, 1)),
            nn.LeakyReLU(0.2))
        self.SE_up2 = nn.Sequential(
            nn.Conv2d(4 * self.cnum, 2 * self.cnum, kernel_size=3, padding=get_pad(128, 3, 1)),
            nn.LeakyReLU(0.2))
        self.T_decoder2 = IMU(f_in = 4 * self.cnum, layout_dim = self.K, f_out = 2 * self.cnum, if_2spade = False)
        
        ## upsample for 256 x 256
        self.T_res_decoder1 = nn.Sequential(
            nn.Conv2d(self.K + 2 * self.cnum + self.cnum, 2 * self.cnum, kernel_size=1, stride=1, padding=get_pad(256, 1, 1)),
            nn.LeakyReLU(0.2))
        self.SE_up1 = nn.Sequential(
            nn.Conv2d(2 * self.cnum, self.cnum, kernel_size=3, padding=get_pad(256, 3, 1)),
            nn.LeakyReLU(0.2))
        self.T_decoder1 = IMU(f_in = 2 * self.cnum, layout_dim = self.K, f_out = self.cnum, if_2spade = False)
        
        self.out = nn.Sequential(
        nn.Conv2d(self.cnum, 3, 3, 1, 1, bias = False),
        nn.Tanh())
    
    
    def forward(self, masked_image, masked_label_list, masks_list):
        masked_label_64 = masked_label_list[0]
        masked_label_128 = masked_label_list[1]
        masked_label_256 = masked_label_list[2]
        
        mask_64 = masks_list[0]
        mask_128 = masks_list[1]
        mask_256 = masks_list[2]
        
        s_fea = self.SE_encoder(torch.cat((masked_image, masked_label_256, mask_256), dim = 1))
        label_64_0 = torch.softmax(self.classifier64\
                    (F.normalize(s_fea, dim=1).permute(0,2,3,1)).permute(0,3,1,2).contiguous()/self.temperature, dim = 1)
        
        t_en1 = self.T_encoder1(torch.cat((masked_image, mask_256), dim = 1))
        t_en1 = self.T_encoder1_rn(t_en1, mask_256)
        t_en1 = self.T_encoder1_relu(t_en1)
        
        t_en2 = self.T_encoder2(t_en1)
        t_en2 = self.T_encoder2_rn(t_en2, mask_256)
        t_en2 = self.T_encoder2_relu(t_en2)
        
        t_en3 = self.T_encoder3(t_en2)
        t_en3 = self.T_encoder3_rn(t_en3, mask_256)
        t_en3 = self.T_encoder3_relu(t_en3)
        
        t_at  = self.T_Rse(t_en3) # 256x64x64
            
            
        ##  Multi-level Interactive
        _label_64_0 = masked_label_64 * mask_64 + label_64_0 * (1 - mask_64) 
        t_m2 = self.T_middle_1(t_at, _label_64_0, mask_64)
        s_m2 = self.S_middle_1(t_m2, s_fea, _label_64_0)
        label_64_1 = torch.softmax(self.classifier64\
                    (F.normalize(s_m2, dim=1).permute(0,2,3,1)).permute(0,3,1,2).contiguous()/self.temperature, dim = 1)
        
        
        _label_64_1 = masked_label_64 * mask_64 + label_64_1 * (1 - mask_64)
        t_m3 = self.T_middle_2(t_m2, _label_64_1, mask_64)
        s_m3 = self.S_middle_2(t_m3, s_m2, _label_64_1)
        label_64_2 = torch.softmax(self.classifier64\
                    (F.normalize(s_m3, dim=1).permute(0,2,3,1)).permute(0,3,1,2).contiguous()/self.temperature, dim = 1)
        
        _label_64_2 = masked_label_64 * mask_64 + label_64_2 * (1 - mask_64)
        t_m4 = self.T_middle_3(t_m3, _label_64_2, mask_64)
        s_m4 = self.S_middle_3(t_m4, s_m3, _label_64_2)
        label_64_3 = torch.softmax(self.classifier64\
                    (F.normalize(s_m4, dim=1).permute(0,2,3,1)).permute(0,3,1,2).contiguous()/self.temperature, dim = 1)
        
        ## Upsampleing
        label_up2 = self.up(label_64_3)
        label_128 = masked_label_128 * mask_128 + label_up2 * (1 - mask_128)
        t_up2 = self.up(t_m4)
        t_fea_up2 = self.T_res_decoder2(torch.cat((label_128, t_up2, t_en2), dim = 1))
        s_fea_up2 = self.SE_up2(t_fea_up2)
        label_128 = torch.softmax(self.classifier128\
                    (F.normalize(s_fea_up2, dim=1).permute(0,2,3,1)).permute(0,3,1,2).contiguous()/self.temperature, dim = 1)
        _label_128 = masked_label_128 * mask_128 + label_128 * (1 - mask_128)
        t_up2 = self.T_decoder2(t_fea_up2, label_128)
        
        label_up1 = self.up(label_128)
        label_256 = masked_label_256 * mask_256 + label_up1 * (1 - mask_256)
        t_up1 = self.up(t_up2)
        t_fea_up1 = self.T_res_decoder1(torch.cat((label_256, t_up1, t_en1), dim = 1))
        s_fea_up1 = self.SE_up1(t_fea_up1)
        label_256 = torch.softmax(self.classifier256\
                    (F.normalize(s_fea_up1, dim=1).permute(0,2,3,1)).permute(0,3,1,2).contiguous()/self.temperature, dim = 1)
        _label_256 = masked_label_256 * mask_256 + label_256 * (1 - mask_256)
        t_up1 = self.T_decoder1(t_fea_up1, label_256)
        
        out = self.out(t_up1)
        return [label_64_0, label_64_1, label_64_2, label_64_3, label_128, label_256], out
    
       
class SPE(nn.Module):
    def __init__(self, args):
        super(SPE, self).__init__()
        self.cnum = 64  # Minimum mediate feature dimension 
        self.temperature = args.temperature
        self.K = args.K_train
        
        vgg19 = models.vgg19(pretrained=True)
        self.SE_encoder256 = nn.Sequential(*vgg19.features[:4])
        self.SE_encoder128 = nn.Sequential(*vgg19.features[4:9])
        self.SE_encoder64  = nn.Sequential(*vgg19.features[9:18])
        
        self.T_encoder256 = nn.Sequential(
            nn.Conv2d(3, self.cnum, kernel_size=3, stride=1, padding=get_pad(256, 3, 1)),
            nn.ReLU(),
            nn.Conv2d(self.cnum, self.cnum, kernel_size=3, stride=1, padding=get_pad(256, 3, 1)),
            nn.ReLU())
        
        self.T_encoder128 = nn.Sequential(
            nn.Conv2d(self.cnum, 2 * self.cnum, kernel_size=3, stride=2, padding=get_pad(128, 3, 1)),
            nn.ReLU(),
            nn.Conv2d(2 * self.cnum, 2 * self.cnum, kernel_size=3, stride=1, padding=get_pad(128, 3, 1)),
            nn.ReLU())
        
        self.T_encoder64 = nn.Sequential(
            nn.Conv2d(2 * self.cnum, 4 * self.cnum, kernel_size=3, stride=2, padding=get_pad(64, 3, 1)),
            nn.ReLU(),
            nn.Conv2d(4 * self.cnum, 4 * self.cnum, kernel_size=3, stride=1, padding=get_pad(64, 3, 1)),
            nn.ReLU())
        
        self.head1 = nn.Conv2d(self.cnum, 3, kernel_size=3, stride=1, padding=get_pad(256, 3, 1)) 
        self.head2 = nn.Conv2d(2 * self.cnum, 3, kernel_size=3, stride=1, padding=get_pad(128, 3, 1))  
        self.head3 = nn.Conv2d(4 * self.cnum, 3, kernel_size=3, stride=1, padding=get_pad(64, 3, 1))     
        
        for param in self.SE_encoder256.parameters():
            param.requires_grad = False
        for param in self.SE_encoder128.parameters():
            param.requires_grad = False
        for param in self.SE_encoder64.parameters():
            param.requires_grad = False
        
        self.nonlinear_clusterer256 = nn.Sequential(
            nn.Conv2d(self.cnum + self.cnum, self.cnum, kernel_size=3, padding=get_pad(256, 3, 1)),
            nn.ReLU(),
            nn.Conv2d(self.cnum, self.cnum, kernel_size=3, padding=get_pad(256, 3, 1)))
        self.classifier256 = nn.Linear(self.cnum, self.K, bias=False)
        
        self.nonlinear_clusterer128 = nn.Sequential(
            nn.Conv2d(2 * self.cnum + 2 * self.cnum, 2 * self.cnum, kernel_size=3, padding=get_pad(128, 3, 1)),
            nn.ReLU(),
            nn.Conv2d(2 * self.cnum, 2 * self.cnum, kernel_size=3, padding=get_pad(128, 3, 1)))
        self.classifier128 = nn.Linear(2 * self.cnum, self.K, bias=False)
        
        self.nonlinear_clusterer64 = nn.Sequential(
            nn.Conv2d(4 * self.cnum + 4 * self.cnum, 4 * self.cnum, kernel_size=3, padding=get_pad(64, 3, 1)),
            nn.ReLU(),
            nn.Conv2d(4 * self.cnum, 4 * self.cnum, kernel_size=3, padding=get_pad(64, 3, 1)))
        self.classifier64 = nn.Linear(4 * self.cnum, self.K, bias=False)
        
        self.classifier64.weight.requires_grad  = False
        self.classifier128.weight.requires_grad = False
        self.classifier256.weight.requires_grad = False
        
        
    def forward(self, image256):
        sf_fea256 = self.SE_encoder256(image256).detach()
        sf_fea128 = self.SE_encoder128(sf_fea256).detach()
        sf_fea64  = self.SE_encoder64(sf_fea128).detach()

        tf_fea256 = self.SE_encoder256(image256).detach()
        tf_fea128 = self.SE_encoder128(tf_fea256).detach()
        tf_fea64  = self.SE_encoder64(tf_fea128).detach()
        
        out1 = self.head1(tf_fea256)
        out2 = self.head2(tf_fea128)
        out3 = self.head3(tf_fea64)
        
        f_fea256 = torch.cat((sf_fea256, tf_fea256), dim = 1)
        f_fea128 = torch.cat((sf_fea128, tf_fea128), dim = 1)
        f_fea64  = torch.cat((sf_fea64, tf_fea64), dim = 1)
        
        s_fea256 = self.nonlinear_clusterer256(f_fea256)
        s_fea128 = self.nonlinear_clusterer128(f_fea128)
        s_fea64  = self.nonlinear_clusterer64(f_fea64)
        
        label256 = torch.softmax(self.classifier256\
                    (F.normalize(s_fea256, dim=1).permute(0,2,3,1)).permute(0,3,1,2).contiguous()/self.temperature, dim = 1)
        label128 = torch.softmax(self.classifier128\
                    (F.normalize(s_fea128, dim=1).permute(0,2,3,1)).permute(0,3,1,2).contiguous()/self.temperature, dim = 1)
        label64  = torch.softmax(self.classifier64\
                    (F.normalize(s_fea64, dim=1).permute(0,2,3,1)).permute(0,3,1,2).contiguous()/self.temperature, dim = 1)

        return [f_fea64, f_fea128, f_fea256], [s_fea64, s_fea128, s_fea256], [label64, label128, label256], [out1, out2, out3]

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            ## norm
            fd = tensor_correlation(norm(f1), norm(f2))
            old_mean = fd.mean()
            fd -= fd.mean([3, 4], keepdim=True)
            fd = fd - fd.mean() + old_mean
            
        cd = tensor_correlation(norm(c1), norm(c2))
        min_val = 0.0
        loss = - cd.clamp(min_val) * (fd - shift)
        return loss
    
            
class D_net(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(D_net, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm)),
        )
        
        self.adversarial_loss = AdversarialLoss('nsgan')

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        outputs = x
        if self.use_sigmoid:
            outputs = torch.sigmoid(x)

        return outputs


class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss