import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import numpy as np
    

class SE_Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 3, stride = 2, padding = 1, if_bn = True, if_ac = True, if_upsample = False):
        super().__init__()
        self.encoder = nn.Conv2d(in_ch, out_ch, kernel_size, stride = stride, padding=(kernel_size - 1)//2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.ac = nn.ReLU(False)
        self.if_bn = if_bn
        self.if_ac = if_ac
        self.upsample = if_upsample
    
    def forward(self, inputs):
        if self.upsample:
            inputs = F.interpolate(inputs, scale_factor=2)
        outputs = self.encoder(inputs)
        if self.if_bn:
            outputs = self.bn(outputs)
        if self.if_ac:
            outputs = self.ac(outputs)
        return outputs


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm)),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm)),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
    
    
class SEU(nn.Module):
    def __init__(self, img_dim, se_dim, out_dim):
        super().__init__()
        self.mlp_se = nn.Sequential(
            nn.Conv2d(img_dim + se_dim, out_dim, kernel_size = 3, padding = 3//2),
            nn.LeakyReLU(0.2))

    def forward(self, x, se, se_k):
        se_k_max = torch.max(se_k, dim = 1)[0].unsqueeze(1)
        out = self.mlp_se(torch.cat((x, se * se_k_max), dim = 1))
        out = out = se * se_k_max + (1 - se_k_max) * out
        return out

class IMU(nn.Module):
    def __init__(self, f_in, layout_dim, f_out, if_2spade = True):
        super(IMU, self).__init__()
        
        f_middle = min(f_in, f_out)
        
        self.spaed_block1 = SPADE(f_in, layout_dim)
        self.spaed_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=f_in, out_channels=f_middle, kernel_size=3, padding=1),
            nn.ReLU())
        
        self.spaed_block2 = SPADE(f_middle, layout_dim)
        self.spaed_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=f_middle, out_channels=f_out, kernel_size=3, padding=1),
            nn.ReLU())        
        
        self.out = nn.Conv2d(in_channels=f_out + f_out + f_in, out_channels=f_out, kernel_size=3, stride=1, padding=get_pad(64, 3, 1))

        
    def forward(self, x, layout, mask=None):
        out1 = self.spaed_block1(x , layout)
        out1 = self.spaed_conv1(out1)
        
        out2 = self.spaed_block2(out1 , layout)
        out2 = self.spaed_conv2(out2)        
        
        out_final = self.out(torch.cat((x, out1, out2), dim = 1))
        return out_final

    
def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

class SPADE(nn.Module):

    def __init__(self, norm_nc, label_nc):

        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)
        # self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        #actv = segmap
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class RN_binarylabel_IN(nn.Module):
    def __init__(self, feature_channels):
        super(RN_binarylabel_IN, self).__init__()
        self.IN_norm = nn.InstanceNorm2d(feature_channels, affine=False, track_running_stats=False) #, track_running_stats=False

    def forward(self, x, label):
        '''
        input:  x: (B,C,M,N), features
                label: (B,1,M,N), 1 for foreground regions, 0 for background regions
        output: _x: (B,C,M,N)
        '''
        label = label.detach()

        rn_foreground_region = self.rn(x * label, label)

        rn_background_region = self.rn(x * (1 - label), 1 - label)

        return rn_foreground_region + rn_background_region

    def rn(self, region, mask):
        '''
        input:  region: (B,C,M,N), 0 for surroundings
                mask: (B,1,M,N), 1 for target region, 0 for surroundings
        output: rn_region: (B,C,M,N)
        '''
        shape = region.size()

        sum = torch.sum(region, dim=[2,3])  # (B, C) -> (B, C)
        Sr = torch.sum(mask, dim=[2,3])    # (B, 1) -> (B, 1)
        Sr[Sr==0] = 1
        mu = (sum / Sr)     # (B, C) -> (B, C)

        return self.IN_norm(region + (1 - mask) * mu[:,:,None,None]) * \
        (torch.sqrt(Sr / (shape[2] * shape[3])))[:,:,None,None]

class RN_B(nn.Module):
    def __init__(self, feature_channels):
        super(RN_B, self).__init__()
        '''
        input: tensor(features) x: (B,C,M,N)
               condition Mask: (B,1,H,W): 0 for background, 1 for foreground
        return: tensor RN_B(x): (N,C,M,N)
        ---------------------------------------
        args:
            feature_channels: C
        '''
        # RN
        self.rn = RN_binarylabel_IN(feature_channels)    # need no external parameters

        # gamma and beta
        self.foreground_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)

    def forward(self, x, mask):
        # mask = F.adaptive_max_pool2d(mask, output_size=x.size()[2:])
        mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')   # after down-sampling, there can be all-zero mask

        rn_x = self.rn(x, mask)

        rn_x_foreground = (rn_x * mask) * (1 + self.foreground_gamma[None,:,None,None]) + self.foreground_beta[None,:,None,None]
        rn_x_background = (rn_x * (1 - mask)) * (1 + self.background_gamma[None,:,None,None]) + self.background_beta[None,:,None,None]

        return rn_x_foreground + rn_x_background
    
    
class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='relu',
                 conv_bias=False):
        super().__init__()
        if sample == 'down-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 2, 2, bias=conv_bias)
        elif sample == 'down-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 2, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.conv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask

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