import os
import numpy as np 
import torch 
import torch.nn as nn 

from modules.ffc import MAPs_IN, MAP_R
from modules import networks

import warnings
from torchvision import models

warnings.filterwarnings('ignore')

def get_model_and_optimizer(args, device, checkpoint):
    # Init model 
    MAP_R_model = MAP_R().to(device)
    MAPs_IN_model = MAPs_IN().to(device)
    D_net = networks.MultiscaleDiscriminator(input_nc = 3 + 1).to(device)
    
    g_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, MAPs_IN_model.parameters()), lr = args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, D_net.parameters()), lr = args.lr, betas=(0.5, 0.999))

    if os.path.exists(checkpoint):
        dict = torch.load(checkpoint)

        model_dict = MAP_R_model.state_dict()
        checkpoint_g = {k.replace('module.', ''): v for k, v in dict['mapr_state_dict'].items()}
        model_dict.update(checkpoint_g)
        MAP_R_model.load_state_dict(model_dict)

        model_dict = MAPs_IN_model.state_dict()
        checkpoint_g = {k.replace('module.', ''): v for k, v in dict['mapin_state_dict'].items()}
        model_dict.update(checkpoint_g)
        MAPs_IN_model.load_state_dict(model_dict)

        model_dict = D_net.state_dict()
        checkpoint_d = {k.replace('module.', ''): v for k, v in dict['d_state_dict'].items()}
        model_dict.update(checkpoint_d)
        D_net.load_state_dict(model_dict)

    args.start_epoch = 1
    return D_net, MAPs_IN_model, MAP_R_model, g_optimizer, d_optimizer

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]