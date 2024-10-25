import torch
import torch.nn as nn
from torchvision import models

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # vgg16 = models.vgg16(pretrained=True)
        vgg16 = models.vgg16(pretrained=False)  # 创建一个空的VGG16模型
        vgg16.load_state_dict(torch.load('vgg16.pth'))

        
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
  
def get_l1_loss(f1, f2, mask = 1):
    return torch.mean(torch.abs(f1 - f2)*mask)

def get_style_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        _, c, w, h = A_feat.size()
        A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
        B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
        A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
        B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
        loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
    return loss_value

def get_TV_loss(x):
    h_x = x.size(2)
    w_x = x.size(3)
    h_tv = torch.mean(torch.abs(x[:,:,1:,:]-x[:,:,:h_x-1,:]))
    w_tv = torch.mean(torch.abs(x[:,:,:,1:]-x[:,:,:,:w_x-1]))
    return h_tv + w_tv

def get_preceptual_loss(A_feats, B_feats):
    assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
    loss_value = 0.0
    for i in range(len(A_feats)):
        A_feat = A_feats[i]
        B_feat = B_feats[i]
        loss_value += torch.mean(torch.abs(A_feat - B_feat))
    return loss_value

def get_mutual_information_loss(x1, x2, num_bins=30, eps=1e-10):
    B, C1, H, W = x1.shape
    B, C2, H, W = x2.shape
    
    x1_flat = x1.view(B, -1)
    x2_flat = x2.view(B, -1)

    def compute_hist2d(x1, x2, num_bins):
        joint_hist = torch.histc(torch.cat((x1, x2), dim=1), bins=num_bins, min=0, max=1)
        x1_hist = torch.histc(x1, bins=num_bins, min=0, max=1)
        x2_hist = torch.histc(x2, bins=num_bins, min=0, max=1)
        return joint_hist, x1_hist, x2_hist

    joint_hist, x1_hist, x2_hist = compute_hist2d(x1_flat, x2_flat, num_bins)
    joint_prob = joint_hist / torch.sum(joint_hist)
    x1_prob = x1_hist / torch.sum(x1_hist)
    x2_prob = x2_hist / torch.sum(x2_hist)
    mutual_info = torch.sum(joint_prob * (torch.log(joint_prob + eps) - torch.log(x1_prob + eps) - torch.log(x2_prob + eps)))
    
    mi_loss = -mutual_info
    return mi_loss