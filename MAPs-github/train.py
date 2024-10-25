# python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py

import argparse
import os
import time as t
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.use_train_dataset import Data
from utils import * 
from commons import * 
from PIL import Image
from loss import get_l1_loss, get_preceptual_loss, get_mutual_information_loss, VGG16FeatureExtractor
from modules import networks
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='psv')
    parser.add_argument('--save_root', type=str, default='checkpoint')
    parser.add_argument('--restart_path', type=str)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for reproducability.')
    parser.add_argument('--num_workers', type=int, default=36, help='Number of workers.')
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=0)
    parser.add_argument('--label_w', type=float, default=0.1)
    # parser.add_argument('--device', default = 'cuda')
    
    # Train.
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--res', type=int, default=256, help='Input size.')
    parser.add_argument('--res1', type=int, default=256, help='Input size scale from.')
    parser.add_argument('--res2', type=int, default=256, help='Input size scale to.')
    parser.add_argument('--batch_size_cluster', type=int, default=64)
    parser.add_argument('--batch_size_train', type=int, default=64)
    parser.add_argument('--batch_size_test', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optim_type', type=str, default='Adam')
    parser.add_argument('--num_init_batches', type=int, default=30)
    parser.add_argument('--num_batches', type=int, default=30)
    parser.add_argument('--kmeans_n_iter', type=int, default=30)
    parser.add_argument('--in_dim', type=int, default=256)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--X', type=int, default=80)
    parser.add_argument('--train_iter_one_epoch', type=int, default=20)

    # Loss.
    parser.add_argument('--metric_train', type=str, default='cosine')
    parser.add_argument('--metric_test', type=str, default='cosine')
    parser.add_argument('--no_balance', action='store_true', default=False)
    parser.add_argument('--mse', action='store_true', default=False)
    parser.add_argument('--MSR_num', type=int, default=3)
    parser.add_argument('--EIU_num', type=int, default=3)
    
    # Dataset.
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--equiv', action='store_true', default=False)
    parser.add_argument('--min_scale', type=float, default=0.5)
    parser.add_argument('--stuff', action='store_true', default=False)
    parser.add_argument('--thing', action='store_true', default=False)
    parser.add_argument('--jitter', action='store_true', default=False)
    parser.add_argument('--grey', action='store_true', default=False)
    parser.add_argument('--blur', action='store_true', default=False)
    parser.add_argument('--h_flip', action='store_true', default=False)
    parser.add_argument('--v_flip', action='store_true', default=False)
    parser.add_argument('--random_crop', action='store_true', default=False)
    parser.add_argument('--val_type', type=str, default='train')
    parser.add_argument('--version', type=int, default=7)
    parser.add_argument('--fullcoco', action='store_true', default=False)

    # Eval-only
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str)

    # Cityscapes-specific.
    parser.add_argument('--cityscapes', action='store_true', default=False)
    parser.add_argument('--data_mode', default=3)
    parser.add_argument('--label_mode', type=str, default='gtFine')
    parser.add_argument('--long_image', action='store_true', default=False)
    
    parser.add_argument('--epoch_use_in_label', default=8)
    parser.add_argument('--epoch_use_ext_label', default=15)
    parser.add_argument('--TP', default = 0.01)
    parser.add_argument('--f_cropsize', default = 60)
    return parser.parse_args()


def train(args, trainloader, testloader, MAP_R, MAP_IN, D_net, lossNet, g_optimizer, d_optimizer, epoch, device, criterionGAN):
    iterate = 0
    criterionFeat = torch.nn.L1Loss()
    MAP_R = MAP_R.half()
    for images_256, images_512, masks, masks512, p_data in trainloader:
        iterate = iterate + 1
        images_256, images_512, masks, masks512 = images_256.to(device), images_512.to(device), masks.to(device), masks512.to(device)
        masked_image = images_256 * masks
        
        with torch.no_grad():
            gt_batch = p_data[1].cuda()
            mask_batch = p_data[2].cuda()
            mask_batch = torch.mean(mask_batch, 1, keepdim=True)
            mask_512 = F.interpolate(mask_batch, 512)
            gt_256_masked = gt_batch * (1.0 - mask_batch) + mask_batch
            gt_512_masked = F.interpolate(gt_256_masked, 512)
            labels = MAP_R(gt_batch.half(), mask_batch.half(), gt_512_masked.half(), mask_512.half())

        labels[0] = labels[0].float()
        labels[1] = labels[1].float()
        labels[2] = labels[2].float()

        inpaint_result, Fea_list, P_list = MAP_IN(torch.cat((masked_image, masks), dim = 1), labels)
        comp_result = inpaint_result * (1 - masks) + images_256 * masks
        
        ############ loss计算 ###########        
        ## 1 重构损失
        valid_loss = get_l1_loss(images_256, inpaint_result, masks)
        hole_loss = get_l1_loss(images_256, inpaint_result, 1 - masks)
        rec_loss = valid_loss + 6 * hole_loss
        
        ## 2 感知损失
        real_feats = lossNet(images_256)
        fake_feats = lossNet(inpaint_result)
        comp_feats = lossNet(comp_result)
        preceptual_loss = get_preceptual_loss(real_feats, fake_feats) + get_preceptual_loss(real_feats, comp_feats)
        # style_loss = get_style_loss(real_feats, comp_feats)

        # 3 对抗损失 and 特征匹配损失
        # Fake Detection and Loss
        if (iterate + 1) % 10 == 0:
            pred_fake_d = D_net(torch.cat((masks, comp_result.detach()), dim=1))
            loss_D_fake = criterionGAN(pred_fake_d, False)
            # Real Detection and Loss
            pred_real_d = D_net(torch.cat((masks, images_256.detach()), dim=1))
            loss_D_real = criterionGAN(pred_real_d, True)
            d_loss = (loss_D_fake + loss_D_real) / 2
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # GAN loss (Fake Passability Loss)
        pred_fake_g = D_net(torch.cat((masks, comp_result), dim=1))
        loss_G_GAN = criterionGAN(pred_fake_g, True)

        loss_G_GAN_Feat = 0
        feat_weights = 4.0 / (3 + 1)
        D_weights = 1.0 / 2

        pred_real_d = D_net(torch.cat((masks, images_256.detach()), dim=1))
        for i in range(3):
            loss_G_GAN_Feat += D_weights * feat_weights * \
                criterionFeat(pred_fake_g[i][0], pred_real_d[i][0].detach()) * 10

        ## 4 互信息损失
        mi_loss = get_mutual_information_loss(Fea_list[0], P_list[0]) + \
                    get_mutual_information_loss(Fea_list[1], P_list[1]) + \
                    get_mutual_information_loss(Fea_list[2], P_list[2])

        g_optimizer.zero_grad()
        (rec_loss + 0.1 * loss_G_GAN + 0.5 * preceptual_loss + 0.5 * loss_G_GAN_Feat + 0.1 * mi_loss).backward()
        g_optimizer.step()
        
        if (iterate + 1) % 50 == 0:
            print('Iterate: {:}, G loss: {:}, D loss: {:}'.format(iterate, loss_G_GAN.detach().cpu().numpy(), d_loss.detach().cpu().numpy()))
            
        if (iterate + 1) % 1000 == 0:
            eval(args, testloader, MAP_IN, MAP_R, epoch, device, iterate)
            torch.save({'epoch': epoch,
                        'args': args,
                        'mapin_state_dict': MAP_IN.state_dict(),
                        'mapr_state_dict': MAP_R.state_dict(),
                        'd_state_dict': D_net.state_dict(),
                        'g_optimizer': g_optimizer.state_dict(),
                        'd_optimizer': d_optimizer.state_dict(),
                        },
                       os.path.join(args.save_model_path, 'checkpoint_{:}.pth'.format((iterate + 1) // 1000)))

def eval(args, testloader, MAP_IN, MAP_R, epoch, device, iterate = 1):

    MAP_IN.eval()
    val_save_dir = 'save_{:}'.format((iterate + 1) // 1000)
    if not os.path.exists(val_save_dir):
        os.makedirs(val_save_dir)

    for i, (images_256, images_512, masks, masks512, p_data) in enumerate(testloader):
        images_256, images_512, masks, masks512 = \
            images_256.to(device), images_512.to(device), masks.to(device), masks512.to(device)
        masked_image = images_256 * masks

        with torch.no_grad():
            gt_batch = p_data[1].cuda().half()
            mask_batch = p_data[2].cuda().half()

            mask_batch = torch.mean(mask_batch, 1, keepdim=True)
            mask_512 = F.interpolate(mask_batch, 512)
            gt_256_masked = gt_batch * (1.0 - mask_batch) + mask_batch
            gt_512_masked = F.interpolate(gt_256_masked, 512)
            labels = MAP_R(gt_batch.half(), mask_batch.half(), gt_512_masked.half(), mask_512.half())
            # labels.reverse()

        labels[0] = labels[0].float()
        labels[1] = labels[1].float()
        labels[2] = labels[2].float()
        
        inpaint_result, _, _ = MAP_IN(torch.cat((masked_image, masks), dim=1), labels)
        comp_result = inpaint_result * (1 - masks) + images_256 * masks

        masked_img = postprocess(masked_image)
        real_img = postprocess(images_256)
        comp_result_img = postprocess(comp_result)

        label2img = Image.fromarray(masked_img[0])
        label2img.save(os.path.join(val_save_dir, '{:}_{:}_masked.jpg'.format(epoch, i)))

        label2img = Image.fromarray(real_img[0])
        label2img.save(os.path.join(val_save_dir, '{:}_{:}_real.jpg'.format(epoch, i)))

        label2img = Image.fromarray(comp_result_img[0])
        label2img.save(os.path.join(val_save_dir, '{:}_{:}_com.jpg'.format(epoch, i)))

        if (i + 1) % 10 == 0:
            MAP_IN.train()
            break


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(args, logger):
    logger.info(args)
    device = 'cuda'

    # Use random seed.
    fix_seed_for_reproducability(args.seed)
    
    # Get model and optimizer.
    D_net, MAP_IN, MAP_R, g_optimizer, d_optimizer = get_model_and_optimizer(args, device, checkpoint = '')
    criterionGAN = networks.GANLoss().cuda()
    lossNet = VGG16FeatureExtractor().to(device)
    
    test_set = Data(args.data_root, mode = 'test', augment = False)
    train_set = Data(args.data_root, mode = 'train')

    train_batch = 8
    num_workers = 4
    trainloader = torch.utils.data.DataLoader(train_set, 
                  batch_size = train_batch,
                  num_workers = num_workers,
                  pin_memory=True,
                  shuffle=True,
                  drop_last = False)
        
    testloader = torch.utils.data.DataLoader(test_set,
                  batch_size=1,
                  num_workers=1,
                  drop_last = False)


    for epoch in range(args.start_epoch, args.num_epoch):        
        train(args, trainloader, testloader, MAP_R, MAP_IN, D_net, lossNet, g_optimizer, d_optimizer, epoch, device, criterionGAN)
    
if __name__ == '__main__':
    args = parse_arguments()
    import torch.distributed as dist
    import os
        
    # Setup the path to save.
    if not args.pretrain:
        args.save_root += '/scratch'
    if args.augment:
        args.save_root += '/augmented/res1={}_res2={}/jitter={}_blur={}_grey={}'.format(args.res1, args.res2,
                                                                                        args.jitter, args.blur,
                                                                                        args.grey)
    if args.equiv:
        args.save_root += '/equiv/h_flip={}_v_flip={}_crop={}/min_scale\={}'.format(args.h_flip, args.v_flip,
                                                                                    args.random_crop, args.min_scale)
    if args.no_balance:
        args.save_root += '/no_balance'
    if args.mse:
        args.save_root += '/mse'

    args.save_model_path = os.path.join(args.save_root, args.comment,
                                        'train={}'.format(args.metric_train))
    args.save_eval_path = os.path.join(args.save_model_path, 'test={}'.format(args.metric_test))

    if not os.path.exists(args.save_eval_path):
        os.makedirs(args.save_eval_path)

    # Setup logger.
    logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
    
    main(args, logger)
