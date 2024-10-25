import random
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
import glob
import os
import torchvision.transforms.functional as F
import numpy as np
from torchvision import transforms
import random
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from skimage.feature import canny
from skimage.color import rgb2gray
import torchvision
import cv2

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")

class Data(torch.utils.data.Dataset):
    def __init__(self, data_name, size=256, miss_rate=0.5,
                 mode='train_classifier', augment=True, mask_type='20_30'):
        super(Data, self).__init__()

        self.mask_transform = transforms.Compose(
            [transforms.ToTensor()])

        self.img_tf32 = transforms.Compose([
            _convert_image_to_rgb,
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        
        self.img_tf64 = transforms.Compose([
            _convert_image_to_rgb,
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        self.img_tf128 = transforms.Compose([
            _convert_image_to_rgb,
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        self.img_tf256 = transforms.Compose([
            _convert_image_to_rgb,
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.img_tf_edge256 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
                     
        self.img_tf512 = Compose([
            Resize(512, interpolation=BICUBIC),
            CenterCrop(512),
            _convert_image_to_rgb,
            ToTensor()
        ])
        
        self.mode = mode
        self.mask_mode = 2
        self.size = size
        self.miss_rate = miss_rate

        # if self.mode == 'train':
        #     self.img_paths = self.get_filelist(os.path.join('datasets', 'paris'))
        #     self.augment = True
        
        # else:
        self.img_paths = glob.glob(os.path.join('./datasets', 'paris_eval', 'paris_eval_gt', '*.png')) + \
                        glob.glob(os.path.join('./datasets', 'paris_train_original', 'paris_train_original', '*.JPG'))
        self.augment = False

        mask_paths_list = glob.glob(os.path.join('./datasets', 'masks', '0_10', '*.png')) + \
                        glob.glob(os.path.join('./datasets', 'masks', '10_20', '*.png')) + \
                        glob.glob(os.path.join('./datasets', 'masks', '20_30', '*.png')) + \
                        glob.glob(os.path.join('./datasets', 'masks', '30_40', '*.png')) + \
                        glob.glob(os.path.join('./datasets', 'masks', '40_50', '*.png'))
        
        mut_num = len(self.img_paths) // len(mask_paths_list) + 1
        self.mask_paths = list()
        for _ in range(mut_num):
            self.mask_paths.extend(mask_paths_list)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        mask_idx = random.randint(0, len(self.mask_paths) - 1)
        ## crop to 227x227 then resize to 256x256

        # load image
        mapr_img = cv2.imread(self.img_paths[index])
        mapr_img = cv2.cvtColor(mapr_img, cv2.COLOR_BGR2RGB)
        # if self.dataset_name == 'celeba':
        #     img = self.resize(img)

        mapr_img_512 = cv2.resize(mapr_img, (512, 512))
        mapr_img_256 = cv2.resize(mapr_img, (256, 256))

        # load mask
        if self.mode == 'train':
            mask = cv2.imread(self.mask_paths[mask_idx])
            mask = cv2.resize(mask, (256,256),interpolation=cv2.INTER_NEAREST)
        else:
            mask = cv2.imread(self.mask_paths[index])
            mask = cv2.resize(mask, (256,256),interpolation=cv2.INTER_NEAREST)
    
        x, y = img.size
        if x != y:
            if self.mode == 'train':
                matrix_length = min(x, y)
                x1 = random.randint(0, x - matrix_length + 1)
                y1 = random.randint(0, y - matrix_length + 1)
                img = img.crop((x1, y1, x1 + matrix_length, y1 + matrix_length))
            else:
                matrix_length = min(x, y)
                x1 = (x - matrix_length) // 2 - 1
                y1 = (y - matrix_length) // 2 - 1
                img = img.crop((x1, y1, x1 + matrix_length, y1 + matrix_length))

        if self.mode == 'train':
            img256 = self.img_tf256(img)
            img512 = self.img_tf512(img)
            
            mask256 = self.load_mask(self.mask_paths[mask_idx])
            mask256 = self.mask_transform(mask256)
            
            mask512 = self.load_mask(self.mask_paths[mask_idx], mask_size = 512)
            mask512 = self.mask_transform(mask512)

            return img256, img512, 1 - mask256, 1 - mask512, [self.to_tensor(mapr_img_512), self.to_tensor(mapr_img_256), self.to_tensor(mask)]

        else:
            img256 = self.img_tf256(img)
            img512 = self.img_tf512(img)
            
            mask256 = self.load_mask(self.mask_paths[index])
            mask256 = self.mask_transform(mask256)
            
            mask512 = self.load_mask(self.mask_paths[index], mask_size = 512)
            mask512 = self.mask_transform(mask512)

            return img256, img512, 1 - mask256, 1 - mask512, [self.to_tensor(mapr_img_512), self.to_tensor(mapr_img_256), self.to_tensor(mask)]

    def __len__(self):
        return len(self.img_paths)

    def get_filelist(self, dir, Filelist=[]):
        newDir = dir
        if os.path.isfile(dir):
            Filelist.append(dir)
        elif os.path.isdir(dir):
            for s in os.listdir(dir):
                newDir = os.path.join(dir, s)
                self.get_filelist(newDir, Filelist)
        return Filelist

    def load_mask(self, mask_path=None, mask_size = 256):
        mask = Image.open(mask_path).convert('L')
        if self.augment:
            mask = transforms.RandomHorizontalFlip()(mask)
            mask = transforms.RandomVerticalFlip()(mask)
            mask = mask.filter(ImageFilter.MaxFilter(3))
        mask = mask.resize((mask_size, mask_size))
        mask = (np.array(mask) > 0).astype(np.uint8) * 255  # threshold due to interpolation
        mask = Image.fromarray(mask).convert('L')
        return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t
