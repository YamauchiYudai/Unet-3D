## DataHelper.py
# Author: Yudai Yamauchi, 2023, Uniklinik RWTH Aachen <yudai.yamauchi@rwth-aachen.de>
#データの Augmentation用のクラスを集めたもの

import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import nibabel as nib
from augmentation import patch_cropper


def train_dataset(img_root,label_root):
    imgs = []
    n = int(len(os.listdir(img_root))/4)
    # n = 32
    for i in range(1,n+1):
        mul_images = []
        for img_name in range (4):
            img_dir = f"{img_root}/Crop_Data_{i:0=3}_{img_name:0=4}.nii.gz"
            mul_images.append(nib.load(img_dir).get_fdata())
        label_dir = f"{label_root}/Crop_Data_{i:0=3}.nii.gz".format(i)
        nii_label = nib.load(label_dir)
        label = nii_label.get_fdata()
        imgs.append((mul_images[0],mul_images[1],mul_images[2],mul_images[3], label))
    return imgs

def test_dataset(img_root):
    imgs = []
    n = int(len(os.listdir(img_root))/4)
    for i in range(1,n+1):
        mul_images = []
        for img_name in range (4):
            img_dir = f"{img_root}/Crop_Data_{i:0=3}_{img_name:0=4}.nii.gz"
            mul_images.append(nib.load(img_dir).get_fdata())
        imgs.append((mul_images[0],mul_images[1],mul_images[2],mul_images[3]))
    return imgs


class TrainDataset(Dataset):
    def __init__(self, img_root, label_root,transform=None, target_transform=None):
        imgs = train_dataset(img_root, label_root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_x0,img_x1,img_x2,img_x3, img_y = self.imgs[index]
        if self.transform is not None:
            img_x0 = self.transform(img_x0)
            img_x1 = self.transform(img_x1)
            img_x2 = self.transform(img_x2)
            img_x3 = self.transform(img_x3)
            img_x = np.stack([img_x0,img_x1,img_x2,img_x3])
        if self.target_transform is not None:
            img_y = preprocess_mask_labels(img_y)
            img_y_0 = self.target_transform(img_y[0])
            img_y_1 = self.target_transform(img_y[1])
            img_y_2 = self.target_transform(img_y[2])
            img_y = torch.stack([img_y_0,img_y_1,img_y_2],dim = 0)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)


class TestDataset(Dataset):
    def __init__(self, img_root,transform=None):
        imgs = test_dataset(img_root)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img_x0,img_x1,img_x2,img_x3 = self.imgs[index]
        if self.transform is not None:
            img_x0 = self.transform(img_x0)
            img_x1 = self.transform(img_x1)
            img_x2 = self.transform(img_x2)
            img_x3 = self.transform(img_x3)
            img_x = np.stack([img_x0,img_x1,img_x2,img_x3])
        return img_x

    def __len__(self):
        return len(self.imgs)


def preprocess_mask_labels(mask):

    mask_WT = mask.copy()
    mask_WT[mask_WT == 1.0] = 1.0
    mask_WT[mask_WT == 2.0] = 0.0
    mask_WT[mask_WT == 4.0] = 0.0

    mask_TC = mask.copy()
    mask_TC[mask_TC == 1.0] = 0.0
    mask_TC[mask_TC == 2.0] = 1.0
    mask_TC[mask_TC == 4.0] = 0.0

    mask_ET = mask.copy()
    mask_ET[mask_ET == 1.0] = 0.0
    mask_ET[mask_ET == 2.0] = 0.0
    mask_ET[mask_ET == 4.0] = 1.0

    mask = np.stack([mask_WT, mask_TC, mask_ET])
    return mask


def labelConcatenation(img):
    after_threshold_img = np.zeros_like(img)
    for idx, val in enumerate(['1', '2', '4']):
        after_threshold_img[idx,:,:,:] = np.where(img[idx,:,:,:] > 0.5, int(val), 0)
    
    after_threshold_img[0,:,:,:] *= (img[0,:,:,:] > img[1,:,:,:]) & (img[0,:,:,:] > img[2,:,:,:])
    after_threshold_img[1,:,:,:] *= (img[1,:,:,:] > img[0,:,:,:]) & (img[1,:,:,:] > img[2,:,:,:])
    after_threshold_img[2,:,:,:] *= (img[2,:,:,:] > img[0,:,:,:]) & (img[2,:,:,:] > img[1,:,:,:])


    img_out = np.sum(after_threshold_img, axis=0)
    return img_out
