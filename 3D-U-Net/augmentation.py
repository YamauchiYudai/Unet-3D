# augmentation.py

# Author: Yudai Yamauchi, 2023, Uniklinik RWTH Aachen <yudai.yamauchi@rwth-aachen.de>


import pdb
import torch
import torchio as tio
from random import random
import nibabel as nib
import numpy as np
import torch.nn.functional as F

from param_list import *


patch_size = [160,160,160]

general_spatial_probability = 0.5
general_intensity_probability = 0.5
lateral_flip_prob = 0.4
AWGN_prob = 0.1
mu_AWGN= 0.0 
sigma_AWGN = 0.03
gamma_prob = 0.4
gamma_range = [0.8, 1.2]

def random_spatial_brats_augmentation(image, label):
    """Both image and the label should be augmented
    Parameters
    ----------
    image: torch tensor (n, c, d, h, w)
    label: torch tensor (n, c, d, h, w)
    confg_path: str

    Returns
    -------
    transformed_image.unsqueeze(0): torch tensor (n, c, d, h, w)
    transformed_label.unsqueeze(0): torch tensor (n, c, d, h, w)
    """

    transform = tio.transforms.RandomFlip(axes='L', flip_probability=lateral_flip_prob)
    image = transform(image)
    label = transform(label)

    transform = tio.transforms.RandomFlip(axes='I', flip_probability=lateral_flip_prob)
    image = transform(image)
    label = transform(label)

    assert len(label.unique()) < 5
    return image, label



def random_intensity_brats_augmentation(image):
    """Only image should be augmented
    """
    # additive Gaussian noise (not needed for min max normalization; 20% prob for the mean std normalization)
    # AWGN : Additive White Gaussian Noise augmentation
    if random() < AWGN_prob:
        transform = tio.RandomNoise(mean=mu_AWGN, std=sigma_AWGN)
        return transform(image)

    elif random() < gamma_prob:
        # transform = tio.RandomGamma(log_gamma=(params['augmentation']['gamma_range'][0], params['augmentation']['gamma_range'][1]))
        X_new = torch.zeros(image.shape)
        for c in range(image.shape[0]):
            im = image[c, :, :, :]
            gain, gamma = (gamma_range[1] - gamma_range[0]) * torch.rand(2) + gamma_range[0]
            im_new = torch.sign(im) * gain * (torch.abs(im) ** gamma)
            X_new[c, :, :, :] = im_new
        return X_new

    else:
        return image





def random_augment(image, label):
    """
    Parameters
    ----------
    image: torch tensor (n, c, d, h, w)
    label: torch tensor (n, c, d, h, w)
    confg_path: str

    Returns
    -------
    transformed_image_list: torch tensor (n, c, d, h, w)
    transformed_label_list: torch tensor (n, c, d, h, w)
    """
    transformed_image_list = []
    transformed_label_list = []

    for image_file, label_file in zip(image, label):

        if random() < general_spatial_probability:
            image_file, label_file = random_spatial_brats_augmentation(image_file, label_file)

            image_file = image_file.float()
            label_file = label_file.long()

        if random() < general_intensity_probability:
            image_file = random_intensity_brats_augmentation(image_file)

            image_file = image_file.float()
            label_file = label_file.long()

        transformed_image_list.append(image_file)
        transformed_label_list.append(label_file)

    transformed_image_list = torch.stack((transformed_image_list), 0)
    transformed_label_list = torch.stack((transformed_label_list), 0)

    return transformed_image_list, transformed_label_list





def patch_cropper(image, label):
    """
    Parameters
    ----------
    image: torch tensor (n, c, d, h, w)
    label: torch tensor (n, c, d, h, w)
    confg_path: str

    Returns
    -------
    transformed_image_list: torch tensor (n, c, d, h, w)
    transformed_label_list: torch tensor (n, c, d, h, w)
    """

    image_list = []
    label_list = []

    # cropping (patching)
    for image_file, label_file in zip(image, label):

        patch_d, patch_h, patch_w = patch_size
        batch_size, channels, slices, rows, columns = image.shape

        if columns < patch_w:
            diff = patch_w - columns
            columns = patch_w
            image_file = F.pad(image_file, (0, diff), "constant", 0)
            label_file = F.pad(label_file, (0, diff), "constant", 0)
        if rows < patch_h:
            diff2 = patch_h - rows
            rows = patch_h
            image_file = F.pad(image_file, (0, 0, 0, diff2), "constant", 0)
            label_file = F.pad(label_file, (0, 0, 0, diff2), "constant", 0)
        if slices < patch_d:
            diff3 = patch_d - slices
            slices = patch_d
            image_file = F.pad(image_file, (0, 0, 0, 0, 0, diff3), "constant", 0)
            label_file = F.pad(label_file, (0, 0, 0, 0, 0, diff3), "constant", 0)

        dd = np.random.randint(slices - patch_d + 1)
        hh = np.random.randint(rows - patch_h + 1)
        ww = np.random.randint(columns - patch_w + 1)
        image_file = image_file[:, dd:dd + patch_d, hh:hh + patch_h, ww:ww + patch_w]
        label_file = label_file[:, dd:dd + patch_d, hh:hh + patch_h, ww:ww + patch_w]

        image_file = image_file.float()
        label_file = label_file.long()
        image_list.append(image_file)
        label_list.append(label_file)

    image_list = torch.stack((image_list), 0)
    label_list = torch.stack((label_list), 0)

    return image_list, label_list

def test_cropper(image):


    image_list = []

    # cropping (patching)
    for image_file in zip(image,):

        patch_d, patch_h, patch_w = patch_size
        batch_size, channels, slices, rows, columns = image.shape
        image_file = image_file[0] #input iamge is tuple. we have to extract Tensor
        if columns < patch_w:
            diff = patch_w - columns
            columns = patch_w
            image_file = F.pad(image_file, (0, diff), "constant", 0)
        if rows < patch_h:
            diff2 = patch_h - rows
            rows = patch_h
            image_file = F.pad(image_file, (0, 0, 0, diff2), "constant", 0)
        if slices < patch_d:
            diff3 = patch_d - slices
            slices = patch_d
            image_file = F.pad(image_file, (0, 0, 0, 0, 0, diff3), "constant", 0)

        dd = np.random.randint(slices - patch_d + 1)
        hh = np.random.randint(rows - patch_h + 1)
        ww = np.random.randint(columns - patch_w + 1)
        image_file = image_file[:, dd:dd + patch_d, hh:hh + patch_h, ww:ww + patch_w]

        image_file = image_file.float()
        image_list.append(image_file)

    image_list = torch.stack((image_list), 0)

    return image_list,dd,hh,ww

