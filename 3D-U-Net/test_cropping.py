# make_cropping.py
# Author: Yudai Yamauchi, 2023, Uniklinik RWTH Aachen <yudai.yamauchi@rwth-aachen.de>
# Original code from https://github.com/tayebiarasteh/federated_he/

# make cropping image from original image 

import os
import numpy as np
import csv
from tqdm import tqdm
import nibabel as nib
from math import ceil
from scipy.ndimage import binary_fill_holes
# from file_path import train_image_dataset_path,train_label_dataset_path,test_dataset_path


train_image_dataset_path = (f"./Dataset/train/imagesTr")
train_label_dataset_path = os.path.join("./Dataset/train/labelsTr",)
test_dataset_path = ("./Dataset/test/imagesTr")


def irm_min_max_preprocess(image, low_perc=1, high_perc=99):
    """Main pre-processing function used for the challenge (seems to work the best).
    Remove outliers voxels first, then min-max scale.
    Warnings
    すべての画像に対して正規化を行う。
    --------
    This will not do it channel wise!!
    """
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)

    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale

    return image

def create_nonzero_mask(data):
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    assert len(mask.shape) == 2 or len(mask.shape) == 3, "mask must have shape (X, Y) or shape (X, Y, Z)"
    if len(mask.shape) == 3:
        mask_voxel_coords = np.where(mask != outside_value)
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    else:
        mask_voxel_coords = []
        minzidx, maxzidx = 0, 1

    minxidx = int(np.min(mask_voxel_coords[-2])) if len(mask_voxel_coords) > 0 else 0
    maxxidx = int(np.max(mask_voxel_coords[-2])) + 1 if len(mask_voxel_coords) > 0 else 1
    minyidx = int(np.min(mask_voxel_coords[-1])) if len(mask_voxel_coords) > 1 else 0
    maxyidx = int(np.max(mask_voxel_coords[-1])) + 1 if len(mask_voxel_coords) > 1 else 1

    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
    # return (minzidx, maxzidx,minxidx, maxxidx,minyidx, maxyidx)


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def make_test_cropping():
    test_path = test_dataset_path
    n = int(len(os.listdir(test_path))/4)
    os.makedirs(f'./Dataset/cropped_image/test', exist_ok=True)
    save_test_path = (f'./Dataset/cropped_image/test')
    os.makedirs(f'./csv/', exist_ok=True)
    
    for i in range(1,n+1):
        #test T1
        img_dir = f"{test_path}/Data_{i:0=3}_0000.nii.gz".format(i)
        x_input_nifti = nib.load(img_dir)
        data = x_input_nifti.get_fdata()
        data = irm_min_max_preprocess(data)
        data = np.expand_dims(data, 0)
        nonzero_mask = create_nonzero_mask(data)
        bbox = get_bbox_from_mask(nonzero_mask, 0)
        cropped_data = []
        for c in range(data.shape[0]):
            cropped = crop_to_bbox(data[c], bbox)
            cropped_data.append(cropped[None])
        data = np.vstack(cropped_data)
        cropping_af_shape = data.shape
        x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
        resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(resultt, f'{save_test_path}/Crop_Data_{i:0=3}_0000.nii.gz'.format(i)) # (h, w, d)

        #test T1nanntoka
        img_dir = f"{test_path}/Data_{i:0=3}_0001.nii.gz".format(i)
        x_input_nifti = nib.load(img_dir)
        data = x_input_nifti.get_fdata()
        data = irm_min_max_preprocess(data)
        data = np.expand_dims(data, 0)
        cropped_data = []
        for c in range(data.shape[0]):
            cropped = crop_to_bbox(data[c], bbox)
            cropped_data.append(cropped[None])
        data = np.vstack(cropped_data)
        x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
        resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(resultt, f'{save_test_path}/Crop_Data_{i:0=3}_0001.nii.gz'.format(i)) # (h, w, d)

        #test T2

        img_dir = f"{test_path}/Data_{i:0=3}_0002.nii.gz".format(i)
        x_input_nifti = nib.load(img_dir)
        data = x_input_nifti.get_fdata()
        data = irm_min_max_preprocess(data)
        data = np.expand_dims(data, 0)
        cropped_data = []
        for c in range(data.shape[0]):
            cropped = crop_to_bbox(data[c], bbox)
            cropped_data.append(cropped[None])
        data = np.vstack(cropped_data)
        x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
        resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(resultt, f'{save_test_path}/Crop_Data_{i:0=3}_0002.nii.gz'.format(i)) # (h, w, d)

        #test FIRL

        img_dir = f"{test_path}/Data_{i:0=3}_0003.nii.gz".format(i)
        x_input_nifti = nib.load(img_dir)
        data = x_input_nifti.get_fdata()
        data = irm_min_max_preprocess(data)
        data = np.expand_dims(data, 0)
        cropped_data = []
        for c in range(data.shape[0]):
            cropped = crop_to_bbox(data[c], bbox)
            cropped_data.append(cropped[None])
        data = np.vstack(cropped_data)
        x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
        resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(resultt, f'{save_test_path}/Crop_Data_{i:0=3}_0003.nii.gz'.format(i)) # (h, w, d)


        location_csv_file = f'./csv_1/Data_{i:0=3}_cropping_info.csv'
        cropping_bf_location = (bbox[0][0],bbox[0][1],bbox[1][0],bbox[1][1],bbox[2][0],bbox[2][1])#処理を後で簡単にするため
        # np.savetxt(location_csv_file, cropping_bf_location.T, delimiter=',',fmt='%d')  # 区切り文字をカンマ「,」に
        with open(location_csv_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(cropping_bf_location) #入力は['z_start', 'z_end', 'y_start', 'y_end', 'x_start', 'x_end']
            csv_writer.writerow(cropping_af_shape)#['depth', 'height', 'width']


if __name__ == '__main__':
    # make_cropping()
    make_test_cropping()
