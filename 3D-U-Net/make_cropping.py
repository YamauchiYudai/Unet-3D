# make_cropping.py
# Author: Yudai Yamauchi, 2023, Uniklinik RWTH Aachen <yudai.yamauchi@rwth-aachen.de>
# Original code from https://github.com/tayebiarasteh/federated_he/

# make cropping image from original image 

import os
import numpy as np
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

"""
create_nonzero_mask(): 与えられたデータの非ゼロ領域をマスクとして作成する。
nonzero_maskという名前のブール型のゼロ行列を作成します。
この行列は、後で非ゼロの領域を表すバイナリマスクを格納するために使用されます
data.shape[1:]は、データの形状の2番目から最後までを取得し、その形状に合わせてゼロ行列を作成します。
forループを使用して、データの各チャンネルに対して以下の処理を行います：
    this_maskという名前のブール型のマスクを作成します。
    このマスクは、data[c]がゼロでない要素の場所をTrue、
    ゼロの要素の場所をFalseで表します
    nonzero_maskとthis_maskを論理和（|）演算子で結合し、
    nonzero_maskを更新します。これにより、すべてのチャンネルで非ゼロの要素が存在する位置がTrueとなります。
最後に、binary_fill_holes関数を使用して、nonzero_mask内の空洞（ゼロの領域）を埋めます。
これにより、オブジェクトの領域が完全に閉じられ、連結したバイナリマスクが得られます。
最終的に、作成された非ゼロマスク（連結したバイナリマスク）を返します。


"""


def get_bbox_from_mask( mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

"""
get_bbox_from_mask(): マスクからバウンディングボックス（切り抜き範囲）を取得します。
X,Y,Z軸に対して最小と最大のインデックスを計算しバウンディングボックスの範囲を計算する。
1を追加する理由は最大値もバウンディングボックスに追加するため
"""

def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def make_cropping():
    image_path = train_image_dataset_path
    label_path = train_label_dataset_path
    n = int(len(os.listdir(image_path))/4)
    os.makedirs(f'./Dataset/cropped_image/imagesTr', exist_ok=True)
    save_image_path = (f'./Dataset/cropped_image/imagesTr')
    os.makedirs(f'./Dataset/cropped_image/labelsTr', exist_ok=True)
    save_label_path = (f'./Dataset/cropped_image/labelsTr')
    os.makedirs(f'./Dataset/cropped_image/test', exist_ok=True)
    save_test_path = (f'./Dataset/cropped_image/test')
    
    for i in range(1,n+1):
        # T1 image
        img_dir = f"{image_path}/Data_{i:0=3}_0000.nii.gz".format(i)
        x_input_nifti = nib.load(img_dir)
        data = x_input_nifti.get_fdata() # (h, w, d)
        data = irm_min_max_preprocess(data)
        data = np.expand_dims(data, 0) # (1, h, w, d)
        nonzero_mask = create_nonzero_mask(data)
        bbox = get_bbox_from_mask(nonzero_mask, 0)
        cropped_data = []
        for c in range(data.shape[0]):
            cropped = crop_to_bbox(data[c], bbox)
            cropped_data.append(cropped[None])
        data = np.vstack(cropped_data) # (1, h, w, d)
        x_input_nifti.header['dim'][1:4] = np.array(data[0].shape)
        resultt = nib.Nifti1Image(data[0], affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(resultt, f'{save_image_path}/Crop_Data_{i:0=3}_0000.nii.gz'.format(i)) # (h, w, d)

        # T1nanntoka
        img_dir = f"{image_path}/Data_{i:0=3}_0001.nii.gz".format(i)
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
        nib.save(resultt, f'{save_image_path}/Crop_Data_{i:0=3}_0001.nii.gz'.format(i)) # (h, w, d)

        #T2

        img_dir = f"{image_path}/Data_{i:0=3}_0002.nii.gz".format(i)
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
        nib.save(resultt, f'{save_image_path}/Crop_Data_{i:0=3}_0002.nii.gz'.format(i)) # (h, w, d)

        #FIRL

        img_dir = f"{image_path}/Data_{i:0=3}_0003.nii.gz".format(i)
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
        nib.save(resultt, f'{save_image_path}/Crop_Data_{i:0=3}_0003.nii.gz'.format(i)) # (h, w, d)

        # full size seg-label
        path_file = f"{label_path}/Data_{i:0=3}.nii.gz".format(i)
        x_input_nifti = nib.load(path_file)
        data = x_input_nifti.get_fdata() # (h, w, d)
        data = np.expand_dims(data, 0)
        cropped_data = []
        for c in range(data.shape[0]):
            cropped = crop_to_bbox(data[c], bbox)
            cropped_data.append(cropped[None])
        data = np.vstack(cropped_data)
        data = np.squeeze(data)
        # print(data.shape)
        x_input_nifti.header['dim'][1:3] = np.array(data[0].shape)
        resultt = nib.Nifti1Image(data.astype(np.uint8), affine=x_input_nifti.affine, header=x_input_nifti.header)
        nib.save(resultt, f'{save_label_path}/Crop_Data_{i:0=3}.nii.gz'.format(i)) # (h, w, d)

def make_test_cropping():
    test_path = test_dataset_path
    n = int(len(os.listdir(test_path))/4)
    os.makedirs(f'./Dataset/cropped_image/test', exist_ok=True)
    save_test_path = (f'./Dataset/cropped_image/test')
    
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

if __name__ == '__main__':
    # make_cropping()
    make_test_cropping()