import cv2  #OpenCVのインポート
import numpy as np
import os
import skimage.io as io
from PIL import Image
import nibabel as nib

from DataHelper import preprocess_mask_labels

def dice_caculate(result,correct, smooth=1e-5):
    intersection = np.sum(result*correct)
    Dice = (2*intersection + smooth)/(np.sum(result) + np.sum(correct) + smooth)
    return Dice


def each_dice(result,correct):
    result = preprocess_mask_labels(result)
    correct = preprocess_mask_labels(correct) 
    n_chanel = 3
    each_dice = []
    for i in range (n_chanel):
        dice = dice_caculate(result[i,:,:,:],correct[i,:,:,:])
        each_dice.append(dice)

    return each_dice

def make_errormap():
    tissue = ['WT','TC','ET']
    for j in range(1,2):
        result_image_path = (f"./report/test_result/BraTS20_Validation_{j:03d}.nii.gz")
        correct_image_path = (f"./report/test_result/BraTS20_Training_{j:03d}_seg.nii.gz")
        os.makedirs(f"./report/error_map/map_{j:03d}", exist_ok=True)
        save_image_path = (f"./report/error_map/map_{j:03d}")

        result_image = nib.load(result_image_path).get_fdata()
        correct_image = nib.load(correct_image_path).get_fdata()
        dice_score = each_dice(result_image,correct_image)
        for z, label in enumerate(['1', '2', '4']):
            label_result  = np.where(result_image[:,:,:] == int(label), 255, 0)
            label_correct  = np.where(correct_image[:,:,:] == int(label), 255, 0)
            n,m,p = label_correct.shape
            for i in range(30,p):
                img_result = label_result[:,:,i]
                img_correct = label_correct[:,:,i]
                X = img_result - img_correct
                Y = img_correct - img_result
                Z = np.zeros((n,m), dtype= 'uint8' )
                #白黒反転
                for a in range(n):
                    for b in range(m):
                        if X[a][b]== 0 & Y[a][b] == 0:
                            X[a][b] = 255
                            Y[a][b] = 255
                            Z[a][b] = 255
                # img = [X,Y,Z]でrbgのやつにする
                img = np.stack([X,Z,Y],2)

                io.imsave(f"{save_image_path}/{i}_errormap{tissue[z]}.png", img)
        print(dice_score)

def keio():
    # np.set_printoptions(threshold=np.inf)
    tissue = ['WT','TC','ET']
    for j in range(1,2):
        result_image_path = (f"./report/test_result/BraTS20_Validation_{j:03d}.nii.gz")

        result_image = nib.load(result_image_path).get_fdata()
        for z, label in enumerate(['1', '2', '4']):
            os.makedirs(f"./report/{label}", exist_ok=True)
            save_image_path = (f"./report/{label}")
            n,m,p = result_image.shape
            for i in range(30,p):
                cv2.imwrite(f'{save_image_path}/{i}.png', np.where(result_image[:,:,i] == int(label), 255, 0))


if __name__ == "__main__":
    # make_errormap()
    keio()