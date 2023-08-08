# Dice_coefficient.py
# Author: Yudai Yamauchi, 2023, Uniklinik RWTH Aachen <yudai.yamauchi@rwth-aachen.de>

# caculate the dice coefficients. each_dice return dice coefficients for each tumor

import os
import cv2
import numpy as np



def dice_caculate(result,correct, smooth=1e-5):
    intersection = np.sum(result*correct)
    Dice = (2*intersection + smooth)/(np.sum(result) + np.sum(correct) + smooth)
    return Dice



def each_dice(result,correct):
    result = result.cpu().detach().numpy() 
    correct = correct.cpu().detach().numpy() 
    each_dice = []
    n_chanel = 3
    for i in range (n_chanel):
        dice = dice_caculate(result[0,i,:,:,:],correct[0,i,:,:,:])
        each_dice.append(dice)

    return each_dice