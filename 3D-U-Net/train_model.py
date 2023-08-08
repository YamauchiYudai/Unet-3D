# train_model.py
# Author: Yudai Yamauchi, 2023, Uniklinik RWTH Aachen <yudai.yamauchi@rwth-aachen.de>

# show the train and validation method

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
import numpy as np
from Dice_coefficient import each_dice
from file_path import *
from DataHelper import labelConcatenation
from augmentation import patch_cropper,random_augment,resize_image
import torch.nn.functional as F

tumor_name = ["TC", "WT", "ET"]

def train_model(model, criterion, optimizer, dataload, batch_size, num_epochs):
    best_model = model
    min_loss = 1000
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch +1, num_epochs))
        epoch_loss = 0
        step = 0
        plt.ion()
        for x, y in tqdm(dataload):
            step += 1
            x = torch.squeeze(x, dim=1) #よくわからない次元が追加されているため削除している
            y = torch.squeeze(y, dim=1)
            # nib.save(nib.Nifti1Image(x.detach().float().cpu().numpy(), affine=np.eye(4)), f'checktoinput/input/validation_Data_{step:0=3}.nii.gz'.format(step))
            # nib.save(nib.Nifti1Image(y.detach().float().cpu().numpy(), affine=np.eye(4)), f'checktoinput/label/validation_Data_{step:0=3}.nii.gz'.format(step))
            x,y = patch_cropper(x,y)
            # nib.save(nib.Nifti1Image(x.detach().float().cpu().numpy(), affine=np.eye(4)), f'checktoinput/out_input/validation_Data_{step:0=3}.nii.gz'.format(step))
            # nib.save(nib.Nifti1Image(y.detach().float().cpu().numpy(), affine=np.eye(4)), f'checktoinput/out_label/validation_Data_{step:0=3}.nii.gz'.format(step))
            x,y = random_augment(x,y)
            # x = resize_image(x)
            # y = resize_image(y)
            # nib.save(nib.Nifti1Image(x.detach().float().cpu().numpy(), affine=np.eye(4)), f'checktoinput/out_input/validation_Data_{step:0=3}.nii.gz'.format(step))
            # nib.save(nib.Nifti1Image(y.detach().float().cpu().numpy(), affine=np.eye(4)), f'checktoinput/out_label/validation_Data_{step:0=3}.nii.gz'.format(step))
            inputs = x.to(device)
            labels = y.to(device)

            optimizer.zero_grad()
            outputs = model(inputs.float())
            # if step//100 == 1:
            #     nib.save(nib.Nifti1Image(inputs.detach().float().cpu().numpy(), affine=np.eye(4)), f'forward/input/validation_Data_{step:0=3}.nii.gz'.format(step))
            #     nib.save(nib.Nifti1Image(labels.detach().float().cpu().numpy(), affine=np.eye(4)), f'forward/label/validation_Data_{step:0=3}.nii.gz'.format(step))
            #     nib.save(nib.Nifti1Image(outputs.detach().float().cpu().numpy(), affine=np.eye(4)), f'forward/output/validation_Data_{step:0=3}.nii.gz'.format(step))
            loss = criterion(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()#loss.itemでlossを取得している
            writer.add_scalar("Training BCE loss",epoch_loss / step,epoch)
        print("epoch %d loss:%0.3f" % (epoch + 1, epoch_loss/step))
        if (epoch_loss/step) < min_loss:
            min_loss = (epoch_loss/step)
            best_model = model
    torch.save(best_model.state_dict(), PATH)
    return best_model

def validation_model(model, criterion, dataload, batch_size):
    model.eval()
    with torch.no_grad():
        all_val_loss = 0
        step = 0
        plt.ion()
        all_accuracy = np.zeros(3)
        for x, y in tqdm(dataload):
            step += 1
            x = torch.squeeze(x, dim=1) #よくわからない次元が追加されているため削除している
            # x = x.unsqueeze(0).expand(batch_size, -1, -1, -1, -1) #今回nn.BatchNorm3dには5次元必要でバッチサイズが含まれていなかったため
            y = torch.squeeze(y, dim=1)
            x,y = patch_cropper(x,y)
            # x = resize_image(x)
            # y = resize_image(y)
            inputs = x.to(device)
            labels = y.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs.float(), labels.float())
            all_val_loss += loss.item()#loss.itemでlossを取得している
            outputs = F.sigmoid(outputs)# loss関数にsigmoidが入っていたが、今回はないため
            np_output = np.squeeze(outputs.cpu().numpy(),0) #モデルに入れるときにバッチサイズを追加していたため削除する。
            accuracy = each_dice(outputs,labels)
            np_output = labelConcatenation(np_output)
            writer.add_scalar("validation BCE loss",all_val_loss / step,step)
            for i, mean in enumerate(tumor_name):
                writer.add_scalar(f"validation Mean {mean} Dice coefficient", accuracy[i].item(), step)
                all_accuracy[i] = all_accuracy[i] + accuracy[i].item()
            val_nii = nib.Nifti1Image(labelConcatenation(np.squeeze(labels.float().cpu().numpy(),0)), affine=np.eye(4))
            output_nii = nib.Nifti1Image(np_output, affine=np.eye(4))
            nib.save(output_nii, f'{val_result_path}/validation_Data_{step:0=3}.nii.gz'.format(step))
            nib.save(val_nii, f'{val_img_path}/validation_image_{step:0=3}.nii.gz'.format(step))
        print("validation_loss:%0.3f" % (all_val_loss/step))
        for i, mean in enumerate(tumor_name):
                writer.add_scalar(f"mean validation Mean {mean} Dice coefficient", (all_accuracy[i].item()/step))
                print(f"mean validation Mean {mean} Dice coefficient", (all_accuracy[i].item()/step))
    return (all_val_loss/step)