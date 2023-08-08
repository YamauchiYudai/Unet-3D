#main.py
# Author: Yudai Yamauchi, 2023, Uniklinik RWTH Aachen <yudai.yamauchi@rwth-aachen.de>

# Segmentetion program for Brain tumor using 3D Unet
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from functools import partial
from monai.inferers import sliding_window_inference

import torch.nn.functional as F
import nibabel as nib
from network_architecture.Unet_3DUnet_3D import UNet_3D
from DataHelper import *
from train_model import *
from file_path import *
from augmentation import resize_image,unresize_image

#ToTensorで正規化しているhttps://qiita.com/Haaamaaaaa/items/925257c0c8f1b115575a

roi = (160,160,160)
train_transforms = transforms.Compose([
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# device = "cpu"
batch_size = 1
num_epochs = 300
model = UNet_3D().to(device).float()
criterion = nn.BCEWithLogitsLoss()#損失関数はクロスエントロピー
# criterion = DiceLoss()
# criterion = FocalLoss(gamma=0.7)
optimizer = optim.Adam(model.parameters(), lr=0.00001)#基準で0.001が与えられているが、lossがnonを返してきたため1/10にした
cross_varidation = "False" #"True" #"False"


def train_val():
    min_loss = 10
    result_of_dataset = TrainDataset(train_image_dataset_path,train_label_dataset_path, transform=train_transforms,target_transform=train_transforms)
    # result_of_dataset = TrainDataset(train_image_dataset_path,train_label_dataset_path,use_img, transform=train_transforms,target_transform=train_transforms)
    writer.add_hparams({"criterion": str(criterion), "optimizer": str(optimizer), "batch_size": batch_size, "num_epochs": num_epochs}, {})
    if cross_varidation == "True":
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        for fold, (train_ids, val_ids) in enumerate(cv.split(result_of_dataset)):
            print(f'FOLD {fold + 1}')
            print('--------------------------------')
            train_set = torch.utils.data.Subset(result_of_dataset, train_ids)
            dataloaders = DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
            train_model(model, criterion, optimizer, dataloaders, batch_size, num_epochs)
            val_set = torch.utils.data.Subset(result_of_dataset, val_ids)
            dataloaders = DataLoader(val_set, batch_size=1)
            val_loss = validation_model(model, criterion, dataloaders,batch_size)
            if (val_loss) < min_loss:
                min_loss = (val_loss)
                best_model = model
        torch.save(best_model.state_dict(), PATH)
    else:
        not_random = KFold(n_splits=100)
        train_ids, val_ids = next(not_random.split(result_of_dataset))
        train_set = torch.utils.data.Subset(result_of_dataset, train_ids)
        dataloaders = DataLoader(train_set, batch_size, shuffle=True, num_workers=4)
        train_model(model, criterion, optimizer, dataloaders, batch_size, num_epochs)
        # model.load_state_dict(torch.load(PATH))
        val_set = torch.utils.data.Subset(result_of_dataset, val_ids)
        dataloaders = DataLoader(val_set, batch_size=1)
        val_loss = validation_model(model, criterion, dataloaders,batch_size)
        if (val_loss) < min_loss:
            min_loss = (val_loss)
            best_model = model
        torch.save(best_model.state_dict(), PATH)

scaler = torch.cuda.amp.GradScaler() #FP16への変更


def test():
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    test_dataset = TestDataset(test_dataset_path,transform=val_transforms)
    dataloaders = DataLoader(test_dataset, batch_size)
    model.eval()
    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=1,
        predictor=model,
        overlap=0.6,
    )
    plt.ion()#plt.ion()を使えば、インタラクティブモードのままで動いてくれる。
    end_num = len(test_dataset)
    with torch.no_grad():
        step = 0
        for img in dataloaders:
            step += 1
            # df = pd.read_csv(f'./csv/Data_{step:0=3}_cropping_info.csv',header=None)
            df  = np.genfromtxt(f'./csv/Data_{step:0=3}_cropping_info.csv', delimiter=',')
            # # bbox = df.iloc[0].to_numpy().astype(int)
            bbox = df.astype(int)
            img = torch.squeeze(img, dim=1)
            # output = model_inferer_test(img.to('cuda').float())
            output = model_inferer_test(img.to('cuda').float())
            output = F.sigmoid(output)
            np_output = np.squeeze(output.cpu().numpy(),0) #モデルに入れるときにバッチサイズを追加していたため削除する。
            np_output = labelConcatenation(np_output)
            val_output = np.pad(np_output, [(bbox[4], (155-bbox[5])),
                               (bbox[0], (240-bbox[1])),
                               (bbox[2], (240-bbox[3]))],
                       mode='constant')
            val_output = np.transpose(val_output, (1,2,0))
            val_output = nib.Nifti1Image(val_output, affine=np.eye(4))
            nib.save(val_output, f'{save_image_path}/BraTS20_Validation_{step:0=3}.nii.gz'.format(step))
            plt.pause(0.01)
            if (step) == end_num:
                break
        plt.show()


if __name__ == '__main__':
    print("Start train")
    print(f"criterion = {criterion},  {optimizer}")
    train_val()
    print("finish train. Save the Learned Network")
    print("-"*20)
    print("Start test")
    test()
    print("random augmentはないでごわすころん")
    print("finish test. Save the result of test")
