# file_path.py

# Author: Yudai Yamauchi, 2023, Uniklinik RWTH Aachen <yudai.yamauchi@rwth-aachen.de>



import os
import torch
import datetime

from tensorboardX import SummaryWriter




# 現在の日付と時刻を取得
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_image_dataset_path = (f"./Dataset/cropped_image/imagesTr")
train_label_dataset_path = os.path.join("./Dataset/cropped_image/labelsTr",)
os.makedirs(f'./Dataset/validation/multi', exist_ok=True)
val_result_path = (f'./Dataset/validation/multi')
os.makedirs(f'./Dataset/valimage/multi', exist_ok=True)
val_img_path = (f'./Dataset/valimage/multi')
test_dataset_path = ("./Dataset/cropped_image/test")


os.makedirs(f'./Dataset/test_result/multi', exist_ok=True)
save_image_path = (f'./Dataset/test_result/multi')
os.makedirs(f"./Dataset/test_accuracy", exist_ok=True)
save_accuracy_path = os.path.join(f"./Dataset/test_accuracy/multi.scv")
os.makedirs(f"./tensorboard", exist_ok=True)
writer = SummaryWriter(log_dir = f"./tensorboard/multi_log/{current_time}")
os.makedirs(f"./network_model", exist_ok=True)
PATH = (f'./network_model/unet_model_multi.pt')

device = torch.device("cuda")