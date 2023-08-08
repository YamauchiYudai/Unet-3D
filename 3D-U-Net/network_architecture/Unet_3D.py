## Unet_3D.py
# Author: Yudai Yamauchi, 2023, Uniklinik RWTH Aachen <yudai.yamauchi@rwth-aachen.de>

# the model for 3D-Unet

import torch
from torch import nn

class TwoConvBlock_3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, middle_channels, kernel_size = 3, padding=1)
        self.bn1 = nn.BatchNorm3d(middle_channels)
        self.rl = nn.ReLU()
        self.conv2 = nn.Conv3d(middle_channels, out_channels, kernel_size = 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x

class UpConv_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True) #trilinearは5次元のアップサンプリング
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 2, padding="same")
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x
    

class UNet_3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.TCB1 = TwoConvBlock_3D(4, 32, 32)  #2D_UNetでは(3, 64, 64)
        self.TCB2 = TwoConvBlock_3D(32, 64, 64)
        self.TCB3 = TwoConvBlock_3D(64, 128, 128)
        self.TCB4 = TwoConvBlock_3D(128, 256, 256)
        self.TCB5 = TwoConvBlock_3D(256, 512, 512)

        self.TCB6 = TwoConvBlock_3D(512, 256, 256)
        self.TCB7 = TwoConvBlock_3D(256, 128, 128)
        self.TCB8 = TwoConvBlock_3D(128, 64, 64)
        self.TCB9 = TwoConvBlock_3D(64, 32, 32)

        self.maxpool = nn.MaxPool3d(2, stride = 2)
        
        self.UC1 = UpConv_3D(512, 256)
        self.UC2 = UpConv_3D(256, 128)
        self.UC3 = UpConv_3D(128, 64)
        self.UC4 = UpConv_3D(64, 32)

        # self.TCB1 = TwoConvBlock_3D(4, 64, 64)  #2D_UNetでは(3, 64, 64)
        # self.TCB2 = TwoConvBlock_3D(64, 128, 128)
        # self.TCB3 = TwoConvBlock_3D(128, 256, 256)
        # self.TCB4 = TwoConvBlock_3D(256, 512, 512)
        # self.TCB5 = TwoConvBlock_3D(512, 1024, 1024)

        # self.TCB6 = TwoConvBlock_3D(1024, 512, 512)
        # self.TCB7 = TwoConvBlock_3D(512, 256, 256)
        # self.TCB8 = TwoConvBlock_3D(256, 128, 128)
        # self.TCB9 = TwoConvBlock_3D(128, 64, 64)

        # self.maxpool = nn.MaxPool3d(2, stride = 2)
        
        # self.UC1 = UpConv_3D(1024,512)
        # self.UC2 = UpConv_3D(512, 256)
        # self.UC3 = UpConv_3D(256, 128)
        # self.UC4 = UpConv_3D(128, 64)

        self.conv1 = nn.Conv3d(32, 3, kernel_size = 1)
        self.drop = nn.Dropout(p=0.2)

        self.initialize_weights()

    def forward(self, x):
        x = self.TCB1(x)
        x1 = x
        x = self.maxpool(x)

        x = self.TCB2(x)
        x2 = x
        x = self.maxpool(x)

        x = self.TCB3(x)
        # x = self.drop(x)
        x3 = x
        x = self.maxpool(x)

        x = self.TCB4(x)
        # x = self.drop(x)
        x4 = x
        x = self.maxpool(x)

        x = self.TCB5(x)
        x = self.UC1(x)

        x = torch.cat([x4, x], dim = 1)
        x = self.TCB6(x)
        x = self.UC2(x)

        x = torch.cat([x3, x], dim = 1)
        x = self.TCB7(x)
        x = self.UC3(x)

        x = torch.cat([x2, x], dim = 1)
        x = self.TCB8(x)
        x = self.UC4(x)

        x = torch.cat([x1, x], dim = 1)
        x = self.TCB9(x)
        x = self.conv1(x)

        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
