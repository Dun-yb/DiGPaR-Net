import torch.nn as nn 
import torch.nn.functional as F
from .modules import *
from .modules import PaRFusion
from .modules import FGDiBlockTrain,  FGDiMBlockTrain

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DiGPaRNet(nn.Module):
    def __init__(self, base_dim=32):
        super(DiGPaRNet, self).__init__()
        self.down1 = nn.Sequential(nn.Conv2d(3, base_dim, kernel_size=3, stride = 1, padding=1))
        self.down2 = nn.Sequential(nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, stride=2, padding=1),nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, stride=2, padding=1), nn.ReLU(True))

        self.up1 = nn.Sequential(nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=3, stride=2, padding=1, output_padding=1,))
        self.up3 = nn.Sequential(nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1))

        self.mix1 = PaRFusion(base_dim * 4)
        self.mix2 = PaRFusion(base_dim * 2)

        #level1
        self.down_level1_block1 = FGDiBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block2 = FGDiBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block3 = FGDiBlockTrain(default_conv, base_dim, 3)
        self.down_level1_block4 = FGDiBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block1 = FGDiBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block2 = FGDiBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block3 = FGDiBlockTrain(default_conv, base_dim, 3)
        self.up_level1_block4 = FGDiBlockTrain(default_conv, base_dim, 3)
        
        #level2
        self.fe_level_2 = nn.Conv2d(in_channels=base_dim * 2, out_channels=base_dim * 2, kernel_size=3, stride=1, padding=1)
        self.down_level2_block1 = FGDiBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block2 = FGDiBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block3 = FGDiBlockTrain(default_conv, base_dim * 2, 3)
        self.down_level2_block4 = FGDiBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block1 = FGDiBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block2 = FGDiBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block3 = FGDiBlockTrain(default_conv, base_dim * 2, 3)
        self.up_level2_block4 = FGDiBlockTrain(default_conv, base_dim * 2, 3)

        #level3
        self.fe_level_3 = nn.Conv2d(in_channels=base_dim * 4, out_channels=base_dim * 4, kernel_size=3, stride=1, padding=1)
        self.level3_block1 = FGDiMBlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block2 = FGDiMBlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block3 = FGDiMBlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block4 = FGDiMBlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block5 = FGDiMBlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block6 = FGDiMBlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block7 = FGDiMBlockTrain(default_conv, base_dim * 4, 3)
        self.level3_block8 = FGDiMBlockTrain(default_conv, base_dim * 4, 3)

    def forward(self, x):
        '''
        4, 4, 8, 4, 4
        编码器
            下采样
            重复特征提取块
            下采样
            重复特征提取块

        特征融合部分
            下采样
            重复特征提取块
            跳跃链接

        
        解码器
            上采样
            重复解码器块
            上采样
            重复解码器块
            上采样-复原图像

        '''
        # 编码器
        # 下采样
        # 重复块
        x_down1 = self.down1(x)
        x_down1 = self.down_level1_block1(x_down1)
        x_down1 = self.down_level1_block2(x_down1)
        x_down1 = self.down_level1_block3(x_down1)
        x_down1 = self.down_level1_block4(x_down1)

        # 下采样
        # 重复块
        x_down2 = self.down2(x_down1)
        x_down2_init = self.fe_level_2(x_down2)
        x_down2_init = self.down_level2_block1(x_down2_init)
        x_down2_init = self.down_level2_block2(x_down2_init)
        x_down2_init = self.down_level2_block3(x_down2_init)
        x_down2_init = self.down_level2_block4(x_down2_init)


        # 上下文聚合模块
        
        # 特征聚合模块
        # 链接方式
        x_down3 = self.down3(x_down2_init)
        x_down3_init = self.fe_level_3(x_down3)
        x1 = self.level3_block1(x_down3_init)
        x2 = self.level3_block2(x1)
        x3 = self.level3_block3(x2)
        x4 = self.level3_block4(x3)
        x5 = self.level3_block5(x4)
        x6 = self.level3_block6(x5)
        x7 = self.level3_block7(x6)
        x8 = self.level3_block8(x7)
        x_level3_mix = self.mix1(x_down3, x8)

        # 解码器部分     
        # 上采样部分
        x_up1 = self.up1(x_level3_mix)
        x_up1 = self.up_level2_block1(x_up1)
        x_up1 = self.up_level2_block2(x_up1)
        x_up1 = self.up_level2_block3(x_up1)
        x_up1 = self.up_level2_block4(x_up1)

        x_level2_mix = self.mix2(x_down2, x_up1)

        x_up2 = self.up2(x_level2_mix)
        x_up2 = self.up_level1_block1(x_up2)
        x_up2 = self.up_level1_block2(x_up2)
        x_up2 = self.up_level1_block3(x_up2)
        x_up2 = self.up_level1_block4(x_up2)

        out = self.up3(x_up2)


        return out 