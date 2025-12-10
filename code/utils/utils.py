import cv2 
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0,mod_pad_h), 'reflect')
    return x


'''
尺度统一
当输入张量x的范围不固定，直接转化为图像可能会因为尺度问题丢失细节，
通过(x - min)/(max-min) 将其压缩到[0,1] 确保所有数值都能够按照比例映射到0-255  像素范围
保留原始数据的相对差异
'''
# 热力图部分
def norm_zero_to_one(x):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))


def save_heat_image(x, save_path, norm=False):
    if norm:
        x = norm_zero_to_one(x)
    x = x.squeeze(dim = 0) # 去除张量的批量维度，
    
    C, H, W = x.shape
    # 将张量转换为符合图像格式的NumPy数组([H, W, C], uint8类型， 值在0 - 255)
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # 将3 通道图像转换为单通道灰度图(热力图可视化需要单通道输入)
    if C == 3:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = cv2.applyColorMap(x, cv2.COLORMAP_JET)[:, :, ::-1]  # 将灰度图转为伪彩色热图，并调整色彩通道顺序。
    x = Image.fromarray(x) # 将Numpy 的数组转换为PIL
    x.save(save_path)
