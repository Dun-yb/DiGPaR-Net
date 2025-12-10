import math
import torch
from torch import nn
from einops.layers.torch import Rearrange


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_cd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_cd[:, :, :] = conv_weight[:, :, :]
        conv_weight_cd[:, :, 4] = conv_weight[:, :, 4] - conv_weight[:, :, :].sum(2)
        conv_weight_cd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_cd)
        return conv_weight_cd, self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_ad, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
        conv_weight_ad = conv_weight - self.theta * conv_weight[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        conv_weight_ad = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[3])(conv_weight_ad)
        return conv_weight_ad, self.conv.bias


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_hd, self).__init__() 
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_hd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_hd[:, :, [0, 3, 6]] = conv_weight[:, :, :]
        conv_weight_hd[:, :, [2, 5, 8]] = -conv_weight[:, :, :]
        conv_weight_hd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_hd)
        return conv_weight_hd, self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        super(Conv2d_vd, self).__init__() 
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 3 * 3).fill_(0)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = Rearrange('c_in c_out (k1 k2) -> c_in c_out k1 k2', k1=conv_shape[2], k2=conv_shape[2])(conv_weight_vd)
        return conv_weight_vd, self.conv.bias


class Conv2d_dd(nn.Module):
    """
    Diagonal Differential Convolution (DDC)：
    对 3×3 卷积核按照主对角线（i,j)->(j,i) 做差分，支持重参数化。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_dd, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        self.theta = theta

    def get_weight(self):
        # 原始卷积权重
        conv_weight = self.conv.weight                      # (C_out, C_in, K, K)
        conv_shape  = conv_weight.shape
        # 展平到 (C_out, C_in, K*K)
        conv_flat  = Rearrange('o i k1 k2 -> i o (k1 k2)')(conv_weight)
        # 主对角差分映射：flatten index 映射 (i*3+j)->(j*3+i)
        diag_idx = [0, 3, 6, 1, 4, 7, 2, 5, 8]
        # 差分：W - θ * W^Tdiag
        conv_flat_dd = conv_flat - self.theta * conv_flat[:, :, diag_idx]
        # 还原形状
        conv_weight_dd = Rearrange(
            'i o (k1 k2) -> o i k1 k2',
            k1=conv_shape[2], k2=conv_shape[3]
        )(conv_flat_dd)
        return conv_weight_dd, self.conv.bias


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, theta=1.0):

        super(Conv2d_rd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        weight, bias = self.get_weight()
        return nn.functional.conv2d(x, weight, bias, stride=self.conv.stride,
                                    padding=self.conv.padding, groups=self.conv.groups)

    def get_weight(self):
        if math.fabs(self.theta - 0.0) < 1e-8:
            return self.conv.weight, self.conv.bias

        with torch.no_grad():
            conv_weight = self.conv.weight  # [out_c, in_c, 3, 3]
            conv_shape = conv_weight.shape  # [out_c, in_c, 3, 3]
            device = conv_weight.device

            conv_weight_flat = conv_weight.view(conv_shape[0], conv_shape[1], -1)  # [out_c, in_c, 9]
            rd_weight_3x3 = torch.zeros_like(conv_weight)  # Same shape: [out_c, in_c, 3, 3]

            # 模拟一个简化的径向差分（仅保持中心、上下左右方向）
            rd_weight_3x3[:, :, 0, 1] = conv_weight_flat[:, :, 1]   # top
            rd_weight_3x3[:, :, 1, 0] = conv_weight_flat[:, :, 3]   # left
            rd_weight_3x3[:, :, 1, 2] = conv_weight_flat[:, :, 5]   # right
            rd_weight_3x3[:, :, 2, 1] = conv_weight_flat[:, :, 7]   # bottom
            rd_weight_3x3[:, :, 1, 1] = conv_weight_flat[:, :, 4] * (1 - self.theta)

            # 可根据需求加入 theta 调整，或进一步精确模拟 RDC 的 5×5 转换效果
            return rd_weight_3x3, self.conv.bias

class DiConv(nn.Module):
    def __init__(self, dim):
        super(DiConv, self).__init__() 
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)

    def forward(self, x):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        w = w1 + w2 + w3 + w4 + w5 
        b = b1 + b2 + b3 + b4 + b5
        res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)

        return res

