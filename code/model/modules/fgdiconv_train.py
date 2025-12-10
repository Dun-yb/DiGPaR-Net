from torch import nn
from .gpamamba import GPaMamba
from .diconv import DiConv


class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x
    
def get_norm_layer(norm_type, channels, num_groups):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)

class FGDiMBlockTrain(nn.Module):
    def __init__(self, conv, dim, kernel_size, reduction=8, norm_type='GN'):
        super(FGDiMBlockTrain, self).__init__()
        #self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.conv1 = DiConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

        self.block = nn.Sequential(
            BottConv(dim, dim, dim // 8, 1, 1, 0),
            get_norm_layer(norm_type, dim, dim // 16),
            nn.ReLU()
        )
        self.psi = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.gpamamba = GPaMamba(dim)

    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        
        x2 = self.block(x)
        gate = self.psi(x2)
        res = res * (1 + gate)
        res = res + x

        res = self.conv2(res)
        res = res.permute(0, 2, 3, 1)
        res = self.gpamamba(res)
        res = res.permute(0, 3, 1, 2)
        res = res + x
        
        return res

class FGDiBlockTrain(nn.Module):#
    def __init__(self, conv, dim, kernel_size, norm_type="GN"):
        super(FGDiBlockTrain, self).__init__()
        #self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.conv1 = DiConv(dim)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)

        self.block = nn.Sequential(
            BottConv(dim, dim, dim // 8, 1, 1, 0),
            get_norm_layer(norm_type, dim, dim // 16),
            nn.ReLU()
        )
        self.psi = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    def forward(self, x):

        res = self.conv1(x)
        res = self.act1(res)
        
        x2 = self.block(x)
        gate = self.psi(x2)
        res = res * (1 + gate)
        res = res + x

        res = self.conv2(res)
        res = res + x
        return res
