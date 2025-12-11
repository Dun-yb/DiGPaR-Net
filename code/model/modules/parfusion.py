import math
from builtins import int
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from einops.layers.torch import Rearrange

class Par(nn.Module):
    def __init__(self, dim, r=16, L=32):                
        super().__init__()                             
        d = max(dim // r, L)                           
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2) 
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)       
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)        
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3) 
        self.conv = nn.Conv2d(dim // 2, dim, 1)        

        self.global_pool = nn.AdaptiveAvgPool2d(1)      
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)   
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),     
            nn.BatchNorm2d(d),                      
            nn.ReLU(inplace=True)                       
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False) 
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, x): 
        batch_size = x.size(0)                              
        dim = x.size(1)                                     
        attn1 = self.conv0(x)                             
        attn2 = self.conv_spatial(attn1)                   

        attn1 = self.conv1(attn1)                           
        attn2 = self.conv2(attn2)                           

        attn = torch.cat([attn1, attn2], dim=1)             
        avg_attn = torch.mean(attn, dim=1, keepdim=True)    
        max_attn, _ = torch.max(attn, dim=1, keepdim=True) 
        agg = torch.cat([avg_attn, max_attn], dim=1)     

        ch_attn1 = self.global_pool(attn)              
        z = self.fc1(ch_attn1)                        
        a_b = self.fc2(z)                               
        a_b = a_b.reshape(batch_size, 2, dim // 2, -1)  
        a_b = self.softmax(a_b)                          

        a1,a2 =  a_b.chunk(2, dim=1)                   
        a1 = a1.reshape(batch_size,dim // 2,1,1)       
        a2 = a2.reshape(batch_size, dim // 2, 1, 1)     

        w1 = a1 * agg[:, 0, :, :].unsqueeze(1)         
        w2 = a2 * agg[:, 0, :, :].unsqueeze(1)          

        attn = attn1 * w1 + attn2 * w2                  
        attn = self.conv(attn).sigmoid()                
        return x * attn                                



class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2
class PaRFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super(PaRFusion, self).__init__()
        self.ksfa = Par(dim)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        
        pattn1 = self.ksfa(initial)
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result