# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 14:54:17 2018

@author: Franc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    
    def __init__(self):
        super(ResNet, self).__init__()
        self.Basic_stem = BasicStem(False)
        self.Conv2d_block_a1 = BottleBlock(64, 64)
#        self.Conv2d_block_a2 = BottleBlock(256, 64)
        self.Conv2d_block_a3 = BottleBlock(256, 64)
        
        self.Conv2d_block_b1 = BottleBlock(256, 128)
        self.Conv2d_block_b2 = BottleBlock(512, 128, stride = 2)
#        self.Conv2d_block_b3 = BottleBlock(512, 128)
        self.Conv2d_block_b4 = BottleBlock(512, 128)
        
        self.Conv2d_block_c1 = BottleBlock(512, 256)
        self.Conv2d_block_c2 = BottleBlock(1024, 256, stride = 2)
        self.Conv2d_block_c3 = BottleBlock(1024, 256)
        self.Conv2d_block_c4 = BottleBlock(1024, 256)
        self.Conv2d_block_c5 = BottleBlock(1024, 256)
#        self.Conv2d_block_c6 = BottleBlock(1024, 256)
        
        self.Conv2d_block_d1 = BottleBlock(1024, 512)
        self.Conv2d_block_d2 = BottleBlock(2048, 512, stride = 2)
        self.Conv2d_block_d3 = BottleBlock(2048, 512)
        
        self.fc = nn.Linear(2048, 4)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_() 
    
    def forward(self, x):
        x = self.Basic_stem(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        x = self.Conv2d_block_a1(x)
        x = self.Conv2d_block_a2(x)
#        x = self.Conv2d_block_a3(x)
        
        x = self.Conv2d_block_b1(x)
        x = self.Conv2d_block_b2(x)
        x = self.Conv2d_block_b3(x)
#        x = self.Conv2d_block_b4(x)
        
        x = self.Conv2d_block_c1(x)
        x = self.Conv2d_block_c2(x)
        x = self.Conv2d_block_c3(x)
        x = self.Conv2d_block_c4(x)
        x = self.Conv2d_block_c5(x)
#        x = self.Conv2d_block_c6(x)
        
        x = self.Conv2d_block_d1(x)
        x = self.Conv2d_block_d2(x)
        x = self.Conv2d_block_d3(x)
        print(x.shape)
        x = F.avg_pool2d(x, kernel_size=5)
        x = F.dropout(x, p = 0.2)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        #x = F.softmax(x, dim=1)
        return x
        
                
class BasicStem(nn.Module):
    
    def __init__(self, transform_input = False):
        self.transform_input = transform_input
        super(BasicStem, self).__init__()
        self.Conv2d_1_7x7 = BasicConv2d(3, 16, kernel_size=7, stride=3, padding=3)
        self.Conv2d_2_7x7 = BasicConv2d(16, 16, kernel_size=7, padding=3)
        self.Conv2d_3_7x7 = BasicConv2d(16, 32, kernel_size=7, stride=3, padding=3)
        self.Conv2d_4_7x7 = BasicConv2d(32, 32, kernel_size=7, padding=3)
    
        self.Conv2d_5_3x3 = BasicConv2d(32, 64, kernel_size=7, 
                                        stride = 2, bias=False)
    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = self.Conv2d_1_7x7(x)
        x = self.Conv2d_2_7x7(x)
        x = self.Conv2d_3_7x7(x)
        x = self.Conv2d_4_7x7(x)
        x = self.Conv2d_5_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class BasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample      
        self.Conv2d_1_3x3 = BasicConv2d(in_channels, out_channels, activation = True,
                                        kernel_size = 3, padding = 1)
        self.Conv2d_2_3x3 = BasicConv2d(out_channels, out_channels, activation = False,
                                        kernel_size = 3, padding = 1)

    def forward(self, x):
        residual = x
        if self.stride != 1:
            self.downsample = BasicConv2d(self.in_channels, self.out_channels,
                                          kernel_size=1, stride=self.stride, 
                                          activation = False, bias=False)
            residual = self.downsample(residual)
            
        x = self.Conv2d_1_3x3(x)
        x = self.Conv2d_2_3x3(x)
        x += residual
        
        x = F.relu(x)
        return x


class BottleBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride = 1):
        super(BottleBlock, self).__init__() 
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = BasicConv2d(in_channels, out_channels*4, activation = False, 
                                      kernel_size=1, stride=stride, bias=False)
        self.Conv2d_1_1x1 = BasicConv2d(in_channels, out_channels, activation = True,
                                        kernel_size = 1)
        self.Conv2d_2_3x3 = BasicConv2d(out_channels, out_channels, activation = True,
                                        kernel_size = 3, stride = self.stride, padding = 1)
        self.Conv2d_3_1x1 = BasicConv2d(out_channels, out_channels*4, activation = False, 
                                        kernel_size = 1)

    def forward(self, x):
        residual = x
        residual = self.downsample(residual)
    
        x = self.Conv2d_1_1x1(x)
        x = self.Conv2d_2_3x3(x)
        x = self.Conv2d_3_1x1(x)
        
        x += residual
        x = F.relu(x)
        return x

class BasicConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, activation = True, **kwargs):
        self.activation = activation
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = F.relu(x)
        return x