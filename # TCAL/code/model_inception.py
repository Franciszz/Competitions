# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 10:32:17 2018

@author: Franc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo

class InceptionNet(nn.Module):
    
    def __init__(self, num_classes = 4):
        super(InceptionNet, self).__init__()
        self.Pre_stem = PreStem(False)
        self.Basic_stem = BasicStem(32)
        self.Inception_A1 = InceptionA(384)
        self.Inception_A2 = InceptionA(384)
        self.Inception_A3 = InceptionA(384)
        self.Reduction_A = ReductionA(384)
        self.Inception_B1 = InceptionB(1024)
        self.Inception_B2 = InceptionB(1024)
        self.Inception_B3 = InceptionB(1024)
        self.Inception_B4 = InceptionB(1024)
        self.Inception_B5 = InceptionB(1024)
        self.Reduction_B = ReductionB(1024)
        self.Inception_C1 = InceptionC(1536)
        self.Inception_C2 = InceptionC(1536)
        self.Inception_C3 = InceptionC(1536)
        self.fc = nn.Linear(1536, num_classes)
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
        x = self.Pre_stem(x)
        x = self.Basic_stem(x)
        x = self.Inception_A1(x)
        x = self.Inception_A2(x)
        x = self.Inception_A3(x)
        x = self.Reduction_A(x)
        x = self.Inception_B1(x)
        x = self.Inception_B2(x)
        x = self.Inception_B3(x)
        x = self.Inception_B4(x)
        x = self.Inception_B5(x)
        x = self.Reduction_B(x)
        x = self.Inception_C1(x)
        x = self.Inception_C2(x)
        x = self.Inception_C3(x)
        x = F.avg_pool2d(x, kernel_size=9)
        x = F.dropout(x, p = 0.2)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        #x = F.softmax(x, dim=1)
        return x
        

class BasicConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwards):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwards)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class PreStem(nn.Module):
    
    def __init__(self, transform_input = False):
        super(PreStem, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1_7x7 = BasicConv2d(3, 16, kernel_size=7, stride=3, padding=3)
        self.Conv2d_2_7x7 = BasicConv2d(16, 16, kernel_size=7, padding=3)
        self.Conv2d_3_7x7 = BasicConv2d(16, 32, kernel_size=7, stride=3, padding=3)
        self.Conv2d_4_7x7 = BasicConv2d(32, 32, kernel_size=7, padding=3)
    
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
        return x
        

class BasicStem(nn.Module):
    
    def __init__(self, in_channels):
        super(BasicStem, self).__init__()
        self.Conv2d_5_3x3 = BasicConv2d(in_channels,32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_6_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_7_3x3 = BasicConv2d(32, 64, kernel_size=3)
        
        self.Conv2d_8b_3x3 = BasicConv2d(64, 96, kernel_size=3, stride=2)
        
        self.Conv2d_10a1_1x1 = BasicConv2d(160, 64, kernel_size=1)
        self.Conv2d_10a2_3x3 = BasicConv2d(64, 96, kernel_size=3)
        
        self.Conv2d_10b1_1x1 = BasicConv2d(160, 64, kernel_size=1)
        self.Conv2d_10b2_7x1 = BasicConv2d(64, 64, kernel_size=(7,1), padding=(2,0))
        self.Conv2d_10b3_1x7 = BasicConv2d(64, 64, kernel_size=(1,7), padding=(0,2))
        self.Conv2d_10b4_3x3 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        
        self.Conv2d_12a_3x3 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        
    def forward(self, x):
        x = self.Conv2d_5_3x3(x)
        x = self.Conv2d_6_3x3(x)
        x = self.Conv2d_7_3x3(x)
        x_1a = F.max_pool2d(x, kernel_size=3, stride=2)
        x_1b = self.Conv2d_8b_3x3(x)
        
        x = torch.cat([x_1a, x_1b], 1)
        
        x_2a = self.Conv2d_10a1_1x1(x)
        x_2a = self.Conv2d_10a2_3x3(x_2a)
        x_2b = self.Conv2d_10b1_1x1(x)
        x_2b = self.Conv2d_10b2_7x1(x_2b)
        x_2b = self.Conv2d_10b3_1x7(x_2b)
        x_2b = self.Conv2d_10b4_3x3(x_2b)
        
        x = torch.cat([x_2a, x_2b], 1)
        
        x_3a = F.max_pool2d(x, kernel_size=2)
        x_3b = self.Conv2d_12a_3x3(x)
        
        x = torch.cat([x_3a, x_3b], 1)
        return x


class InceptionA(nn.Module):
    
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.Conv2d_a1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.Conv2d_a2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.Conv2d_a3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        
        self.Conv2d_b1 = BasicConv2d(in_channels, 96, kernel_size=3, padding=1)
        self.Conv2d_b2 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        
        self.Conv2d_c1 = BasicConv2d(in_channels, 96, kernel_size=1)
        
        self.Conv2d_d2 = BasicConv2d(in_channels, 96, kernel_size=1)
    
    def forward(self, x):
        x_a1 = self.Conv2d_a1(x)
        x_a2 = self.Conv2d_a2(x_a1)
        x_a = self.Conv2d_a3(x_a2)
        
        x_b1 = self.Conv2d_b1(x)
        x_b = self.Conv2d_b2(x_b1)
        
        x_c = self.Conv2d_c1(x)
        
        x_d1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_d = self.Conv2d_d2(x_d1)
        
        x = torch.cat([x_a, x_b, x_c, x_d], 1)
        return x
        

class InceptionB(nn.Module):
    
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.Conv2d_a1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.Conv2d_a2 = BasicConv2d(192, 192, kernel_size=(1,7), padding=(0,3))
        self.Conv2d_a3 = BasicConv2d(192, 224, kernel_size=(7,1), padding=(3,0))
        self.Conv2d_a4 = BasicConv2d(224, 224, kernel_size=(1,7), padding=(0,3))
        self.Conv2d_a5 = BasicConv2d(224, 256, kernel_size=(7,1), padding=(3,0))
        
        self.Conv2d_b1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.Conv2d_b2 = BasicConv2d(192, 224, kernel_size=(1,7), padding=(0,3))
        self.Conv2d_b3 = BasicConv2d(224, 256, kernel_size=(7,1), padding=(3,0))
        
        
        self.Conv2d_c1 = BasicConv2d(in_channels, 384, kernel_size=1)
        
        self.Conv2d_d2 = BasicConv2d(in_channels, 128, kernel_size=1)        
    
    def forward(self, x):
        x_a1 = self.Conv2d_a1(x)
        x_a2 = self.Conv2d_a2(x_a1)
        x_a3 = self.Conv2d_a3(x_a2)
        x_a4 = self.Conv2d_a4(x_a3)
        x_a = self.Conv2d_a5(x_a4)
        
        x_b1 = self.Conv2d_b1(x)
        x_b2 = self.Conv2d_b2(x_b1)
        x_b = self.Conv2d_b3(x_b2)
        
        x_c = self.Conv2d_c1(x)
        
        x_d1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_d = self.Conv2d_d2(x_d1)
        x = torch.cat([x_a, x_b, x_c, x_d], 1)
        return x
    
    
class InceptionC(nn.Module):
    
    def __init__(self, in_channels):
        super(InceptionC, self).__init__()
        self.Conv2d_a1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.Conv2d_a2 = BasicConv2d(384, 448, kernel_size=(1,3), padding=(0,1))
        self.Conv2d_a3 = BasicConv2d(448, 512, kernel_size=(3,1), padding=(1,0))
        self.Conv2d_a4_a = BasicConv2d(512, 256, kernel_size=(3,1), padding=(1,0))
        self.Conv2d_a4_b = BasicConv2d(512, 256, kernel_size=(1,3), padding=(0,1))
        
        self.Conv2d_b1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.Conv2d_b2_a = BasicConv2d(384, 256, kernel_size=(3,1), padding=(1,0))
        self.Conv2d_b2_b = BasicConv2d(384, 256, kernel_size=(1,3), padding=(0,1))
        
        self.Conv2d_c1 = BasicConv2d(in_channels, 256, kernel_size=1)
        
        self.Conv2d_d2 = BasicConv2d(in_channels, 256, kernel_size=1)
            
    def forward(self, x):
        x_a1 = self.Conv2d_a1(x)
        x_a2 = self.Conv2d_a2(x_a1)
        x_a3 = self.Conv2d_a3(x_a2)
        x_a_1 = self.Conv2d_a4_a(x_a3)
        x_a_2 = self.Conv2d_a4_b(x_a3)
        
        x_b1 = self.Conv2d_b1(x)
        x_b_1 = self.Conv2d_b2_a(x_b1)
        x_b_2 = self.Conv2d_b2_b(x_b1)
        
        x_c = self.Conv2d_c1(x)
        
        x_d1 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_d = self.Conv2d_d2(x_d1)
        
        x = torch.cat([x_a_1, x_a_2, x_b_1, x_b_2, x_c, x_d], 1)
        return x


class ReductionA(nn.Module):
    
    def __init__(self, in_channels):
        super(ReductionA, self).__init__()
        self.Conv2d_a1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.Conv2d_a2 = BasicConv2d(192, 224, kernel_size=3, padding=1)
        self.Conv2d_a3 = BasicConv2d(224, 256, kernel_size=3, stride=2, padding=1)
    
        self.Conv2d_b = BasicConv2d(in_channels, 384, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x_a1 = self.Conv2d_a1(x)
        x_a2 = self.Conv2d_a2(x_a1)
        x_a = self.Conv2d_a3(x_a2)
        
        x_b = self.Conv2d_b(x)
        
        x_c = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        
        x = torch.cat([x_a, x_b, x_c], 1)
        return x


class ReductionB(nn.Module):
    
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()
        self.Conv2d_a1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.Conv2d_a2 = BasicConv2d(256, 256, kernel_size=(1,7), padding=(0,3))
        self.Conv2d_a3 = BasicConv2d(256, 256, kernel_size=(7,1), padding=(3,0))
        self.Conv2d_a4 = BasicConv2d(256, 320, kernel_size=3, stride=2, padding=1)
    
        self.Conv2d_b1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.Conv2d_b2 = BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        x_a1 = self.Conv2d_a1(x)
        x_a2 = self.Conv2d_a2(x_a1)
        x_a3 = self.Conv2d_a3(x_a2)
        x_a = self.Conv2d_a4(x_a3)
        
        x_b1 = self.Conv2d_b1(x)
        x_b = self.Conv2d_b2(x_b1)
        
        x_c = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        x = torch.cat([x_a, x_b, x_c], 1)
        return x