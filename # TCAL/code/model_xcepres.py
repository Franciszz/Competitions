# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 23:09:41 2018

@author: Franc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo

class XceptionRes(nn.Module):
    
    def __init__(self, num_classes = 4):
        super(XceptionRes, self).__init__()
        self.Conv2d_pre = PreStem(False)
        self.Conv2d_entry = EntryStem(32)
        self.Conv2d_middle1 = MiddleStem(728)
        self.Conv2d_middle2 = MiddleStem(728)
        self.Conv2d_middle3 = MiddleStem(728)
        self.Conv2d_middle4 = MiddleStem(728)
        self.Conv2d_middle5 = MiddleStem(728)
        self.Conv2d_middle6 = MiddleStem(728)
        self.Conv2d_middle7 = MiddleStem(728)
        self.Conv2d_middle8 = MiddleStem(728)
        self.Conv2d_exit = ExitStem(728)
        self.fc = nn.Linear(2048, num_classes)
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
        x = self.Conv2d_pre(x)
        x = self.Conv2d_entry(x)
        x = self.Conv2d_middle1(x)
        x = self.Conv2d_middle2(x)
        x = self.Conv2d_middle3(x)
        x = self.Conv2d_middle4(x)
        x = self.Conv2d_middle5(x)
        x = self.Conv2d_middle6(x)
        x = self.Conv2d_middle7(x)
        x = self.Conv2d_middle8(x)
        x = self.Conv2d_exit(x)
        x = F.avg_pool2d(x, kernel_size=9)
        x = F.dropout(x, p=0.2)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    
#
#class EntryStem(nn.Module):
#    
#    def __init__(self):
#        super(EntryStem, self).__init__()
#        self.Conv2d_1_7x7 = BasicConv2d(3, 16, kernel_size=7, stride=3, padding=3)
#        self.Conv2d_2_7x7 = BasicConv2d(16, 16, kernel_size=7, padding=3)
#        self.Conv2d_3_7x7 = BasicConv2d(16, 32, kernel_size=7, stride=3, padding=3)
#        self.Conv2d_4_7x7 = BasicConv2d(32, 32, kernel_size=7, padding=3)
#        
#        self.Conv2d_5_3x3 = BasicConv2d(32, 32, kernel_size=3, stride=2)
#        self.Conv2d_6_3x3 = BasicConv2d(32, 64, kernel_size=3)
#        
#        self.Conv2d_7a_1x1 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
#        self.Conv2d_7b1_3x3 = SeperableConv2d(64, 128, activation=False)
#        self.Conv2d_7b2_3x3 = SeperableConv2d(128, 128, activation=True)
#        
#        self.Conv2d_8a_1x1 = nn.Conv2d(256, 256, kernel_size=1, stride=2)
#        self.Conv2d_8b1_3x3 = SeperableConv2d(256, 256, activation=True)
#        self.Conv2d_8b2_3x3 = SeperableConv2d(256, 256, activation=True)
#        
#        self.Conv2d_9a_1x1 = nn.Conv2d(512, 728, kernel_size=1, stride=2)
#        self.Conv2d_9b1_3x3 = SeperableConv2d(512, 728, activation=False)
#        self.Conv2d_9b2_3x3 = SeperableConv2d(728, 728, activation=True)
#        
#        self.Conv2d_10_3x3 = BasicConv2d(1456,728, kernel_size=3, padding=1)        
#    def forward(self, x):
#        x = self.Conv2d_1_7x7(x)
#        x = self.Conv2d_3_7x7(x)
#        x = self.Conv2d_4_7x7(x)
#        x = self.Conv2d_4_7x7(x)
#        
#        x = self.Conv2d_5_3x3(x)
#        x = self.Conv2d_6_3x3(x)
#        
#        residual_1 = self.Conv2d_7a_1x1(x)
#        residual_1 = nn.BatchNorm2d(128, eps=0.001)(residual_1)
#        
#        x = self.Conv2d_7b1_3x3(x)
#        x = self.Conv2d_7b2_3x3(x)
#        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
#        
#        x = torch.cat([x, residual_1], 1)
#        
#        residual_2 = self.Conv2d_8a_1x1(x)
#        residual_2 = nn.BatchNorm2d(256, eps=0.001)(residual_2)
#        
#        x = self.Conv2d_8b1_3x3(x)
#        x = self.Conv2d_8b2_3x3(x)
#        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
#        
#        x = torch.cat([x, residual_2], 1)
#        
#        residual_3 = self.Conv2d_9a_1x1(x)
#        residual_3 = nn.BatchNorm2d(728, eps=0.001)(residual_3)
#        
#        x = self.Conv2d_9b1_3x3(x)
#        x = self.Conv2d_9b2_3x3(x)
#        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
#        
#        x = torch.cat([x, residual_3],1)
#        x = self.Conv2d_10_3x3(x)        
#        return x
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
        x = self.Conv2d_3_7x7(x)
        x = self.Conv2d_4_7x7(x)
        x = self.Conv2d_4_7x7(x)
        return x

class EntryStem(nn.Module):
    
    def __init__(self, in_channels):
        super(EntryStem, self).__init__()
        
        self.Conv2d_5_3x3 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2)
        self.Conv2d_6_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        
        self.Conv2d_7_res = ResidualNet(64,128, activation=False)
        self.Conv2d_8_res = ResidualNet(256,256, activation=True)
        self.Conv2d_9_res = ResidualNet(512,728, activation=True)
        
        self.Conv2d_10_3x3 = BasicConv2d(1456,728, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.Conv2d_5_3x3(x)
        x = self.Conv2d_6_3x3(x)
        
        x = self.Conv2d_7_res(x)
        x = self.Conv2d_8_res(x)
        x = self.Conv2d_9_res(x)
        
        x = self.Conv2d_10_3x3(x)
        return x
        
class MiddleStem(nn.Module):
    
    def __init__(self, in_channels):
        super(MiddleStem, self).__init__()
        self.Conv2d_1 = SeperableConv2d(in_channels, in_channels, activation=True)
        self.Conv2d_2 = SeperableConv2d(in_channels, in_channels, activation=True)
        self.Conv2d_3 = SeperableConv2d(in_channels, in_channels, activation=True)
        self.Conv2d_4 = BasicConv2d(1456,728, kernel_size=3, padding=1)
        
    def forward(self, x):
        residual = x
        x = self.Conv2d_1(x)
        x = self.Conv2d_2(x)
        x = self.Conv2d_3(x)
        x = torch.cat([x, residual], 1)
        x = self.Conv2d_4(x)
        return x
   
    
class ExitStem(nn.Module):
    
    def __init__(self, in_channels):
        super(ExitStem, self).__init__()
        self.Conv2d_1 = SeperableConv2d(in_channels, 728, activation=True)
        self.Conv2d_2 = SeperableConv2d(728, 1024, activation=True)
        
        self.Conv2d_3a = nn.Conv2d(in_channels, 512, kernel_size=1, stride=2)
        self.Conv2d_3b1 = SeperableConv2d(in_channels, in_channels, activation=True)
        self.Conv2d_3b2 = SeperableConv2d(in_channels, in_channels, activation=True)
        
        self.Conv2d_4 = SeperableConv2d(1536, 1536, activation=False)
        self.Conv2d_5 = SeperableConv2d(1536, 2048, activation=True)
        
    def forward(self, x):
        residual = self.Conv2d_3a(x)
        residual = nn.BatchNorm2d(512, eps=0.001)(residual)
        
        x = self.Conv2d_1(x)
        x = self.Conv2d_2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        x = torch.cat([x, residual], 1)
        
        x = self.Conv2d_4(x)
        x = self.Conv2d_5(x)
        
        return F.relu(x, inplace=True)
        
    
    
class BasicConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, **kwards):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwards)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
    
class SeperableConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, activation=True, **kwards):
        super(SeperableConv2d, self).__init__()
        self.activation = True
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 
                                  kernel_size = 1)
        self.conv_d = nn.Conv2d(out_channels, out_channels, 
                                kernel_size = (1,3), padding = (0,1),
                                bias=False, **kwards)
        self.conv_w = nn.Conv2d(out_channels, out_channels, 
                                kernel_size = (3,1), padding = (1,0),
                                bias=False, **kwards)
        self.conv_3x3 = nn.Conv2d(out_channels, out_channels, 
                                  kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x):
        if self.activation:
            x = F.relu(x)
        x = self.conv_1x1(x)
        x = self.conv_d(x)
        x = self.conv_w(x)
        x = self.conv_3x3(x)
        x = self.bn(x)
        return x    
    
    
class ResidualNet(nn.Module):
    
    def __init__(self, in_channels, out_channels, activation, **kwards):
        self.activation=True
        super(ResidualNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv2d_a = nn.Conv2d(in_channels, out_channels, 
                                  kernel_size=1, stride=2)
        self.Conv2d_b1_3x3 = SeperableConv2d(in_channels, out_channels, 
                                             activation=activation)
        self.Conv2d_b2_3x3 = SeperableConv2d(out_channels, out_channels, 
                                             activation=True)
    
    def forward(self, x):
        residual = self.Conv2d_a(x)
        residual = nn.BatchNorm2d(self.out_channels, eps=0.001)(residual)
        
        x = self.Conv2d_b1_3x3(x)
        x = self.Conv2d_b2_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        x = torch.cat([x, residual], 1)
        return x