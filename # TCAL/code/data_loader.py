# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:11:40 2018

@author: Franc
"""

from torch.utils.data import Dataset
from pipeline_config import to_array, array_to_pil

class ImageBaseDataset(Dataset):
    def __init__(self, X, Y, image_augment, image_transform):
        #super(ImageBaseDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.image_augment = image_augment
        self.image_transform = image_transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        x = to_array([x])
        if self.image_augment(x) is not None:
            x = self.image_augment(x)
        x = array_to_pil([x])
        if self.image_transform is not None:
            x = self.image_transform(x)
        if self.Y is not None:
            y = self.Y[index]
            return x, y
        else:
            return x