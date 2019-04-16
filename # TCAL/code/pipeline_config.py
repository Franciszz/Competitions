# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 13:28:03 2018

@author: Franc
"""
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from imgaug import augmenters as iaa

data_dir = 'data'

labels = [0,1,2,3,4,5,6,7,8,9,10]
types = ['crash','leakage','powder','scratch']
counts = [34,106,46,61,43,41,30,125,55,68,51]
weights = torch.FloatTensor([3.50,1.18,2.75,2.00,3.00,3.00,4.00,1.00,2.25,1.75,2.50])


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
aug = iaa.Sequential([
        iaa.Scale({'height':2560,'width':2560}),
        iaa.Pad(px = 27, pad_mode = 'edge', keep_size = False),
        iaa.Dropout(p=(0,0.01))
        ])

image_tran = transforms.Compose([
        transforms.Resize((2560,2560)),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
        ])
mask_tran = transforms.Compose([
        #transforms.Resize(128),
        transforms.ToTensor()
        ])

def set_seed(seed=789):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
def to_array(images):
    images = [np.array(image) for image in images]
    if len(images) > 1:
        return images
    else:
        return images[0]
    
def to_tensor(images):
    loader = transforms.ToTensor()
    images = [loader(image) for image in images]
    if len(images) > 1:
        return images
    else:
        return images[0]
    
def tensor_to_pil(images):
    unloader = transforms.ToPILImage()
    images = [unloader(image) for image in images]
    if len(images) > 1:
        return images
    else:
        return images[0]
def array_to_pil(images):
    images = [Image.fromarray(image) for image in images]
    if len(images) > 1:
        return images
    else:
        return images[0]
