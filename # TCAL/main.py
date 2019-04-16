# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 19:35:05 2018

@author: Franc
"""

import sys
sys.path.append('code')

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from cv2 import imread
from imgaug import augmenters as iaa

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

from data_input_update import ImageRead
from data_loader import ImageBaseDataset

from pipeline_config import image_tran
from pipeline_train import NetClassifier, train_model, eval_model

from model_inception import InceptionNet
from model_xception import XceptionNet
from model_xcepres import XceptionRes
from model_resnet import ResNet
# ====== data_input
image_gen = ImageRead()
meta_data = image_gen.generate_metadata()
image_data = image_gen.generate_imagedata(meta_data)
# ====== train_valid_split


# ====== data_load
image_dataset = {
        x:ImageBaseDataset(image_data[x]['image'],
                           image_data[x]['label'],
                           image_augment = iaa.Noop().augment_image,
                           image_transform = image_tran) 
        for x in ['train','test']}#['train','valid','test']}

image_loader = {
        'train': DataLoader(image_dataset['train'], batch_size=15, 
                            shuffle=True, num_workers=4),
        # 'valid': DataLoader(image_dataset['valid'], batch_size=17, 
        #                     shuffle=True, num_workers=4), 
        'test': DataLoader(image_dataset['test'], batch_size=6, 
                           shuffle=False, num_workers=4),                                     
        }
# ====== model setup
inceptionNet = InceptionNet()
#xceptionNet = XceptionNet()
#xceptionRes = XceptionRes()
resNet = ResNet()
img, label = next(iter(image_loader['train']))
#img_s = resNet(img)
#plt.imshow(img[0].transpose(0,2).transpose(0,1))
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.01)
#data_loader = image_loader['train']
#num_epochs = 10
#model = train_model(model, criterion, optimizer, data_loader, num_epochs)

netclassifier = NetClassifier()
model = netclassifier.train_model(resNet, image_loader)


