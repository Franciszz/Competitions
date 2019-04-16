# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 19:01:17 2018

@author: Franc
"""

import os
os.chdir('C:/Users/Franc/Desktop/Dir/Competitions/# TCAL')
         
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image
from glob import glob

from steppy.base import BaseTransformer

      
class ImageRead(BaseTransformer):
    
    def __init__(self):
        super(ImageRead, self).__init__()
        self.meta_data = None
        self.image_data = None
    
    def generate_metadata(self):
        filename_list = sorted(os.listdir('data\\train'))
        filepath_list = glob('data\\train\\*')
        meta_train = pd.DataFrame({
                'type': [x[:2] for x in filename_list],
                'id': [x[2:-8] for x in filename_list],
                'path': filepath_list})
        meta_train['type'].replace({'擦花':'scratch','漏底':'leakage',
                  '凸粉':'powder','碰凹':'crash'}, inplace=True)
        meta_train['id'].replace({'擦花':'S','漏底':'L','凸粉':'P',
                  '碰凹':'C'},inplace=True)
        meta_train.loc[meta_train['id'].isin(['20180901141803','20180901141814',
                       '20180901141824']),'type'] = 'powder'
        
        meta_valid = meta_train.groupby(['type'], as_index=False).\
            apply(pd.DataFrame.sample, frac=0.2, replace=False).\
            reset_index(drop=True)
        meta_train = meta_train[~(meta_train['id'].isin(meta_valid['id']))].\
            reset_index(drop=True)

        meta_test = None
        if len(os.listdir('data\\test')):
            file_ids = sorted(os.listdir('data\\test'))
            meta_test = pd.DataFrame({
                    'type': None,
                    'id': file_ids,
                    'path': filepath_list})
        self.meta_data = {'train': meta_train,
                          'valid':meta_valid,
                          'test': meta_test}
        return self.meta_data

    def generate_imagedata(self, meta_data):
        img_data = {x: {'image': self.load_images(meta_data[x]['path']),
                        'id': meta_data[x]['id'],
                        'type': meta_data[x]['type'],
                        'label':pd.get_dummies(meta_data[x],columns=['type'],
                                               drop_first=False).values[:,2:].astype(np.float)} \
                        if meta_data[x] is not None else None \
                    for x in ['train', 'valid', 'test']}
        self.image_data = img_data
        return self.image_data
    
    def load_images(self, filepaths):
        images = []
        for filepath in tqdm(filepaths):
            image = Image.open(filepath, 'r')
            images.append(image)
        return images