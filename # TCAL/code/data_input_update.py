# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 19:43:34 2018

@author: Franc
"""

import os
os.chdir('C:/Users/Franc/Desktop/Dir/Competitions/# TCAL')

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder      
         
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image
from glob import glob


from pipeline_config import data_dir
#from steppy.base import BaseTransformer

      
class ImageRead(object):
    
    def __init__(self):
        super(ImageRead, self).__init__()
        
        self.meta_data = None
        self.image_data = None
    
    def generate_metadata(self):
        
        ids = range(len(ImageFolder(data_dir + '\\train')))
        type_list = os.listdir(data_dir + '\\train')
        path_list, label_list = [], []
        for i, type in enumerate(type_list):
            filepaths = glob(data_dir + f'\\train\\{type}\\*.jpg') \
                if type != '其他' else \
                glob(data_dir + f'\\train\\{type}\\*\\*.jpg')
            labels = [i]*len(filepaths)
            path_list += filepaths
            label_list += labels
        meta_train = pd.DataFrame({'id': ids,
                                   'label': label_list,
                                   'path': path_list})
#        meta_valid = meta_train.groupby(['label'], as_index=False).\
#            apply(pd.DataFrame.sample, frac=136/712, replace=False).\
#            reset_index(drop=True)
#        meta_train = meta_train[~(meta_train['id'].isin(meta_valid['id']))].\
#            reset_index(drop=True)
        meta_test = self.meta_test_generate()
        return {'train': meta_train,
                #'valid': meta_valid,
                'test': meta_test}
    
    def meta_test_generate(self):
        category_data = pd.read_csv('data\\test.csv').sort_values(['image_id'])
        ids = category_data['image_id'].apply(lambda x: x.split('.')[0]).tolist()
        path_list = glob(data_dir+'\\test\\*.jpg')
        label_list = category_data['image_category'].tolist()
        meta_test = pd.DataFrame({'id': ids,
                                  'label': label_list,
                                  'path': path_list}).\
            sort_values(['id']).reset_index(drop=True)
        return meta_test
    
    def generate_imagedata(self, meta_data):
        img_data = {x: {'image': self.load_images(meta_data[x]['path']),
                        'id': meta_data[x]['id'],
                        'label':meta_data[x]['label']} \
                    for x in ['train', 'test']}
        self.image_data = img_data
        return self.image_data
    
    def load_images(self, filepaths):
        images = []
        for filepath in tqdm(filepaths):
            image = Image.open(filepath, 'r')
            images.append(image)
        return images
        
        
        