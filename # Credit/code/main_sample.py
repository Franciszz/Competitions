# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:50:44 2018

@author: Franc
"""
import numpy as np

from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()

class ApplResample(BaseTransformer):
    
    def __init__(self, data, ratio, seed):
        self.df = data
        self.ratio = ratio
        self.seed = seed
        self.output = {'train':{},'test':None,'eval':None}
        
    def train_test_split(self):
        df_0 = self.df.query('TARGET == 0').reset_index(drop=True)
        df_1 = self.df.query('TARGET == 1').reset_index(drop=True)
        df_test = self.df.query('TARGET == 2').reset_index(drop=True)
        
        m, n = len(df_0), len(df_1)
        np.random.seed(self.seed)
        m_eval_index = np.random.permutation(m)
        np.random.seed(self.seed)
        n_eval_index = np.random.permutation(n)
        
        df_0_eval = df_0.iloc[m_eval_index[:1],:]
        df_defaultless = df_0.iloc[m_eval_index[1:],:]
        
        df_1_eval = df_1.iloc[n_eval_index[:1],:]
        df_default = df_1.iloc[n_eval_index[1:],:]
        
        df_eval = df_0_eval.append(df_1_eval).reset_index(drop=True)
        
        self.output['test'] = df_test
        self.output['eval'] = df_eval
        return {'defaultless': df_defaultless,
                'default': df_default}
    
    def data_split(self):
        df = self.train_test_split()
        df_default, df_defaultless = df['default'], df['defaultless']
        
        n_default = len(df_default)
        n_defaultless = self.ratio*n_default
        m = len(df_defaultless)
        
        np.random.seed(self.seed)
        m_shuffle = np.random.permutation(m)
        for i in range(int(m/n_defaultless)):
            self.output['train']['sample_%d'%i] = df_defaultless.\
                iloc[m_shuffle[int(i*n_defaultless):int((i+1)*n_defaultless)],:].\
                    append(df_default)
        return self.output
                    
            
            
            
        
        