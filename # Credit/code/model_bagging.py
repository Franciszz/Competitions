# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:37:55 2018

@author: Franc
"""
import pandas as pd
import numpy as np

from steppy.base import BaseTransformer
from steppy.utils import get_logger

#from sklearn.metrics import roc_auc_score

from feature_application import ApplicationFeatures
from model_lgb import LightGBM
from model_xgb import XGBoost
from model_cat import CatBoost

logger = get_logger()

class ModelBagging(BaseTransformer):
    def __init__(self, params, **kwargs):
        self.lgb_params = params['lgb_params']
        self.xgb_params = params['xgb_params']
        self.cat_params = params['cat_params']
        self.eval_result = {'lgb':{},'xgb':{},'cat':{}}
        self.eval_result_transform = {'lgb':{},'xgb':{},'cat':{}}
        self.eval_auc = {'lgb':{},'xgb':{},'cat':{}}
    
    def predict_proba(self, df):
        for i in range(len(df['train'])):
            self.eval_result['lgb']['sample_%d'%i] = \
                LightGBM(df['train']['sample_%d'%i],
                         self.lgb_params).fit().\
                             transform(df['eval'].iloc[:,2:])
#            self.eval_auc['lgb']['sample_%d'%i] = \
#                roc_auc_score(df['eval']['TARGET'], 
#                              self.eval_result['lgb']['sample_%d'%i])  
#                          
            self.eval_result['xgb']['sample_%d'%i] = \
                XGBoost(df['train']['sample_%d'%i],
                        self.xgb_params).fit().\
                            transform(pd.get_dummies(df['eval'],\
                                columns = ApplicationFeatures(np.nan).categorical_feature,
                                dtype='int64').iloc[:,2:])
#            self.eval_auc['xgb']['sample_%d'%i] = \
#                roc_auc_score(df['eval']['TARGET'],
#                              self.eval_result['xgb']['sample_%d'%i]) 
#                
            self.eval_result['cat']['sample_%d'%i] = \
                CatBoost(df['train']['sample_%d'%i],
                         self.cat_params).fit().\
                             transform(df['eval'].iloc[:,2:])
#            self.eval_auc['cat']['sample_%d'%i] = \
#                roc_auc_score(df['eval']['TARGET'],
#                              self.eval_result['cat']['sample_%d'%i]) 
        return self
    
    def predict_transform(self):
        for method, value in self.eval_result.items():
            for sample, result in value.items():
                self.eval_result[method][sample] = \
                    1/(1+np.exp(-result))
        return self
            
        
        
    
    