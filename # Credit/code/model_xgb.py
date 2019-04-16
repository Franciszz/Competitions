# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 08:45:23 2018

@author: Franc
"""
import pandas as pd
import numpy as np
from steppy.base import BaseTransformer
from steppy.utils import get_logger
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from update_data_info import cat_feature
logger = get_logger()

class XGBoost(BaseTransformer):
    def __init__(self, xgb_params):
        self.df = None
        self.params = xgb_params['params']
        self.nfold = xgb_params['nfold']
        self.seed = xgb_params['seed']
        self.eval_ratio = xgb_params['eval_ratio']
        self.test_ratio = xgb_params['test_ratio']
        self.num_boost_round = xgb_params['num_boost_round']
        self.learning_rates = xgb_params['params']['learning_rate']
        self.early_stopping_rounds = xgb_params['early_stopping_rounds']
        self.categorical_feature = cat_feature
            
    def cat_transform(self, df):
        df = pd.get_dummies(df, columns = self.categorical_feature,
                            dtype='int64')
        return df
    
    def loglikelood(self, preds, train_data):
        labels = train_data.get_label()
        preds = 1. / (1. + np.exp(-preds))
        grad = preds - labels
        hess = preds * (1. - preds)
        return grad, hess
    
    def roc_auc_error(self, preds, train_data):
        labels = train_data.get_label()
        return 'error', 1 - roc_auc_score(labels, preds)
    
    def fit(self, df):
        df = pd.get_dummies(df, columns = self.categorical_feature,
                            dtype='int64')
        feature_name = list(df.columns[2:])
            
        df_x, df_y = \
                df.iloc[:,2:], df.TARGET
            
        x_train, x_test, y_train, y_test = \
                train_test_split(df_x, df_y,         
                                 test_size = self.test_ratio,
                                 random_state = self.seed)
            
#        x_train, x_eval, y_train, y_eval = \
#                train_test_split(x_train_eval, y_train_eval, 
#                                 test_size = self.eval_ratio,
#                                 random_state = self.seed)
            
        xgb_train = \
                xgb.DMatrix(x_train.values, y_train.values, 
                            feature_names = feature_name)
        xgb_eval = \
                xgb.DMatrix(x_test.values, y_test.values, 
                            feature_names = feature_name)
        
        self.estimator = \
                xgb.train(params = self.params,
                          dtrain = xgb_train,
                          num_boost_round = self.num_boost_round,
                          early_stopping_rounds = self.early_stopping_rounds,
                          evals = [(xgb_eval,'valid'),(xgb_train,'train')],
                          obj = self.loglikelood,
                          feval = self.roc_auc_error)
        return self
    def cv(self, df):
        df = self.cat_transform(df)
        feature_name = list(df.columns[2:])
            
        df_x, df_y = \
                df.iloc[:,2:], df.TARGET
        
        xgb_train = \
                xgb.DMatrix(df_x.values, df_y.values, 
                            feature_names = feature_name)
                
        xgb_cv_hist = \
                xgb.cv(params = self.params,
                       nfold = self.nfold,
                       dtrain = xgb_train,
                       xgb_model = None,
                       num_boost_round = self.num_boost_round,
                       obj = self.loglikelood,
                       feval = self.roc_auc_error,
                       verbose_eval = True)
        return xgb_cv_hist
    
    def transform(self, x_test):
        xgb_test = xgb.DMatrix(x_test)
        prediction = self.estimator.predict(xgb_test)
        prediction = 1/(1+np.exp(-prediction))
        return {'prediction':prediction}
    
    
    