# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:48:25 2018

@author: Franc
"""
import numpy as np
from steppy.base import BaseTransformer
from steppy.utils import get_logger

import catboost as cat

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logger = get_logger()

class CatBoost(BaseTransformer):
    def __init__(self, cat_params):
        self.df = None
        self.params = cat_params['params']
        self.nfold = cat_params['nfold']
        self.seed = cat_params['seed']
        self.eval_ratio = cat_params['eval_ratio']
        self.test_ratio = cat_params['test_ratio']
        self.num_boost_round = cat_params['num_boost_round']
        self.learning_rates = cat_params['params']['learning_rate']
        self.categorical_feature = None
    
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
        self.df = df
        feature_name = list(df.columns[2:])
        self.categorical_feature = \
                list(df.dtypes.reset_index().\
                         index[df.dtypes=='object']-2)
        df_x, df_y = df.iloc[:,2:], df.TARGET
        
        x_train, x_eval, y_train, y_eval = \
                train_test_split(df_x, df_y,         
                             test_size = self.test_ratio,
                             random_state = self.seed)
            
        cat_train = cat.Pool(
                data = x_train.values,
                label = y_train.values,
                cat_features = self.categorical_feature,
                feature_names = feature_name)
        cat_eval = cat.Pool(
                data = x_eval.values,
                label = y_eval.values,
                cat_features = self.categorical_feature,
                feature_names = feature_name)
        
        self.estimator = cat.train(
                params = self.params,
                pool = cat_train,
                num_boost_round = self.num_boost_round,
                eval_set = cat_eval
#                learning_rate = self.params['learning_rate'],
#                max_depth = self.params['max_depth'],
#                l2_leaf_reg = self.params['l2_leaf_reg'],
#                rsm = self.params['colsample_ratio'],
#                subsample = self.params['subsample_ratio'],
#                class_weights = self.params['class_weights'],
#                loss_function = self.loglikeloss,
#                custom_loss = self.loglikeloss,
#                custom_metric = self.roc_auc_error,
#                eval_metric = self.roc_auc_error
                )
        
#        self.fit(cat_train, eval_set = cat_eval, 
#                 cat_features = self.categorical_feature)
        return self
    
    def cv(self, df):
        feature_name = list(df.columns[2:])
        self.categorical_feature = \
                list(df.dtypes.reset_index().\
                         index[df.dtypes=='object']-2)
        df_x, df_y = \
                df.iloc[:,2:], df.TARGET
        
        cat_train = \
                cat.Pool(data = df_x.values,
                         label = df_y.values,
                         cat_features = self.categorical_feature,
                         feature_names = feature_name)
        
        cat_cv_hist = \
                cat.cv(pool = cat_train, 
                       params = self.params,
                       num_boost_round = self.num_boost_round,
                       nfold = self.nfold,
                       seed = self.seed)
        return cat_cv_hist
    def transform(self, x_test):
        feature_name = list(self.df.columns[2:])
        cat_test = \
                cat.Pool(data = x_test, 
                         cat_features = self.categorical_feature,
                         feature_names = feature_name)
        prediction = self.estimator.predict(cat_test)
        prediction = 1/(1+np.exp(-prediction))
        return {'prediction':prediction}
    
        
    
        