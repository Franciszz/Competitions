# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 13:58:27 2018

@author: Franc
"""
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def loglikelood(preds, train_data):
        labels = train_data.get_label()
        preds = 1. / (1. + np.exp(-preds))
        grad = preds - labels
        hess = preds * (1. - preds)
        return grad, hess
    
def roc_auc_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', 1-roc_auc_score(labels, preds), False

def lgb_clf(df, df_eval, df_test, params):
        feature_name = list(df.columns[2:])
        
        categorical_feature = list(df.dtypes.index[df.dtypes=='object'])

        df_x, df_y = df.iloc[:,2:], df.TARGET
            
        x_train, x_eval, y_train, y_eval = \
                train_test_split(df_x, df_y,         
                                 test_size = params['test_ratio'])            
        lgb_train = \
                lgb.Dataset(x_train, y_train, 
                            feature_name = feature_name,
                            categorical_feature = categorical_feature,
                            free_raw_data = False)
        lgb_eval = \
                lgb.Dataset(x_eval, y_eval, 
                            feature_name = feature_name,
                            categorical_feature = categorical_feature,
                            free_raw_data = False)
        lgb_clf = \
                lgb.train(params = params['params'],
                          train_set = lgb_train,
                          num_boost_round = params['num_boost_round'],
                          init_model = None,
                          valid_sets = lgb_eval,
                          fobj = loglikelood,
                          feval = roc_auc_error,
                          early_stopping_rounds = params['early_stopping_rounds'],
                          learning_rates = \
                              lambda x: params['params']['learning_rate']*(0.99**x),
                          verbose_eval = True,
                          feature_name = feature_name,
                          categorical_feature = categorical_feature)
        
        y_eval_predict = lgb_clf.predict(df_eval.iloc[:,2:].values)
        lgb_clf_score = roc_auc_score(df_eval['TARGET'], y_eval_predict)
        
        y_test_predict = lgb_clf.predict(df_test.iloc[:,2:].values)
        return y_eval_predict, lgb_clf_score, y_test_predict

p1, s1, p2 = lgb_clf(df_model['train']['sample_0'], 
                     df_model['eval'], df_model['test'], 
                     params['lgb_params'])