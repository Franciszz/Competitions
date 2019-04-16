# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:19:00 2018

@author: Franc
"""
import os
os.chdir('C:/Users/Franc/Desktop/Dir/Competitions/#Kaggle/#Credit')
         
import sys
sys.path.append('code')

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

from feature_application import ApplicationFeatures
from main_sample import ApplResample

from model_lgb import LightGBM
from model_xgb import XGBoost
from model_cat import CatBoost

from model_bagging import ModelBagging


df_appl = pd.read_csv('data/applications_features.csv')
df_appl[ApplicationFeatures(np.nan).categorical_feature] = \
    df_appl[ApplicationFeatures(np.nan).categorical_feature].\
    apply(pd.Categorical,axis=0)
#df_appl_copy = copy.deepcopy(df_appl)
#list_corr = df_appl.corr()['TARGET']
#df_appl_adv = df_appl.loc[:,list_corr.index[list_corr.abs()>0.0005]]
params = dict(
lgb_params = {
        'params':{'boosting_type': 'dart', #rf, dart, goss
                  'objective': 'binary', #
                  'metric': 'auc',
                  'max_bin':500,
                  'num_leaves': 75,
                  'subsample': 0.75,
                  'learning_rate': 0.05,
                  'feature_fraction': 1,
#                  'bagging_fraction': 0.8,
#                  'bagging_freq': 5,
                  'max_depth': -1,
                  #'lambda_l1': 0.01,
                  #'lambda_l2': 0.1,
                  'verbose': 0},
        'seed': 909,
        'test_ratio': 0.25,
        'eval_ratio': 0.25,
        'num_boost_round': 5000,
        'early_stopping_rounds': 200,
        'nfold':5
},
xgb_params = {
        'params':{'booster': 'gbtree', #rf, dart, goss
                  'objective': 'binary:logistic', #
                  'eval_metric': 'auc',
#                  'learning_rate': 0.05,
                  'max_depth': 8,
                  'min_child_weight':1,
                  'subsample':1,
                  'colsample_bytree':0.9,
                  'lambda':0.1,
                  'learning_rate':0.05,
                  'scale_pos_weight':1
                  #'lambda_l1': 0.01,
                  #'lambda_l2': 0.1,
                  },               
        'seed': 909,
        'test_ratio': 0.25,
        'eval_ratio': 0.25,
        'num_boost_round': 2000,
        'early_stopping_rounds': 200,
        'nfold':5
},
cat_params = {
        'params':{'learning_rate': 0.05,
                  'max_depth': 8,
                  'l2_leaf_reg': 1,
                  'rsm': 0.9,
                  #'subsample': 0.9,
                  'class_weights': [1,1],
                  'loss_function': 'Logloss',
#                  'custom_loss': 'Logloss',
#                  'custom_metrics': 'AUC',
                  'eval_metric':'AUC'},
        'seed': 909,
        'test_ratio': 0.25,
        'eval_ratio': 0.25,
        'num_boost_round': 1000,
        'nfold': 5
}
)

appl_resample = ApplResample(df_appl, 1, 909)
df_model = appl_resample.data_split()

modelbag = ModelBagging(params)
modelbag.predict_proba(df_model)    
    

lgbm_clf = LightGBM(df_model['train']['sample_0'], params['lgb_params'])
#lgb_roc = lgboost_clf.cv()
lgb_clf = lgbm_clf.fit()
  
xgboost_clf = XGBoost(df_model['train']['sample_0'], params['xgb_params'])
#xgb_roc = xgboost_clf.cv()
xgb_clf = xgboost_clf.fit()

catboost_clf = CatBoost(df_model['train']['sample_0'], params['cat_params'])
#cat_roc = catboost_clf.cv()
cat_clf = catboost_clf.fit()

df_predict_lgb = lgb_clf.transform(df_model['eval'].iloc[:,2:])
df_predict_xgb = xgb_clf.transform(pd.get_dummies(df_model['eval'],\
            columns = ApplicationFeatures(np.nan).categorical_feature,
            dtype='int64').iloc[:,2:])
df_predict_cat = cat_clf.transform(df_model['eval'].iloc[:,2:])

roc_auc_score(df_model['eval']['TARGET'], df_predict_lgb['prediction'])
roc_auc_score(df_model['eval']['TARGET'], df_predict_xgb['prediction'])
roc_auc_score(df_model['eval']['TARGET'], df_predict_cat['prediction'])

roc_auc_score(df_model['eval']['TARGET'], 
              (1/(1+np.exp(-df_predict_cat['prediction']))+\
              1/(1+np.exp(-df_predict_xgb['prediction']))+\
              1/(1+np.exp(-df_predict_cat['prediction'])))/3)





