# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:01:31 2018

@author: Franc
"""

cat_feature = ['FLAG_OWN_REALTY',
 'ASSET_WALLSMATERIAL_MODE',
 'NAME_TYPE_SUITE',
 'NAME_CONTRACT_TYPE',
 'BASE_CODE_GENDER',
 #'FLAG_OWN_CAR',
 'NAME_INCOME_TYPE',
 'BASE_OCCUPATION_TYPE',
 #'ASSET_HOUSETYPE_MODE',
 'BASE_ORGANIZATION_TYPE',
 #'ASSET_EMERGENCYSTATE_MODE',
 'NAME_HOUSING_TYPE',
 'NAME_EDUCATION_TYPE',
 'NAME_FAMILY_STATUS',
 'NAME_WEEKDAY_APPR_PROCESS_START',
 'ASSET_FONDKAPREMONT_MODE']

params = dict(
lgb_params = {
        'params':{'boosting_type': 'dart', #rf, dart, goss
                  'objective': 'binary', #
                  'metric': 'auc',
                  #'max_bin':400,
                  #'num_leaves': 75,
                  #'subsample': 0.8,
                  'learning_rate': 0.012,
                  #'feature_fraction': 1,
#                  'bagging_fraction': 0.8,
#                  'bagging_freq': 5,
                  'max_depth': 8,
                  #'lambda_l1': 0.01,
                  #'lambda_l2': 0.1,
                  'verbose': 0},
        'seed': 909,
        'test_ratio': 0.25,
        'eval_ratio': 0.25,
        'num_boost_round': 5000,
        'early_stopping_rounds': 300,
        'nfold':5
},
lgb_params_gs = {
        'params':{'boosting_type': 'dart',
                  'num_leaves':75,
                  'max_depth':-1,
                  'n_estimators':5000,
                  'learning_rate':0.05,
                  'max_bin':500,
                  'objective':'binary',
#                  'min_child_weight':0.001,
#                  'min_child_sample':100,
                  'subsample':0.9,
                  'colsample_bytree':0.9,
                  #'reg_alpha':0.01
#                  'reg_lambda':0.01,
                  'random_state':909,
                  'n_jobs':-1,
                  'silent':False
                },
        'params_gs':{'boosting_type': ['dart','gbtree','goss'],
                  'num_leaves':[40,75,100],
                  'max_depth':[6,8,10],
                  'learning_rate':[0.001,0.025,0.05],
                  'max_bin':[250,500,1000]
#                  'objective':'binary',
#                  'min_child_weight':0.001,
#                  'min_child_sample':100,
#                  'subsample':0.9,
#                  'colsample_bytree':0.9,
                  #'reg_alpha':0.01
#                  'reg_lambda':0.01,
#                  'random_state':909,
#                  'n_jobs':-1,
#                  'silent':False
                },
        'params_fit':{'early_stopping_rounds':500,
                      'eval_metric':'auc'},
        'seed': 909,
        'test_ratio': 0.25,
        'eval_ratio': 0.25,
        'num_boost_round': 5000,
        'early_stopping_rounds': 500,
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
                  'scale_pos_weight':7
                  #'lambda_l1': 0.01,
                  #'lambda_l2': 0.1,
                  },               
        'seed': 909,
        'test_ratio': 0.25,
        'eval_ratio': 0.25,
        'num_boost_round': 5000,
        'early_stopping_rounds': 250,
        'nfold':5
},
cat_params = {
        'params':{'learning_rate': 0.015,#0.054 #0.025,0.025,0.01,0.018
                  'max_depth': 7,#7
                  #'l2_leaf_reg': 1,
                  'rsm': 1,
                  #'subsample': 0.8,
                  'class_weights': [1,11], #1.5
                  'loss_function': 'Logloss',
#                  'custom_loss': 'Logloss',
#                  'custom_metrics': 'AUC',
                  'eval_metric':'AUC'},
        'seed': 909,
        'test_ratio': 0.25,
        'eval_ratio': 0.25,
        'num_boost_round': 6000,
        'nfold': 5
}                  
)