# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 14:56:16 2018

@author: Franc
"""

import os
os.chdir('C:/Users/Franc/Desktop/Dir/Competitions/#Kaggle/#Credit')
         
import sys
sys.path.append('code')

import pandas as pd
import numpy as np
from feature_application_update import ApplicationFeatures

# ======================== Application Data =================================
def appl_input():
    df_test = pd.read_csv('data/application_test.csv')
    df_test.insert(1,'TARGET',2)
    df = pd.read_csv('data/application_train.csv').append(df_test)
    return df
df_appll = appl_input()
AppClean = ApplicationFeatures(na_value = np.nan)
df_appll = AppClean.outcome(df_appll)

df_feature = df_appl.drop(list(df_appll.columns)[2:94],axis=1)
df_appll = df_appll.merge(df_feature, on=['SK_ID_CURR','TARGET'], how='left')
df_appll[ApplicationFeatures(np.nan).categorical_feature] = \
    df_appll[ApplicationFeatures(np.nan).categorical_feature].\
    apply(pd.Categorical,axis=0)

appll_resample = ApplResample(df_appll, 1, 909)
df_model = appl_resample.data_split()

    
lgbm_clf = LightGBM(df_model['train']['sample_0'], params['lgb_params'])
#lgb_roc = lgboost_clf.cv()
lgb_clf = lgbm_clf.fit()
