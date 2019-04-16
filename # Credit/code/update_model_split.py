# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 19:28:37 2018

@author: Franc
"""

import os
os.chdir('C:/Users/Franc/Desktop/Dir/Competitions/#Kaggle/#Credit')
         
import sys
sys.path.append('code')

import pandas as pd

from sklearn.metrics import roc_auc_score

from update_data_info import params
from update_main_sample import ApplResample

from model_lgb import LightGBM
#from model_xgb import XGBoost
from model_cat import CatBoost

df_all = pd.read_csv('data/df_feature_all.csv') 

result = pd.read_csv('data/feature_imp.csv')
col_target = list(result.name[(result.imp>5)|(result.cat_clf>0)])
df_all = df_all.loc[:,['SK_ID_CURR','TARGET']+col_target]

#df_all.query('TARGET ==1').DAYS_BIRTH.plot.hist(bins=1000)
#df_all.query('TARGET ==0').DAYS_BIRTH.plot.hist(bins=1000)
df_partition_1 = df_all[df_all.DAYS_BIRTH>-9600].reset_index(drop=True)
df_partition_2 = df_all[(df_all.DAYS_BIRTH<=-9600)&
                        (df_all.DAYS_BIRTH>=-17500)].reset_index(drop=True)
df_partition_3 = df_all[df_all.DAYS_BIRTH<-17500].reset_index(drop=True)
#df_partition_1.TARGET.value_counts()
#df_partition_2.TARGET.value_counts()
#df_partition_3.TARGET.value_counts()


lgbm_clf = LightGBM(params['lgb_params']) 
catboost_clf = CatBoost(params['cat_params'])

appl_sample = ApplResample(df_partition_3, 2, 909)
df_model = appl_sample.data_split()

for i in range(1):
    lgb_clf = lgbm_clf.fit(df_model['train']['sample_%d'%i])
    df_predict_lgb = lgb_clf.transform(df_model['eval'].iloc[:,2:])
    roc_auc_score(df_model['eval']['TARGET'], df_predict_lgb['prediction'])
    
    cat_clf = catboost_clf.fit(df_model['train']['sample_%d'%i])
    df_predict_cat = cat_clf.transform(df_model['eval'].iloc[:,2:])
    
    roc_auc_score(df_model['eval']['TARGET'], df_predict_cat['prediction'])
    pd.DataFrame({'SK_ID_CURR':df_model['eval'].SK_ID_CURR.astype('int'), 
                  'TARGET':list(df_predict_lgb['prediction']),
                  'TARGET_CAT':list(df_predict_cat['prediction'])}).\
        to_csv('prediction_update/prediction_3/sample_%d.csv'%i,index=False)
    df_test_lgb = lgb_clf.transform(df_model['test'].iloc[:,2:]) 
    df_test_cat = cat_clf.transform(df_model['test'].iloc[:,2:])
    pd.DataFrame({'SK_ID_CURR':df_model['test'].SK_ID_CURR.astype('int'),
                  'TARGET':list(df_test_lgb['prediction']),
                  'TARGET_CAT':list(df_test_cat['prediction'])}).\
        to_csv('prediction_update/prediction_3/test_%d.csv'%i,index=False)

