# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 18:22:59 2018

@author: Franc
"""

import os
os.chdir('C:/Users/Franc/Desktop/Dir/Competitions/#Kaggle/#Credit')
         
import sys
sys.path.append('code')

import pandas as pd

from sklearn.metrics import roc_auc_score

from update_data_info import params
from main_sample import ApplResample

from model_cat import CatBoost
from model_lgb import LightGBM
  

#result = pd.read_csv('data/feature_imp.csv')
#feature_imp = pd.read_csv('prediction5/feature_importance_2.csv')
#col_rm = list(result.name[(result.imp>5)|(result.cat_clf>0)|(feature_imp.imp>0)])

df_all = pd.read_csv('data/df_feature_all.csv')
#df_all = df_all.loc[:,['SK_ID_CURR','TARGET']+col_rm]

appl_resample = ApplResample(df_all, 11, 909)
df_model = appl_resample.data_split()
      
catboost_clf = CatBoost(params['cat_params'])

cat_clf = catboost_clf.fit(df_model['train']['sample_0'])
df_predict_cat = cat_clf.transform(df_model['eval'].iloc[:,2:])

roc_auc_score(df_model['eval']['TARGET'], df_predict_cat['prediction']) 
pd.DataFrame({'SK_ID_CURR':df_model['eval'].SK_ID_CURR.astype('int'), 
              'TARGET':list(df_predict_cat['prediction'])}).\
        to_csv('prediction10/sample_cat_4_4.csv',index=False)
              #'TARGET_CAT':list(df_predict_cat['prediction'])}).\
    #to_csv('prediction4/ssample_%d.csv'%i,index=False)
#    df_test_lgb = lgb_clf.transform(df_model['test'].iloc[:,2:]) 
df_test_cat = cat_clf.transform(df_model['test'].iloc[:,2:])
pd.DataFrame({'SK_ID_CURR':df_model['test'].SK_ID_CURR.astype('int'),
              'TARGET':list(df_test_cat['prediction'])}).\
        to_csv('prediction10/test_cat_11.csv',index=False)
        
lgbm_clf = LightGBM(params['lgb_params']) 
lgb_clf = lgbm_clf.fit(df_model['train']['sample_0'])
df_predict_lgb = lgb_clf.transform(df_model['eval'].iloc[:,2:])
roc_auc_score(df_model['eval']['TARGET'], df_predict_lgb['prediction'])
pd.DataFrame({'SK_ID_CURR':df_model['eval'].SK_ID_CURR.astype('int'),
              'TARGET':list(df_predict_lgb['prediction'])}).\
    to_csv('prediction10/sample_lgb_0.csv',index=False)
df_test = lgb_clf.transform(df_model['test'].iloc[:,2:])    
pd.DataFrame({'SK_ID_CURR':df_model['eval'].SK_ID_CURR.astype('int'),
              'TARGET':list(df_test['prediction'])}).\
    to_csv('prediction6/test_lgb_0.csv',index=False)        