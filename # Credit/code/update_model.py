# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 17:08:31 2018

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

from model_lgb import LightGBM
from model_xgb import XGBoost
from model_cat import CatBoost


df_application = pd.read_csv('data/feature_application_update.csv')
df_previous = pd.read_csv('data/feature_previous_update.csv')
df_bureau = pd.read_csv('data/feature_bureau_update.csv')
df_credit = pd.read_csv('data/feature_credit_update.csv')
df_pos = pd.read_csv('data/feature_pos_update.csv')
df_instal = pd.read_csv('data/feature_instal_update.csv')
# ====== feature union ======
## 0.763 
#appl_resample = ApplResample(df_application, 1, 909)
## 0.767
#df_appl_prev = df_application.merge(df_previous, on = ['SK_ID_CURR'], how = 'left')
## 0.775
#df_appl_bure = df_application.merge(df_bureau, on = ['SK_ID_CURR'], how = 'left')
## 
#df_appl_cred = df_application.merge(df_credit, on = ['SK_ID_CURR'], how='left')
## 
#df_appl_pos = df_application.merge(df_pos, on = ['SK_ID_CURR'], how='left')
## 
#df_appl_instl = df_application.merge(df_instal, on = ['SK_ID_CURR'], how='left')
## 0.774
#df_appl_bure_prev = df_appl_bure.merge(df_previous, on = ['SK_ID_CURR'], how = 'left')
# 0.772
df_all = df_application.merge(df_previous, on = ['SK_ID_CURR'], how = 'left').\
        merge(df_bureau, on = ['SK_ID_CURR'], how = 'left').\
        merge(df_pos, on = ['SK_ID_CURR'], how = 'left').\
        merge(df_instal, on = ['SK_ID_CURR'], how = 'left').\
        merge(df_credit, on=['SK_ID_CURR'], how = 'left')        
# df_all.to_csv('data/df_feature_all.csv',index=False)
df_all = pd.read_csv('data/df_feature_all.csv')        
lgbm_clf = LightGBM(params['lgb_params']) 
catboost_clf = CatBoost(params['cat_params'])

#result = pd.DataFrame(dict(imp = lgbm_clf.estimator.feature_importance(),
#                           name = lgbm_clf.estimator.feature_name()))
#result.to_csv('data/feature_imp.csv',index=False)
result = pd.read_csv('data/feature_imp.csv')
col_rm = list(result.name[(result.imp>5)|(result.cat_clf>0)])
df_all = df_all.loc[:,['SK_ID_CURR','TARGET']+col_rm]
# ====== 数据分割 ======
appl_resample = ApplResample(df_all, 2.2, 909)
df_model = appl_resample.data_split()

# lgbm
#0.7788, 0.7780, 0.7828
#0.7856, 0.7850, 0.7872, 0.7849, 0.7854, 0.7864, 0.7847, 0.7838, 0.7863, 0.7856, 0.7856
for i in range(9):
    lgb_clf = lgbm_clf.fit(df_model['train']['sample_%d'%i])
    df_predict_lgb = lgb_clf.transform(df_model['eval'].iloc[:,2:])
    roc_auc_score(df_model['eval']['TARGET'], df_predict_lgb['prediction'])
    pd.DataFrame({'SK_ID_CURR':df_model['eval'].SK_ID_CURR.astype('int'),
                  'TARGET':list(df_predict_lgb['prediction'])}).\
        to_csv('prediction6/sample_1.2_%d.csv'%i,index=False)
    df_test = lgb_clf.transform(df_model['test'].iloc[:,2:])    
    pd.DataFrame({'SK_ID_CURR':df_model['eval'].SK_ID_CURR.astype('int'),
                  'TARGET':list(df_test['prediction'])}).\
        to_csv('prediction6/test_1.2_%d.csv'%i,index=False)
##    
cat_clf = catboost_clf.fit(df_model['train']['sample_0'])
df_predict_cat = cat_clf.transform(df_model['eval'].iloc[:,2:])

roc_auc_score(df_model['eval']['TARGET'], df_predict_cat['prediction']) 
pd.DataFrame({'SK_ID_CURR':df_model['eval'].SK_ID_CURR.astype('int'), 
              'TARGET':list(df_predict_cat['prediction'])}).\
        to_csv('prediction6/sample_cat_10.csv',index=False)
              #'TARGET_CAT':list(df_predict_cat['prediction'])}).\
    #to_csv('prediction4/ssample_%d.csv'%i,index=False)
#    df_test_lgb = lgb_clf.transform(df_model['test'].iloc[:,2:]) 
df_test_cat = cat_clf.transform(df_model['test'].iloc[:,2:])
pd.DataFrame({'SK_ID_CURR':df_model['test'].SK_ID_CURR.astype('int'),
              'TARGET':list(df_test_cat['prediction'])}).\
        to_csv('prediction6/test_cat_10.csv',index=False)
              #'TARGET_CAT':list(df_test_cat['prediction'])}).\
   #     to_csv('prediction4/ttest_%d.csv'%i,index=False)
feature_imp = pd.DataFrame(dict(imp=cat_clf.estimator.feature_importances_,
                                name = list(df_all.columns)[2:]))
feature_imp.to_csv('prediction5/feature_importance_2.csv',index=False)
# catboost
catboost_clf = CatBoost(params['cat_params'])
#cat_roc = catboost_clf.cv()
for i in range(11):
    cat_clf = catboost_clf.fit(df_model['train']['sample_%d'%i])
    df_predict_cat = cat_clf.transform(df_model['eval'].iloc[:,2:])
    roc_auc_score(df_model['eval']['TARGET'], df_predict_cat['prediction'])
    pd.DataFrame({'SK_ID_CURR':df_model['eval'].SK_ID_CURR.astype('int'),
                  'TARGET':list(df_predict_cat['prediction'])}).\
        to_csv('prediction3/cat_sample_%d.csv'%i,index=False)
    df_test = cat_clf.transform(df_model['test'].iloc[:,2:])    
    pd.DataFrame({'SK_ID_CURR':df_model['eval'].SK_ID_CURR.astype('int'),
                  'TARGET':list(df_test['prediction'])}).\
        to_csv('prediction3/cat_test_%d.csv'%i,index=False)
        
#summit.to_csv('prediction/submit_0.csv', index=False)
#summit.SK_ID_CURR = summit.SK_ID_CURR.astype('int')
# xgboost
xgboost_clf = XGBoost(params['xgb_params'])
##xgb_roc = xgboost_clf.cv()
xgb_clf = xgboost_clf.fit(df_model['train']['sample_0'])
#df_predict_xgb = xgb_clf.transform(pd.get_dummies(df_model['eval'],\
#            columns = ApplicationFeatures(np.nan).categorical_feature,
#            dtype='int64').iloc[:,2:])
#roc_auc_score(df_model['eval']['TARGET'], df_predict_xgb['prediction'])
df_all = df_all.drop(list(feature_imp.name[feature_imp.imp==0]),axis=1)
cat_clf = catboost_clf.fit(df_model['train']['sample_0'])
df_predict_cat = cat_clf.transform(df_model['eval'].iloc[:,2:])
