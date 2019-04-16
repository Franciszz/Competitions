# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 07:47:21 2018

@author: Franc
"""
import pandas as pd
import sys
sys.path.append('code')
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from data_input import googleDataInput
from pipeline_cache import from_pickle, to_pickle
#from models import CatBoost, LightGBM
# =========== Original Data input ================================
googleInputProcessor = googleDataInput()

train_df = googleInputProcessor.stupidFit(
        path = 'data/train.csv', train_mode = True)
test_df = googleInputProcessor.stupidFit(
        path = 'data/test.csv',train_mode=False)
to_pickle(train_df,'train_df.csv')
to_pickle(test_df,'test_df.csv')
train_df = from_pickle('train_df.csv')
# ========== Horizonal Features ==================================
from feature_engineer_horizonal import googleFeatureEngineerHorizonal
googleFeatureHorizonalProcessor = googleFeatureEngineerHorizonal()
train_horizonalize = googleFeatureHorizonalProcessor.transform(
        train_df, train_df, train_mode=True)
test_horizonalize = googleFeatureHorizonalProcessor.transform(
        test_df, train_df, train_mode=False)
#feature_horizonalize = {
#        'id_feature':googleFeatureHorizonalProcessor.id_feature,
#        'numerical_feature':googleFeatureHorizonalProcessor.numerical_feature,
#        'categorical_feature':googleFeatureHorizonalProcessor.categorical_feature,
#        'manual_categorical':googleFeatureHorizonalProcessor.manual_categorical,
#        'manual_numerical':googleFeatureHorizonalProcessor.manual_numerical}
#to_pickle(train_horizonalize,'train_horizonalize.csv')
#to_pickle(test_horizonalize,'test_horizonalize.csv')
#to_pickle(feature_horizonalize,'feature_horizonalize.csv')
#train_horizonalize = from_pickle('train_horizonalize.csv')
#test_horizonalize = from_pickle('test_horizonalize.csv')
#feature_horizonalize = from_pickle('feature_horizonalize.csv')
# =========== Vertical Features ==================================
from feature_engineer_vertical import googleFeatureEngineerVertical
googleFeatureVerticalProcessor = googleFeatureEngineerVertical()
train_verticalize = googleFeatureVerticalProcessor.transform(
        train_horizonalize)
test_verticalize = googleFeatureVerticalProcessor.transform(
        test_horizonalize)
#feature_verticalize = googleFeatureVerticalProcessor.manual_numerical
#to_pickle(train_verticalize,'train_verticalize.csv')
#to_pickle(test_verticalize,'test_verticalize.csv')
#to_pickle(feature_verticalize,'feature_verticalize.csv')
#train_verticalize = from_pickle('train_verticalize.csv')
#test_verticalize = from_pickle('test_verticalize.csv')

# =========== External Features ==================================
from feature_engineer_external import googleFeatureEngineerExternal
googleFeatureExternalProcessor = googleFeatureEngineerExternal()
train_exteralize = googleFeatureExternalProcessor.transform(
        train_verticalize, train_verticalize)
test_exteralize = googleFeatureExternalProcessor.transform(
        test_verticalize, train_verticalize)
#feature_exteralize = googleFeatureExternalProcessor.manual_numerical
to_pickle(train_exteralize,'train_exteralize.csv')
to_pickle(test_exteralize,'test_exteralize.csv')
#to_pickle(feature_exteralize,'feature_exteralize.csv')
train_exteralize = from_pickle('train_exteralize.csv')
test_exteralize = from_pickle('test_exteralize.csv')

# =========== Factorize Categorical Feature =======================
from data_encoder import googleLabelEncoder
googleLabelEncoderProcessor = googleLabelEncoder()
train_google, test_google = googleLabelEncoderProcessor.transform(
        train_exteralize, train_exteralize)
categorical_feature = googleLabelEncoderProcessor.feature_category
#to_pickle(train_google,'train_google.csv')
#to_pickle(test_google,'test_google.csv')
#train_google = from_pickle('train_google.csv')
#test_google = from_pickle('test_google.csv')

# =========== LGBM Fit ===========================================
#lgb_clf = LightGBM()
#lgb_clf.fit(train_last, clf=True)
#predict_clf = lgb_clf.transform(test_last)
#to_pickle(predict_clf,'predict_clf')
#predict_clf=from_pickle('predict_clf')
#train_clf = lgb_clf.transform(train_last)
#train_clf = from_pickle('train_clf')
#train_clfed = train_last[(train_clf>0.5)|(train_last['validRevenue']==1)]
#train_clfed = from_pickle('train_clfed')
#lgb_reg = LightGBM()
#lgb_reg.fit(train_clfed, clf=False)
#predict_reg = lgb_reg.transform(test_last)
#def mk_sub_simple(data):
#    data = pd.DataFrame({'fullVisitorId':test_last['fullVisitorId'],
#                            'PredictedLogRevenue':data})
#    data.loc[data['PredictedLogRevenue']<0,'PredictedLogRevenue']=0
#    data = pd.DataFrame(data.groupby('fullVisitorId')\
#                        ['PredictedLogRevenue'].sum().reset_index())
#    return data
#def mk_sub_compound(regs, clfs, threshold=0.1):
#    regs[clfs<threshold]=0
#    regs = pd.DataFrame({'fullVisitorId':test_last['fullVisitorId'],
#                            'PredictedLogRevenue':regs})
#    regs = pd.DataFrame(regs.groupby('fullVisitorId')\
#                        ['PredictedLogRevenue'].sum().reset_index())
#    return regs
#submission_0 = mk_sub_compound(predict_reg, predict_clf, 0.3)
#submission_0.to_csv('submission/submit_comfound_%d_lgb_%s_%d.csv'%
#                  (0.3,datetime.date(2018,9,28),1),index=False)
## =========== CatBoost Fit ===========================================
#cat_clf = CatBoost()
#cat_clf.fit(train_last, clf=False)





