# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 12:29:31 2018

@author: Franc
"""
import pandas as pd
import sys
sys.path.append('code')
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from data_input import GapDataInput
from pipeline_config import features
from pipeline_cache import VarRelation, VarPlot, to_pickle, from_pickle
from feature_engineer_horizonal import GapFeatureEngineerHorizonal
from feature_engineer_vertical import GapFeatureEngineerVertical
from feature_engineer_external import GapFeatureEngineerExternal
from models import CatBoost, LightGBM
# input
gap_input = GapDataInput()
train_df = gap_input.fitStupid()
test_df = gap_input.fitStupid(path = 'data/test.csv',train_mode=False)
#to_pickle(train_df,'train.csv')
#to_pickle(test_df,'test.csv')
train_df = from_pickle('train.csv')
test_df = from_pickle('test.csv')

# horizonal feature
gap_feh = GapFeatureEngineerHorizonal()
train_fe = gap_feh.transform(train_df, train_df)
test_fe = gap_feh.transform(test_df, train_df, train_mode=False)
to_pickle(train_fe,'train_fe.csv')
to_pickle(test_fe,'test_fe.csv')
train_fe = from_pickle('train_fe.csv')
test_fe = from_pickle('test_fe.csv')

# vertical feature
gap_fev = GapFeatureEngineerVertical()
train_all = gap_fev.transform(train_fe)
test_all = gap_fev.transform(test_fe)
train_all['visitMonth'] = train_all['visitId'].apply(lambda x: x.month)
test_all['visitMonth'] = test_all['visitId'].apply(lambda x: x.month)

to_pickle(train_all,'train_all.csv')
to_pickle(test_all,'test_all.csv')
train_all = from_pickle('train_all.csv')
test_all = from_pickle('test_all.csv')

# external feature
gap_ext = GapFeatureEngineerExternal()
train_full = gap_ext.transform(train_all, train_all)
test_full = gap_ext.transform(test_all, train_all)
train_full = train_full.reindex(
        columns=features['target_feature']+features['id_feature']+\
            sorted(features['categorical_feature']+features['numerical_feature']))
test_full = test_full.reindex(
        columns=features['target_feature']+features['id_feature']+\
            sorted(features['categorical_feature']+features['numerical_feature'])) 
to_pickle(train_full,'train_full.csv')
to_pickle(test_full,'test_full.csv')





# model
lgb_clf = LightGBM()
lgb_clf.fit(train_full,clf=True)

cat_clf = CatBoost()
lgb_clf.fit(train_full,clf=True)







train_nnd = pd.merge(train_full.nunique().reset_index(),
                     train_full.isna().mean(axis=0).reset_index(),
                     on=['index']).\
                     merge(train_full.dtypes.reset_index()).\
                     rename(columns = {'index':'colname','0_x':'nunique',
                                       '0_y':'na_ratio',0:'dtype'})

# ====== duplicates ======
train_duplicates_df = train_fe[train_fe[[
        'fullVisitorId','visitId','sessionId']].duplicated(keep=False)]

train_duplicates = train_fe[train_fe[[
        'fullVisitorId','visitId','sessionId']].duplicated(keep=False)]

judge_id_multi = train_fe['fullVisitorId'].value_counts().sort_values(ascending=False)
id_multi_category = judge_id_multi.index[judge_id_multi>1].tolist()
train_multivisit = train_fe[train_fe['fullVisitorId'].isin(id_multi_category)]
# ============================================================================
# ========== Univariate Exploration ==========================================
# ============================================================================

RelationFeature = VarRelation(train_fe,'geoNetwork_city')
VarPlot(train_df,'totals_hits')

# channelGrouping Organic Search:Low, Referral:High

# DeviceBrowser Chrome:High

# DeviceCategory desktop:High

# OperatingSystem Macintosh&Chrome OS:High, Windows:Low

# geoNetwork city    not_available,not set : Low

# geoNetwork continent Americas : High

# geoNetwork country Americas&Canada : High


