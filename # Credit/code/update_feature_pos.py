# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 15:17:14 2018

@author: Franc
"""

import numpy as np

from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()

class PosCashFeature(BaseTransformer):
    """cat_feature, num_feature待补充"""   
    def __init__(self, **kwargs):
        self.data = None
        self.cat_feature = []
        self.num_feature = []
        self.na_value = np.nan
        self.na_fill = 0
    
    def transform(self, df):
        df = df.sort_values(['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'])
        # cat_feature
        df['CONTRACT_STATUS_IS_REFUSED'] = \
                (df['NAME_CONTRACT_STATUS'] == 'Refused').astype('int64')
        df['CONTRACT_STATUS_IS_CANCEL'] = \
                (df['NAME_CONTRACT_STATUS'] == 'Canceled').astype('int64')
        df['CONTRACT_STATUS_IS_SIGN'] = \
                (df['NAME_CONTRACT_STATUS'] == 'Signed').astype('int64')
        df['CONTRACT_STATUS_IS_APPROVED'] = \
                (df['NAME_CONTRACT_STATUS'] == 'Approved').astype('int64')
        df['CONTRACT_STATUS_IS_RETURNED'] = \
                (df['NAME_CONTRACT_STATUS'] == 'Returned to the store').astype('int64')
        df['CONTRACT_STATUS_IS_DEBT'] = \
                (df['NAME_CONTRACT_STATUS'] == 'Amortized debt').astype('int64')
        
        df['SK_DPD_IS_POSITIVE'] = \
                (df['SK_DPD'] > 0).astype('int64')
        df['SK_DPD_DEF_IS_POSITIVE'] = \
                (df['SK_DPD_DEF'] > 0).astype('int64')
        df['CNT_INSTALMENT_FUTURE_IS_POSITIVE'] = \
                (df['CNT_INSTALMENT_FUTURE'] > 0).astype('int64')        
        # num_feature 
        df['CNT_INSTALMENT_FUTURE_PERCENTAGE'] = \
                df['CNT_INSTALMENT']/df['CNT_INSTALMENT_FUTURE']
        return df
    
    def fit_prev(self, df):
        df_groupby_sk_id_prev = df.groupby(['SK_ID_PREV'])
        df_feature = df_groupby_sk_id_prev.size().reset_index().\
                rename(columns = {0:'POS_NUMS_OF_MONTHS_BALANCE_RECORD'})
        df_feature['POS_NUMS_OF_MONTHS'] = \
                df_groupby_sk_id_prev['MONTHS_BALANCE'].first().abs()
        df_feature['POS_NUMS_OF_INSTALMENT'] = \
                df_groupby_sk_id_prev['CNT_INSTALMENT'].nunique()
        varlist_sum = list(df.columns)[8:14]
        varlist_last = ['CNT_INSTALMENT_FUTURE','CNT_INSTALMENT_FUTURE_IS_POSITIVE',
                        'CNT_INSTALMENT_FUTURE_PERCENTAGE']
        varlist_max = ['SK_DPD','SK_DPD_DEF','CNT_INSTALMENT',
                       'CNT_INSTALMENT_FUTURE_PERCENTAGE']
        varlist_min = ['CNT_INSTALMENT_FUTURE_PERCENTAGE','CNT_INSTALMENT']
        df_min = df_groupby_sk_id_prev[varlist_min].agg('min').\
                reset_index().\
                rename(columns = dict(zip(varlist_min,
                                          ['%s_min'%name for name in varlist_min])))
        df_feature = df_feature.merge(df_min, on = ['SK_ID_PREV'], how = 'left')
        df_sum = df_groupby_sk_id_prev[varlist_sum].agg('sum').\
                reset_index().\
                rename(columns = dict(zip(varlist_sum,
                                          ['%s_sum'%name for name in varlist_sum])))
        df_feature = df_feature.merge(df_sum, on = ['SK_ID_PREV'], how = 'left')
        df_max = df_groupby_sk_id_prev[varlist_max].agg('max').\
                reset_index().\
                rename(columns = dict(zip(varlist_max,
                                          ['%s_max'%name for name in varlist_max])))
        df_feature = df_feature.merge(df_max, on = ['SK_ID_PREV'], how = 'left')
        df_last = df_groupby_sk_id_prev[varlist_last].last().\
                reset_index().\
                rename(columns = dict(zip(varlist_last,
                                          ['%s_last'%name for name in varlist_last])))
        df_feature = df_feature.merge(df_last, on = ['SK_ID_PREV'], how = 'left')
        df_feature = df[['SK_ID_CURR','SK_ID_PREV']].merge(df_feature, 
                       on=['SK_ID_PREV'], how='left')
        return df
    
    def fit_curr(self, df):
        df_groupby_sk_id_curr = df.groupby(['SK_ID_CURR'])
        df_feature = df_groupby_sk_id_curr.size().reset_index().\
                rename(columns = {0:'POS_NUMS_OF_CREDITS'})
        varlist = list(df.columns)[2:]
        df_math = df_groupby_sk_id_curr[varlist].agg(['mean','max','min','sum','std']).\
                reset_index()
        df_math.columns = ['SK_ID_CURR']+ ['POS_%s_%s' % (var,fun) \
                                   for fun in ['mean','max','min','sum','std'] 
                                   for var in varlist[:-1]]
        df_feature = df_feature.merge(df_math, on='SK_ID_CURR',how='left')
        self.data = df_feature
        return df_feature
    def feature_extract(self, df):
        df = self.transform(df)
        df = self.fit_prev(df)
        df = self.fit_curr(df)
        return df