# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 15:19:11 2018

@author: Franc
"""

import numpy as np

from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()

class InstalFeature(BaseTransformer):
    """cat_feature, num_feature待补充"""   
    def __init__(self, **kwargs):
        self.data = None
        self.cat_feature = []
        self.num_feature = []
        self.na_value = np.nan
        self.na_fill = 0
    
    def transform(self, df):
        df = df.sort_values(['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_NUMBER'])
        # cat_feature
        df['PAYMENT_OVERDUE'] = \
                (df['DAYS_ENTRY_PAYMENT'] > df['DAYS_INSTALMENT']).astype('int64')
        df['PAYMENT_NOT_ENOUGH'] = \
                (df['AMT_PAYMENT'] < df['AMT_INSTALMENT']).astype('int64')
        df['PAYMENT_FOR_CREDIT'] = \
                (df['NUM_INSTALMENT_VERSION']==0).astype('int64')
        # num_feature
        df['RATIO_PAYMENT_ENTRY'] = \
                df['DAYS_ENTRY_PAYMENT']/df['DAYS_INSTALMENT']
        df['RATIO_PAYMENT_INSTALMENT'] = \
                df['AMT_PAYMENT']/df['AMT_INSTALMENT']
        return df
    
    def fit_prev(self, df):
        df_groupby_sk_id_prev = df.groupby(['SK_ID_PREV'])
        df_feature = df_groupby_sk_id_prev.size().reset_index().\
                rename(columns = {0:'INSTL_NUMS_OF_INSTALMENT_VERSION'})
        df_feature['INSTAL_NUMS_OF_INSTALMENTS'] = \
                df_groupby_sk_id_prev['NUM_INSTALMENT_NUMBER'].max()
        df_feature['INSTAL_NUMS_OF_INSTALMENTS_VERSION'] = \
                df_groupby_sk_id_prev['NUM_INSTALMENT_VERSION'].nunique()
        varlist = list(df.columns[4:])
        df_avg = df_groupby_sk_id_prev[varlist].agg('mean').\
                reset_index().\
                rename(columns = dict(zip(varlist,
                                          ['%s'%name for name in varlist])))
        df_feature = df_feature.merge(df_avg, on = ['SK_ID_PREV'], how = 'left')
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
                                   for var in varlist]
        df_feature = df_feature.merge(df_math, on='SK_ID_CURR',how='left')
        self.data = df_feature
        return df_feature
    
    def feature_extract(self, df):
        df = self.transform(df)
        df = self.fit_prev(df)
        df = self.fit_curr(df)
        return df