# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 12:00:25 2018

@author: Franc
"""

import numpy as np

from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()

class CreditFeature(BaseTransformer):
    """cat_feature, num_feature待补充"""  
    
    def __init__(self, **kwargs):
        self.data = None
        self.cat_feature = []
        self.num_feature = []
        self.na_value = np.nan
        self.na_fill = 0
    
    def transform(self, df):
        df = df.sort_values(['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'])
        df.loc[df['AMT_DRAWINGS_ATM_CURRENT'] < 0,'AMT_DRAWINGS_ATM_CURRENT'] = np.nan
        df.loc[df['AMT_DRAWINGS_CURRENT'] < 0,'AMT_DRAWINGS_CURRENT'] = np.nan
        # cat_feature
        df['NAME_CONTRACT_IS_COMPLETED'] = \
            (df['NAME_CONTRACT_STATUS'] == 'Completed').astype('int64')
        df['NAME_CONTRACT_IS_APPROVED'] = \
            (df['NAME_CONTRACT_STATUS']=='Approved').astype('int64')
        df['NAME_CONTRACT_IS_REFUESD'] = \
            (df['NAME_CONTRACT_STATUS']=='Refused').astype('int64')
        df['NAME_CONTRACT_IS_PROPOSAL'] = \
            (df['NAME_CONTRACT_STATUS']=='Sent proposal').astype('int64')
        df['NAME_CONTRACT_IS_DEMAND'] = \
            (df['NAME_CONTRACT_STATUS']=='Demand').astype('int64')
        df['NAME_CONTRACT_IS_SIGNED'] = \
            (df['NAME_CONTRACT_STATUS']=='Signed').astype('int64')
        
        df['NAME_BALANCE_IS_POSITIVE'] = \
            (df['AMT_BALANCE'] > 0).astype('int64')
        df['NAME_BALANCE_LT_PAYMENT'] = \
            (df['AMT_BALANCE'] > df['AMT_PAYMENT_TOTAL_CURRENT']).astype('int64')
        df['NAME_BALANCE_LT_MIN_PAYMENT'] = \
            (df['AMT_BALANCE'] > df['AMT_INST_MIN_REGULARITY']).astype('int64')
        df['NAME_RECEIVE_LT_MIN_PAYMENT'] = \
            (df['AMT_RECEIVABLE_PRINCIPAL'] > df['AMT_INST_MIN_REGULARITY']).astype('int64')
        df['NAME_RECEIVE_LT_PAYMENT'] = \
            (df['AMT_RECEIVABLE_PRINCIPAL'] > df['AMT_PAYMENT_TOTAL_CURRENT']).astype('int64')
        df['NAME_PAYMENT_LT_REGULATION'] = \
            (df['AMT_PAYMENT_CURRENT'] > df['AMT_INST_MIN_REGULARITY']).astype('int64')
        df['NAME_PAYMENT_LT_TOTAL'] = \
            (df['AMT_PAYMENT_CURRENT'] > df['AMT_PAYMENT_TOTAL_CURRENT']).astype('int64')
        df['NAME_CREDIT_LIMIT_IS_NONE'] = \
            (df['AMT_CREDIT_LIMIT_ACTUAL'] == 0).astype('int64')
        df['NAME_CREDIT_LIMIT_IS_NONE'] = \
            (df['AMT_CREDIT_LIMIT_ACTUAL'] == 0).astype('int64')
            
        # num_feature
        df['AMT_RATIO_DRAWINGS_ATM'] = \
            df['AMT_DRAWINGS_ATM_CURRENT']/df['AMT_DRAWINGS_CURRENT']
        df['AMT_RATIO_DRAWINGS_POS'] = \
            df['AMT_DRAWINGS_POS_CURRENT']/df['AMT_DRAWINGS_CURRENT']
        df['AMT_RATIO_DRAWINGS_OTHER'] = \
            df['AMT_DRAWINGS_OTHER_CURRENT']/df['AMT_DRAWINGS_CURRENT']
        
        df['AMT_RATIO_DRAWINGS_BALANCE'] = \
            df['AMT_DRAWINGS_CURRENT']/df['AMT_BALANCE']
        df['AMT_RATIO_DRAWINGS_PAYMENT'] = \
            df['AMT_DRAWINGS_CURRENT']/df['AMT_PAYMENT_TOTAL_CURRENT']
        df['AMT_RATIO_DRAWINGS_RECEIVABLE'] = \
            df['AMT_DRAWINGS_CURRENT']/df['AMT_RECEIVABLE_PRINCIPAL']
        df['AMT_RATIO_DRAWINGS_CREDIT'] = \
            df['AMT_DRAWINGS_CURRENT']/df['AMT_CREDIT_LIMIT_ACTUAL']
      
        df['AMT_RATIO_BALANCE_CREDIT'] = \
            df['AMT_BALANCE']/df['AMT_CREDIT_LIMIT_ACTUAL']
        df['AMT_RATIO_BALANCE_PAYMENT'] = \
            df['AMT_BALANCE']/df['AMT_PAYMENT_TOTAL_CURRENT']
        df['AMT_RATIO_BALANCE_MIN_REGULATION'] = \
            df['AMT_BALANCE']/df['AMT_INST_MIN_REGULARITY']
        df['AMT_RATIO_BALANCE_RECEIVABLE'] = \
            df['AMT_BALANCE']/df['AMT_RECEIVABLE_PRINCIPAL']
        
        df['AMT_RATIO_PAYMENT_MIN'] = \
            df['AMT_INST_MIN_REGULARITY'] / df['AMT_PAYMENT_CURRENT']
        df['AMT_RATIO_MIN_TOTAL'] = \
            df['AMT_INST_MIN_REGULARITY'] / df['AMT_PAYMENT_TOTAL_CURRENT']
        df['AMT_RATIO_PAYMENT_TOTAL'] = \
            df['AMT_PAYMENT_TOTAL_CURRENT'] / df['AMT_PAYMENT_CURRENT']  
        df['AMT_RATIO_PAYMENT_REVEIVABLE'] = \
            df['AMT_PAYMENT_CURRENT'] / df['AMT_RECEIVABLE_PRINCIPAL']
        df['AMT_RATIO_MIN_RECEIVABLE'] = \
            df['AMT_PAYMENT_TOTAL_CURRENT'] / df['AMT_RECEIVABLE_PRINCIPAL']
        df['AMT_RATIO_TOTAL_RECEIVABLE'] = \
            df['AMT_INST_MIN_REGULARITY'] / df['AMT_RECEIVABLE_PRINCIPAL']
        
        df['CNT_RATIO_DRAWINGS_ATM'] = \
            df['CNT_DRAWINGS_ATM_CURRENT']/df['CNT_DRAWINGS_CURRENT']
        df['CNT_RATIO_DRAWINGS_POS'] = \
            df['CNT_DRAWINGS_POS_CURRENT']/df['CNT_DRAWINGS_CURRENT']
        df['CNT_RATIO_DRAWINGS_OTHER'] = \
            df['CNT_DRAWINGS_OTHER_CURRENT']/df['CNT_DRAWINGS_CURRENT']
        df['CNT_RATIO_DRAWINGS'] = \
            df['CNT_DRAWINGS_CURRENT']/df['CNT_DRAWINGS_CURRENT']
        df['CNT_RATIO_DRAWINGS_ATM'] = \
            df['CNT_DRAWINGS_ATM_CURRENT']/df['CNT_DRAWINGS_ATM_CURRENT']
        df['CNT_RATIO_DRAWINGS_POS'] = \
            df['CNT_DRAWINGS_POS_CURRENT']/df['CNT_DRAWINGS_POS_CURRENT']
        df['CNT_RATIO_DRAWINGS_OTHER'] = \
            df['CNT_DRAWINGS_OTHER_CURRENT']/df['CNT_DRAWINGS_OTHER_CURRENT']
        return df
    
    def fit_prev(self, df):
        df_groupby_sk_id_prev = df.groupby(['SK_ID_PREV'])
        df_feature = df_groupby_sk_id_prev.size().reset_index().\
                rename(columns = {0:'CREDIT_NUMS_OF_MONTHS_BALANCE_RECORD'})
        df_feature['CREDIT_NUMS_OF_MONTHS'] = \
                df_groupby_sk_id_prev['MONTHS_BALANCE'].first().abs()
        varlist_mean = ['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL',
                        'AMT_DRAWINGS_CURRENT','AMT_INST_MIN_REGULARITY',
                        'AMT_PAYMENT_CURRENT','AMT_PAYMENT_TOTAL_CURRENT',
                        'AMT_RECIVABLE','CNT_DRAWINGS_CURRENT',
                        'AMT_RECEIVABLE_PRINCIPAL'] + list(df.columns)[29:62]
        varlist_sum = list(df.columns)[23:38]
        varlist_last = ['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL',
                        'AMT_DRAWINGS_CURRENT','AMT_INST_MIN_REGULARITY',
                        'AMT_PAYMENT_CURRENT','AMT_PAYMENT_TOTAL_CURRENT',
                        'AMT_RECIVABLE','CNT_DRAWINGS_CURRENT',
                        'AMT_RECEIVABLE_PRINCIPAL'] + list(df.columns)[29:62]
        varlist_max = ['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL',
                       'AMT_DRAWINGS_CURRENT','AMT_INST_MIN_REGULARITY',
                       'AMT_PAYMENT_CURRENT','AMT_PAYMENT_TOTAL_CURRENT',
                       'AMT_RECIVABLE','CNT_DRAWINGS_CURRENT',
                       'AMT_RECEIVABLE_PRINCIPAL','CNT_INSTALMENT_MATURE_CUM',
                       'SK_DPD_DEF','SK_DPD'] + list(df.columns)[23:29]
        df_mean = df_groupby_sk_id_prev[varlist_mean].agg('mean').\
                reset_index().\
                rename(columns = dict(zip(varlist_mean,
                                          ['%s_avg'%name for name in varlist_mean])))
        df_feature = df_feature.merge(df_mean, on = ['SK_ID_PREV'], how = 'left')
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
        return df_feature
    
    def fit_curr(self, df):
        df_groupby_sk_id_curr = df.groupby(['SK_ID_CURR'])
        df_feature = df_groupby_sk_id_curr.size().reset_index().\
                rename(columns = {0:'CREDIT_NUMS_OF_CREDITS'})
        varlist = list(df.columns)[2:]
        df_math = df_groupby_sk_id_curr[varlist].agg(['mean','max','min','std']).\
                reset_index()
        df_math.columns = ['SK_ID_CURR']+ ['CREDIT_%s_%s' % (var,fun) \
                                   for fun in ['mean','max','min','std'] 
                                   for var in varlist]
        df_feature = df_feature.merge(df_math, on='SK_ID_CURR',how='left')
        self.data = df_feature
        return df_feature
    def feature_extract(self, df):
        df = self.transform(df)
        df = self.fit_prev(df)
        df = self.fit_curr(df)
        return df