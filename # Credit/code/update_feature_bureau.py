# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 11:01:21 2018

@author: Franc
"""

import numpy as np
import pandas as pd

from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()

class BureauFeature(BaseTransformer):
    
    def __init__(self, na_value = np.nan, fill_value = 0,**kwargs):
        self.data = None
        self.cat_feature = []
        self.num_feature = []
        self.na_value = na_value
        self.na_fill = fill_value
    
    def transform_bal(self, df):
        df['BUREAU_IS_DPD'] = \
            (df['STATUS'].isin(['1','2','3','4','5'])).astype('int64')
        df['BUREAU_DPD_STATUS'] = df['STATUS'].\
            apply(lambda x: int(x) if x in ['1','2','3','4','5'] else 0)
        df = df.groupby('SK_ID_BUREAU')[['BUREAU_IS_DPD',
                      'BUREAU_DPD_STATUS']].agg('max')
        return df

    def transform(self, df, df_bal):
        df = df.sort_values(['SK_ID_CURR']).reset_index(drop=True)
        
        df.loc[df['DAYS_CREDIT_ENDDATE'] < -40000,
               'DAYS_CREDIT_ENDDATE'] = self.na_fill
        df.loc[df['DAYS_CREDIT_UPDATE'] < -40000,
               'DAYS_CREDIT_UPDATE'] = self.na_fill
        df.loc[df['DAYS_ENDDATE_FACT'] < -40000,
               'DAYS_ENDDATE_FACT'] = self.na_fill
        
        df['AMT_CREDIT_SUM'].fillna(self.na_fill, inplace=True)
        df['AMT_CREDIT_SUM_DEBT'].fillna(self.na_fill, inplace=True)
        df['AMT_CREDIT_SUM_OVERDUE'].fillna(self.na_fill, inplace=True)
        df['CNT_CREDIT_PROLONG'].fillna(self.na_fill, inplace=True)
        
        df_dpd = self.transform_bal(df_bal)
        df = df.join(df_dpd,on='SK_ID_BUREAU',how='left')
        # cat_feature
        df['CREDIT_ACTIVE_IS_ACTIVE'] = \
                (df['CREDIT_ACTIVE'] == 'Active').astype('int64')
        df['CREDIT_ACTIVE_IS_BAD'] = \
                (df['CREDIT_ACTIVE'] == 'Bad debt').astype('int64')
        df['CREDIT_ACTIVE_IS_CLOSED'] = \
                (df['CREDIT_ACTIVE'] == 'Closed').astype('int64')
        df['CREDIT_ACTIVE_IS_SOLD'] = \
                (df['CREDIT_ACTIVE'] == 'Sold').astype('int64')
        
        df['CREDIT_CURRENCY_IS_1'] = \
                (df['CREDIT_CURRENCY'] == 'currency 1').astype('int64')
        df['CREDIT_CURRENCY_IS_2'] = \
                (df['CREDIT_CURRENCY'] == 'currency 2').astype('int64')
        df['CREDIT_CURRENCY_IS_3'] = \
                (df['CREDIT_CURRENCY'] == 'currency 3').astype('int64')
        df['CREDIT_CURRENCY_IS_4'] = \
                (df['CREDIT_CURRENCY'] == 'currency 4').astype('int64')
        
        df['CREDIT_TYPE_IS_REVOLVING'] = \
                (df['CREDIT_TYPE'] == 'Credit card').astype('int64')
        df['CREDIT_TYPE_IS_MICRO'] = \
                (df['CREDIT_TYPE'] == 'Microloan').astype('int64')
        df['CREDIT_TYPE_IS_CASH'] = \
                (df['CREDIT_TYPE'] == 'Cash loan (non-earmarked)').astype('int64')
        df['CREDIT_TYPE_IS_BUSINESS'] = \
                (df['CREDIT_TYPE'].isin(['Loan for business development',
                                         'Loan for working capital replenishment',
                                         'Loan for the purchase of equipment',
                                         'Mobile operator loan','Mortgage',
                                         'Interbank credit'])).astype('int64')
        df['CREDIT_TYPE_IS_CAPITAL'] = \
                (df['CREDIT_TYPE'] == 'Loan for purchase of shares (margin lending)').\
                astype('int64')
        df['CREDIT_TYPE_IS_ASSET'] = \
                (df['CREDIT_TYPE'].isin(['Car loan','Real estate loan'])).astype('int64')
        
        # num_feature
        df['AMT_RATIO_ANNUITY_LIMIT'] = \
            df['AMT_ANNUITY']/df['AMT_CREDIT_SUM_LIMIT']
        df['AMT_RATIO_ANNUITY_SUM'] = \
            df['AMT_ANNUITY']/df['AMT_CREDIT_SUM']
        df['AMT_RATIO_ANNUITY_DEBT'] = \
            df['AMT_ANNUITY']/df['AMT_CREDIT_SUM_DEBT']
        df['AMT_RATIO_DEBT_SUM'] = \
            df['AMT_CREDIT_SUM_DEBT']/df['AMT_CREDIT_SUM']
        df['AMT_RATIO_LIMIT_SUM'] = \
            df['AMT_CREDIT_SUM_LIMIT']/df['AMT_CREDIT_SUM']
        df['AMT_RATIO_OVERDUE_SUM'] = \
            df['AMT_CREDIT_SUM_OVERDUE']/df['AMT_CREDIT_SUM']
        df['AMT_RATIO_MAXOVERDUE_SUM'] = \
            df['AMT_CREDIT_MAX_OVERDUE']/df['AMT_CREDIT_SUM']
        df['AMT_RATIO_LIMIT_OVERDUE'] = \
            df['AMT_CREDIT_SUM_OVERDUE']/df['AMT_CREDIT_SUM_LIMIT']
        df['AMT_RATIO_DEBT_OVERDUE'] = \
            df['AMT_CREDIT_SUM_OVERDUE']/df['AMT_CREDIT_SUM_DEBT']
        
        df['DAYS_RATIO_OVERDUE_CREDIT'] = \
            df['CREDIT_DAY_OVERDUE']/df['DAYS_CREDIT']
        df['DAYS_RATIO_OVERDUE_CREDIT'] = \
            df['CREDIT_DAY_OVERDUE']/df['DAYS_CREDIT']
        df['DAYS_RATIO_OVERDUE_FACT'] = \
            df['CREDIT_DAY_OVERDUE']/df['DAYS_CREDIT_ENDDATE']
        return df
    
    def fit(self, df):
        df_groupby_sk_id_curr = df.groupby(['SK_ID_CURR'])
        df_feature = df_groupby_sk_id_curr.size().reset_index().\
                rename(columns = {0:'BUREAU_NUMS_OF_CARD'})
        
        count_feature = ['CREDIT_ACTIVE','CREDIT_CURRENCY','CREDIT_TYPE']
        df_feature_count = df_groupby_sk_id_curr[count_feature].agg('nunique').\
                reset_index().\
                rename(columns = dict(zip(count_feature,
                                          ['BUREAU_%s_NUNIQUE'%name for name in count_feature])))
        df_feature = df_feature.merge(df_feature_count, on = 'SK_ID_CURR', how = 'left')
                
        cat_feature = list(df.columns)[19:33] + ['BUREAU_IS_DPD']
        df_feature_sum = df_groupby_sk_id_curr[cat_feature].agg('sum').\
                reset_index().\
                rename(columns = dict(zip(cat_feature,
                                          ['BUREAU_%s_SUM'%name for name in cat_feature])))
        df_feature = df_feature.merge(df_feature_sum, on = 'SK_ID_CURR', how = 'left')
        
        df_feature_avg = df_groupby_sk_id_curr[cat_feature].agg('mean').\
                reset_index().\
                rename(columns = dict(zip(cat_feature,
                                          ['BUREAU_%s_MEAN'%name for name in cat_feature])))
        df_feature = df_feature.merge(df_feature_avg, on = 'SK_ID_CURR', how = 'left')
        
        num_feature = ['AMT_ANNUITY','AMT_CREDIT_MAX_OVERDUE','AMT_CREDIT_SUM',
                       'AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_OVERDUE',
                       'CNT_CREDIT_PROLONG','CREDIT_DAY_OVERDUE','DAYS_CREDIT',
                       'DAYS_CREDIT_ENDDATE','DAYS_CREDIT_UPDATE','DAYS_ENDDATE_FACT',
                       'BUREAU_DPD_STATUS'] + list(df.columns[33:])
        df_feature_num = df_groupby_sk_id_curr[num_feature].\
                agg(['sum','mean','median','max','min','std']).reset_index()
        df_feature_num.columns = ['SK_ID_CURR']+ ['BUREAU_%s_%s' % (var,fun) \
                                   for fun in ['sum','mean','median','max','min','std'] 
                                   for var in num_feature]
        df_feature = df_feature.merge(df_feature_num, on = 'SK_ID_CURR', how = 'left')
        self.data = df_feature
        return df_feature
    
    def feature_extract(self, df, df_bal):
        df = self.transform(df, df_bal)
        df = self.fit(df)
        return df
    
    