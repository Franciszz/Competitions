# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 19:26:07 2018

@author: Franc
"""
import pandas as pd
from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()
### 这代码有问题，速度偏慢，分组了两次，可以合在一起
class CreditCardBalFeatures(BaseTransformer):
    def __init__(self, fill_value, **kwargs):
        self.fill_value = fill_value
        self.df_feature = None
        
    def transform(self, df):
        df = df.sort_values(['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE']).\
            fillna(self.fill_value)
        df['NAME_MONTHS_BALANCE_IS_POSITIVE'] = \
            (df['AMT_BALANCE'] > 0).astype('int64')
        df['NAME_CONTRACT_STATUS_IS_REFUSED'] = \
            (df['NAME_CONTRACT_STATUS'] == 'Refused').astype('int64')
        
        df['AMT_BALANCE_DRAWINGS_RATIO'] = \
            df['AMT_DRAWINGS_CURRENT']/df['AMT_BALANCE']
        df['AMT_DRAWINGS_AMT_RATIO'] = \
            df['CNT_DRAWINGS_ATM_CURRENT']/df['AMT_DRAWINGS_CURRENT']
        df['AMT_DRAWINGS_POS_RATIO'] = \
            df['AMT_DRAWINGS_ATM_CURRENT']/df['AMT_DRAWINGS_CURRENT']
        df['AMT_DRAWINGS_OTHER_RATIO'] = \
            df['AMT_DRAWINGS_OTHER_CURRENT']/df['AMT_DRAWINGS_CURRENT']
        df['AMT_DRWAING_CNT_RATIO'] = \
            df['AMT_DRAWINGS_CURRENT']/df['CNT_DRAWINGS_CURRENT']
        
        df['AMT_BALANCE_MINPAYMENT'] = \
            df['AMT_INST_MIN_REGULARITY']/df['AMT_BALANCE']
        df['AMT_PAYMENT_MINPAYMENT'] = \
            df['AMT_INST_MIN_REGULARITY']/df['AMT_PAYMENT_CURRENT']
        return df
        
    def fit(self, df):
        df_prev_features = \
            pd.DataFrame({'SK_ID_PREV':df.SK_ID_PREV.unique()})
        df_groupby_prev = df.groupby(['SK_ID_PREV'])
        df_prev_features['SURVIVAL_MONTHS'] = \
            df_groupby_prev['MONTHS_BALANCE'].min().abs()
            
        df_prev_features['AMT_BALANCE'] = \
            df_groupby_prev['AMT_BALANCE'].mean()
        df_prev_features['AMT_BALANCE_MAX'] = \
            df_groupby_prev['AMT_BALANCE'].max()
        df_prev_features['AMT_CREDIT_LIMIT'] = \
            df_groupby_prev['AMT_CREDIT_LIMIT_ACTUAL'].max()
            
        df_prev_features['AMT_DRAWINGS_CURRENT'] = \
            df_groupby_prev['AMT_DRAWINGS_CURRENT'].mean()
        df_prev_features['AMT_PAYMENT_CURRENT'] = \
            df_groupby_prev['AMT_PAYMENT_CURRENT'].mean()
        
        df_prev_features['CNT_DRAWINGS_CURRENT'] = \
            df_groupby_prev['CNT_DRAWINGS_CURRENT'].mean()
        df_prev_features['CNT_INSTALMENT_MATURE_RATIO'] = \
            df_groupby_prev['CNT_INSTALMENT_MATURE_CUM'].\
            max()/df_prev_features['SURVIVAL_MONTHS']
            
        df_prev_features['SK_DPD'] = \
            df_groupby_prev['SK_DPD'].max()
        df_prev_features['SK_DPD_DEF'] = \
            df_groupby_prev['SK_DPD_DEF'].max()
        
        df_prev_features['NAME_AMT_BALANCE_IS_POSITIVE'] = \
            df_groupby_prev['NAME_MONTHS_BALANCE_IS_POSITIVE'].\
                sum()/df_prev_features['SURVIVAL_MONTHS']
        df_prev_features['NAME_CONTRACT_STATUS_IS_REFUSED'] = \
            df_groupby_prev['NAME_CONTRACT_STATUS_IS_REFUSED'].sum()
        
        for var in ['AMT_BALANCE_DRAWINGS_RATIO','AMT_DRAWINGS_AMT_RATIO',
                    'AMT_DRAWINGS_POS_RATIO','AMT_DRAWINGS_OTHER_RATIO',
                    'AMT_DRWAING_CNT_RATIO','AMT_BALANCE_MINPAYMENT',
                    'AMT_PAYMENT_MINPAYMENT']:
            df_prev_features[var] = \
                df_groupby_prev[var].mean()
        return df_prev_features
        
    def feature_extract(self, df):
        df = df.groupby(['SK_ID_CURR']).apply(self.fit).reset_index(drop=False)
        df_feature = pd.DataFrame({'SK_ID_CURR':df.SK_ID_CURR.unique()})
        df_groupby_curr = df.groupby(['SK_ID_CURR'])
        df_feature['CREDIT_nums_card_total'] = \
            df_groupby_curr.size()
        varlist_max = ['SURVIVAL_MONTHS','AMT_BALANCE','AMT_BALANCE_MAX',
                       'AMT_CREDIT_LIMIT','CNT_INSTALMENT_MATURE_RATIO',
                       'SK_DPD','SK_DPD_DEF','NAME_AMT_BALANCE_IS_POSITIVE',
                       'NAME_CONTRACT_STATUS_IS_REFUSED','AMT_BALANCE_DRAWINGS_RATIO',
                       'AMT_DRAWINGS_AMT_RATIO','AMT_DRAWINGS_POS_RATIO',
                       'AMT_DRAWINGS_OTHER_RATIO','AMT_DRWAING_CNT_RATIO',
                       'AMT_BALANCE_MINPAYMENT','AMT_PAYMENT_MINPAYMENT']
        varlist_sum = ['AMT_BALANCE','AMT_BALANCE_MAX','SK_DPD','SK_DPD_DEF',
                       'AMT_BALANCE_DRAWINGS_RATIO','AMT_DRAWINGS_AMT_RATIO',
                       'AMT_DRAWINGS_POS_RATIO','AMT_DRAWINGS_OTHER_RATIO',
                       'AMT_DRWAING_CNT_RATIO','AMT_BALANCE_MINPAYMENT',
                       'AMT_PAYMENT_MINPAYMENT']
        varlist_median = ['SURVIVAL_MONTHS','AMT_BALANCE','AMT_BALANCE_MAX',
                          'AMT_CREDIT_LIMIT','AMT_DRAWINGS_CURRENT',
                          'AMT_PAYMENT_CURRENT','CNT_DRAWINGS_CURRENT',
                          'CNT_INSTALMENT_MATURE_RATIO','SK_DPD','SK_DPD_DEF',
                          'NAME_AMT_BALANCE_IS_POSITIVE','AMT_BALANCE_DRAWINGS_RATIO',
                          'AMT_DRAWINGS_AMT_RATIO','AMT_DRAWINGS_POS_RATIO',
                          'AMT_DRAWINGS_OTHER_RATIO','AMT_DRWAING_CNT_RATIO',
                          'AMT_BALANCE_MINPAYMENT','AMT_PAYMENT_MINPAYMENT',
                          'NAME_CONTRACT_STATUS_IS_REFUSED']
        varlist_min = ['SURVIVAL_MONTHS','AMT_BALANCE','AMT_BALANCE_MAX',
                       'AMT_CREDIT_LIMIT','AMT_PAYMENT_CURRENT',
                       'CNT_INSTALMENT_MATURE_RATIO','SK_DPD','SK_DPD_DEF',
                       'NAME_AMT_BALANCE_IS_POSITIVE','AMT_BALANCE_DRAWINGS_RATIO',
                       'AMT_DRAWINGS_AMT_RATIO','AMT_DRAWINGS_POS_RATIO',
                       'AMT_DRAWINGS_OTHER_RATIO','AMT_DRWAING_CNT_RATIO',
                       'AMT_BALANCE_MINPAYMENT','AMT_PAYMENT_MINPAYMENT',
                       'NAME_CONTRACT_STATUS_IS_REFUSED']
        df_feature['CREDIT_survival_month_'] = \
            df_groupby_curr['SURVIVAL_MONTHS'].max()
        for var in varlist_max:
            df_feature['CREDIT_{}_max'.format(var.lower())] = \
                df_groupby_curr[var].max()
        for var in varlist_min:
            df_feature['CREDIT_{}_min'.format(var.lower())] = \
                df_groupby_curr[var].min()
        for var in varlist_sum:
            df_feature['CREDIT_{}_sum'.format(var.lower())] = \
                df_groupby_curr[var].sum()
        for var in varlist_median:
            df_feature['CREDIT_{}_med'.format(var.lower())] = \
                df_groupby_curr[var].median()
        df_feature = df_feature.reset_index(drop=True)
        return df_feature
            
    def outcome(self, df):
        self.df_feature = self.transform(df).pipe(self.feature_extract)
        return self.df_feature
        
