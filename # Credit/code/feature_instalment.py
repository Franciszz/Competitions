# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 22:52:28 2018

@author: Franc
"""
import pandas as pd
from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()

class InstalmentFeatures(BaseTransformer):
    def __init__(self, **kwargs):
        self.df_feature = None
        
    def transform(self, df):
        df = df.sort_values(['SK_ID_CURR','SK_ID_PREV','NUM_INSTALMENT_NUMBER'])
        df['PAYMENT_OVERDUE'] = \
            (df['DAYS_ENTRY_PAYMENT'] > df['DAYS_INSTALMENT']).astype('int64')
        df['PAYMENT_NOT_ENOUGH'] = \
            (df['AMT_PAYMENT'] < df['AMT_INSTALMENT']).astype('int64')
        df['PAYMENT_FOR_CREDIT'] = \
            (df['NUM_INSTALMENT_VERSION']==0).astype('int64')
        return df
    
    def fit(self, df):
        df_prev_feature = pd.DataFrame({'SK_ID_PREV':df['SK_ID_PREV'].unique()})
        df_groupby_prev = df.groupby(by=['SK_ID_PREV'])
        
        df_prev_feature['NUM_INSTALMENT_NUMBER'] = \
            df_groupby_prev['NUM_INSTALMENT_NUMBER'].max()
        df_prev_feature['INSTALMENT_IS_CREDIT'] = \
            df_groupby_prev['PAYMENT_FOR_CREDIT'].sum()/df_groupby_prev.size()
        df_prev_feature['PAYMENT_OVERDUE'] = \
            df_groupby_prev['PAYMENT_OVERDUE'].mean()
        df_prev_feature['PAYMENT_NOT_ENOUGH'] = \
            df_groupby_prev['PAYMENT_NOT_ENOUGH'].mean()
        return df_prev_feature
    
    def feature_extract(self,df):
        df = df.groupby(by=['SK_ID_CURR']).\
            apply(self.fit).reset_index()
        df_groupby_curr = df.groupby(['SK_ID_CURR'])
        df_feature = pd.DataFrame({
                'SK_ID_CURR':
                    df['SK_ID_CURR'].unique(),
                'PREV_num_of_application':
                    df_groupby_curr.size(),
                'PREV_num_of_instalment_median':
                    df_groupby_curr['NUM_INSTALMENT_NUMBER'].median(),
                'PREV_num_of_instalment_max':
                    df_groupby_curr['NUM_INSTALMENT_NUMBER'].max(),
                'PREV_instalment_is_credit_median':
                    df_groupby_curr['INSTALMENT_IS_CREDIT'].median(),
                'PREV_instalment_is_credit_max':
                    df_groupby_curr['INSTALMENT_IS_CREDIT'].max(),
                'PREV_payment_overdue_max':
                    df_groupby_curr['PAYMENT_OVERDUE'].max(),
                'PREV_payment_overdue_median':
                    df_groupby_curr['PAYMENT_OVERDUE'].median(),
                'PREV_payment_overdue_min':
                    df_groupby_curr['PAYMENT_OVERDUE'].min(),
                'PREV_payment_not_enough_max':
                    df_groupby_curr['PAYMENT_NOT_ENOUGH'].max(),
                'PREV_payment_not_enough_median':
                    df_groupby_curr['PAYMENT_NOT_ENOUGH'].median(),
                'PREV_payment_not_enough_min':
                    df_groupby_curr['PAYMENT_NOT_ENOUGH'].min()
                }).reset_index(drop=True)
        return df_feature
    
    def outcome(self, df):
        self.df_feature = self.transform(df).pipe(self.feature_extract)
        return self.df_feature

        
class PosBalFeatures(BaseTransformer):
    def __init__(self, **kwargs):
        self.df_feature = None
        
    def transform(self, df):
        df = df.sort_values(['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE'])
        df['CONTRACT_STATUS_IS_REFUSED'] = \
            (df['NAME_CONTRACT_STATUS'] == 'Refused').astype('int64')
        return df
    
    def fit(self, df):
        df_prev_feature = pd.DataFrame({'SK_ID_PREV':df['SK_ID_PREV'].unique()})
        df_groupby_prev = df.groupby(by=['SK_ID_PREV'])
        
        df_prev_feature['SURVIVAL_MONTHS'] = \
            df_groupby_prev['MONTHS_BALANCE'].min().abs()
        df_prev_feature['CNT_INSTALMENT'] = \
            df_groupby_prev['CNT_INSTALMENT'].max()
        df_prev_feature['CNT_INSTALMENT_FUTURE'] = \
            df_groupby_prev['CNT_INSTALMENT_FUTURE'].last()
        df_prev_feature['CNT_INSTALMENT_FUTURE_PERCENTAGE'] = \
            df_prev_feature['CNT_INSTALMENT_FUTURE']/df_prev_feature['CNT_INSTALMENT']
        df_prev_feature['CONTRANCT_STATUS_IS_REFUSED'] = \
            df_groupby_prev['CONTRACT_STATUS_IS_REFUSED'].\
                sum()/df_prev_feature['SURVIVAL_MONTHS']
        df_prev_feature['SK_DPD'] = \
            df_groupby_prev['SK_DPD'].max()
        df_prev_feature['SK_DPD_DEF'] = \
            df_groupby_prev['SK_DPD_DEF'].max()
        return df_prev_feature
    
    def feature_extract(self,df):
        df = df.groupby(by=['SK_ID_CURR']).\
            apply(self.fit).reset_index()
        df_groupby_curr = df.groupby(['SK_ID_CURR'])
        df_feature = pd.DataFrame({
                'SK_ID_CURR':
                    df['SK_ID_CURR'].unique(),
                'POS_num_of_pos_card':
                    df_groupby_curr.size(),
                'PREV_survival_month_median':
                    df_groupby_curr['SURVIVAL_MONTHS'].median(),
                'PREV_survival_month_min':
                    df_groupby_curr['SURVIVAL_MONTHS'].min(),
                'PREV_cnt_instalment_max':
                    df_groupby_curr['CNT_INSTALMENT'].max(),
                'PREV_cnt_instalment_median':
                    df_groupby_curr['CNT_INSTALMENT'].median(),
                'PREV_cnt_instalment_future_max':
                    df_groupby_curr['CNT_INSTALMENT_FUTURE'].max(),
                'PREV_cnt_instalment_future_median':
                    df_groupby_curr['CNT_INSTALMENT_FUTURE'].median(),
                'PREV_cnt_instalment_future_percentage_max':
                    df_groupby_curr['CNT_INSTALMENT_FUTURE_PERCENTAGE'].max(),
                'PREV_cnt_instalment_future_percentage_median':
                    df_groupby_curr['CNT_INSTALMENT_FUTURE_PERCENTAGE'].median(),
                'PREV_contract_is_refused_max':
                    df_groupby_curr['CNT_INSTALMENT_FUTURE'].min(),
                'PREV_contract_is_refused_median':
                    df_groupby_curr['CNT_INSTALMENT_FUTURE'].median(),
                'PREV_sk_dpd_median':
                    df_groupby_curr['SK_DPD'].median(),
                'PREV_sk_dpd_max':
                    df_groupby_curr['SK_DPD'].max(),
                'PREV_sk_dpd_def_median':
                    df_groupby_curr['SK_DPD_DEF'].median(),
                'PREV_sk_dpd_def_max':
                    df_groupby_curr['SK_DPD_DEF'].max()
                })
        return df_feature
    
    def outcome(self, df):
        self.df_feature = self.transform(df).pipe(self.feature_extract)
        return self.df_feature


