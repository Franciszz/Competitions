# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 18:49:13 2018

@author: Franc
"""

import numpy as np

from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()

class PreviousFeature(BaseTransformer):
    """cat_feature, num_feature待补充"""   
    def __init__(self, **kwargs):
        self.data = None
        self.cat_feature = []
        self.num_feature = []
        self.na_value = np.nan
        self.na_fill = 0
    
    def transform(self, df):
        df = df.replace({'XNA':self.na_value, 365243:self.na_value}).\
                sort_values(['SK_ID_CURR','SK_ID_PREV'])
        # Categorial feature
        df['NAME_CONTRACT_TYPE_IS_CONSUMER'] = \
                (df['NAME_CONTRACT_TYPE'] == 'Consumer loans').astype('int64')
        df['NAME_CONTRACT_TYPE_IS_CASH'] = \
                (df['NAME_CONTRACT_TYPE'] == 'Cash loans').astype('int64')
        df['NAME_CONTRACT_TYPE_IS_REVOLVING'] = \
                (df['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype('int64')
        
        df['NAME_WEEKDAY_APPR_IS_MONDAY'] = \
                (df['WEEKDAY_APPR_PROCESS_START'] == 'MONDAY').astype('int64')
        df['NAME_WEEKDAY_APPR_IS_TUESDAY'] = \
                (df['WEEKDAY_APPR_PROCESS_START'] == 'TUESDAY').astype('int64')
        df['NAME_WEEKDAY_APPR_IS_WEDNESDAY'] = \
                (df['WEEKDAY_APPR_PROCESS_START'] == 'WEDNESDAY').astype('int64')
        df['NAME_WEEKDAY_APPR_IS_THURSDAY'] = \
                (df['WEEKDAY_APPR_PROCESS_START'] == 'THURSDAY').astype('int64')
        df['NAME_WEEKDAY_APPR_IS_FRIDAY'] = \
                (df['WEEKDAY_APPR_PROCESS_START'] == 'FRIDAY').astype('int64')
        df['NAME_WEEKDAY_APPR_IS_SATURDAY'] = \
                (df['WEEKDAY_APPR_PROCESS_START'] == 'SATURDAY').astype('int64')
        df['NAME_WEEKDAY_APPR_IS_SUNDAY'] = \
                (df['WEEKDAY_APPR_PROCESS_START'] == 'SUNDAY').astype('int64')
        df['DATE_HOUR_IS_NIGHT'] = \
            ((df['HOUR_APPR_PROCESS_START']>20)&
             (df['HOUR_APPR_PROCESS_START']<6)).astype('int64')
        df['DATE_HOUR_IS_AM'] = \
            ((df['HOUR_APPR_PROCESS_START']>5)&
             (df['HOUR_APPR_PROCESS_START']<12)).astype('int64')
        df['DATE_HOUR_IS_PM'] = \
            ((df['HOUR_APPR_PROCESS_START']>11)&
             (df['HOUR_APPR_PROCESS_START']<21)).astype('int64')
        
        df['FLAG_LAST_APPL_PER_CONTRACT'] = \
                (df['FLAG_LAST_APPL_PER_CONTRACT']=='Y').astype('int64')
                
        df['NAME_CASH_LOAN_PURPOSE_IS_URGENT'] = \
            (df['NAME_CASH_LOAN_PURPOSE'].\
                 isin(['Payments on other loans','Urgent needs',
                       'Refusal to name the goal'])).astype('int64')
        df['NAME_CASH_LOAN_PURPOSE_IS_REGULAR'] = \
            (df['NAME_CASH_LOAN_PURPOSE'].\
                 isin(['Everyday expenses', 'Hobby', 'Gasification / water supply',
                       'Repairs','Car Repairs','Journey','Medichine',
                       'Education','Gasification / water supply'])).astype('int64')
        df['NAME_CASH_LOAN_PURPOSE_IS_BUSINESS'] = \
            (df['NAME_CASH_LOAN_PURPOSE'].\
                 isin(['Business development',
                       'Money for a third person'])).astype('int64')
        
        df['NAME_CONTRACT_IS_CANCEL'] = \
            (df['NAME_CONTRACT_STATUS'] == 'Cancelled').astype('int64')
        df['NAME_CONTRACT_IS_REFUSED'] = \
            (df['NAME_CONTRACT_STATUS']=='Refused').astype('int64')
        df['NAME_CONTRACT_IS_UNUSED'] = \
            (df['NAME_CONTRACT_STATUS']=='Unused offer').astype('int64')
        df['NAME_CONTRACT_IS_APPROVED'] = \
            (df['NAME_CONTRACT_STATUS']=='Approved').astype('int64')
        
        df['NAME_PAYMENT_TYPE_FROM_CASH'] = \
            (df['NAME_PAYMENT_TYPE'] == 'Cash through the bank').\
                    astype('int64')
        df['NAME_PAYAMENT_TYPE_FROM_ACCOUNT'] = \
            (df['NAME_PAYMENT_TYPE'] == 'Non-cash from your account').\
                    astype('int64')
        df['NAME_PAYAMENT_TYPE_FROM_EMPLOYER'] = \
            (df['NAME_PAYMENT_TYPE'] == 'Cashless from the account of the employer').\
                    astype('int64')
        
        df['NAME_CODE_REJECT_REASON_IS_XAP'] = \
            (df['CODE_REJECT_REASON'] == 'XAP').astype('int64')
        df['NAME_CODE_REJECT_REASON_IS_HC'] = \
            (df['CODE_REJECT_REASON']=='HC').astype('int64')
        df['NAME_CODE_REJECT_REASON_IS_LIMIT'] = \
            (df['CODE_REJECT_REASON']=='LIMIT').astype('int64')
        df['NAME_CODE_REJECT_REASON_IS_CLIENT'] = \
            (df['CODE_REJECT_REASON']=='CLIENT').astype('int64')
        df['NAME_CODE_REJECT_REASON_IS_SCOFR'] = \
            (df['CODE_REJECT_REASON'] == 'SCOFR').astype('int64')
        df['NAME_CODE_REJECT_REASON_IS_SCO'] = \
            (df['CODE_REJECT_REASON']=='SCO').astype('int64')
        df['NAME_CODE_REJECT_REASON_IS_VERIF'] = \
            (df['CODE_REJECT_REASON']=='VERIF').astype('int64')
        
        df['NAME_TYPE_SUITE_IS_FAMILY'] = \
                (df['NAME_TYPE_SUITE'] == 'Family').astype('int64')
        df['NAME_TYPE_SUITE_IS_CHILDREN'] = \
                (df['NAME_TYPE_SUITE'] == 'Children').astype('int64')
        df['NAME_TYPE_SUITE_IS_SPOUSER'] = \
                (df['NAME_TYPE_SUITE'] == 'Spouse, partner').astype('int64')
        df['NAME_TYPE_SUITE_IS_ALONE'] = \
                (df['NAME_TYPE_SUITE'] == 'Unaccompanied').astype('int64')
        df['NAME_TYPE_SUITE_IS_OTHER_A'] = \
                (df['NAME_TYPE_SUITE'] == 'Other_A').astype('int64')
        df['NAME_TYPE_SUITE_IS_OTHER_B'] = \
                (df['NAME_TYPE_SUITE'] == 'Other_B').astype('int64')
        df['NAME_TYPE_SUITE_IS_GROUP'] = \
                (df['NAME_TYPE_SUITE'] == 'Group of people').astype('int64')
        
        df['NAME_CLIENT_TYPE_IS_NEW'] = \
                (df['NAME_CLIENT_TYPE'] == 'New').astype('int64')
        df['NAME_CLIENT_TYPE_IS_REFRESH'] = \
                (df['NAME_CLIENT_TYPE'] == 'Refreshed').astype('int64')
        df['NAME_CLIENT_TYPE_IS_REPEATER'] = \
                (df['NAME_CLIENT_TYPE'] == 'Repeater').astype('int64')
        
        df['NAME_PORTFOLIO_IS_POS'] = \
                (df['NAME_CLIENT_TYPE'] == 'POS').astype('int64')
        df['NAME_PORTFOLIO_IS_CASH'] = \
                (df['NAME_CLIENT_TYPE'] == 'Cash').astype('int64')
        df['NAME_PORTFOLIO_IS_CARDS'] = \
                (df['NAME_CLIENT_TYPE'] == 'Cards').astype('int64')
        df['NAME_PORTFOLIO_IS_CARS'] = \
                (df['NAME_CLIENT_TYPE'] == 'Cars').astype('int64')
        
        df['NAME_PRODUCT_TYPE_IS_XSHELL'] = \
                (df['NAME_PRODUCT_TYPE'] == 'x-shell').astype('int64')
        df['NAME_PRODUCT_TYPE_IS_WALKIN'] = \
                (df['NAME_PRODUCT_TYPE'] == 'walk-in').astype('int64')
        
        df['NAME_CHANNEL_TYPE_IS_COUNTRY'] = \
                (df['CHANNEL_TYPE'] == 'Country-wide').astype('int64')
        df['NAME_CHANNEL_TYPE_IS_CENTER'] = \
                (df['CHANNEL_TYPE'] == 'Contact center').astype('int64')
        df['NAME_CHANNEL_TYPE_IS_OFFICE'] = \
                (df['CHANNEL_TYPE'] == 'Credit and cash offices').astype('int64')
        df['NAME_CHANNEL_TYPE_IS_STONE'] = \
                (df['CHANNEL_TYPE'] == 'Stone').astype('int64')
        df['NAME_CHANNEL_TYPE_IS_LOCAL'] = \
                (df['CHANNEL_TYPE'] == 'Regional / Local').astype('int64')
        df['NAME_CHANNEL_TYPE_IS_AP'] = \
                (df['CHANNEL_TYPE'] == 'AP+ (Cash loan)').astype('int64')
        df['NAME_CHANNEL_TYPE_IS_SALES'] = \
                (df['CHANNEL_TYPE'] == 'Channel of corporate sales').astype('int64')
        df['NAME_CHANNEL_TYPE_IS_DEALER'] = \
                (df['CHANNEL_TYPE'] == 'Car dealer').astype('int64')
        
        df['NAME_SELLER_INDUSTRY_IS_CONSUMING'] = \
                (df['NAME_SELLER_INDUSTRY']=='Clothing').astype('int64')
        df['NAME_SELLER_INDUSTRY_IS_CONNECTIVITY'] = \
                (df['NAME_SELLER_INDUSTRY']=='Connectivity').astype('int64')
        df['NAME_SELLER_INDUSTRY_IS_ELECTRONICS'] = \
                (df['NAME_SELLER_INDUSTRY']=='Consumer electronics').astype('int64')
        df['NAME_SELLER_INDUSTRY_IS_INDUSTRY'] = \
                (df['NAME_SELLER_INDUSTRY']=='Industry').astype('int64')
        df['NAME_SELLER_INDUSTRY_IS_FURNITURE'] = \
                (df['NAME_SELLER_INDUSTRY']=='Furniture').astype('int64')
        df['NAME_SELLER_INDUSTRY_IS_CONSTRUCTION'] = \
                (df['NAME_SELLER_INDUSTRY']=='Construction').astype('int64')
        df['NAME_SELLER_INDUSTRY_IS_JEWELRY'] = \
                (df['NAME_SELLER_INDUSTRY']=='Jewelry').astype('int64')
        df['NAME_SELLER_INDUSTRY_IS_TECHNOLOGY'] = \
                (df['NAME_SELLER_INDUSTRY']=='Auto technology').astype('int64')
        df['NAME_SELLER_INDUSTRY_IS_MLM'] = \
                (df['NAME_SELLER_INDUSTRY']=='MLM partners').astype('int64')
        df['NAME_SELLER_INDUSTRY_IS_TOURISM'] = \
                (df['NAME_SELLER_INDUSTRY']=='Tourism').astype('int64')
        
        df['NAME_YIELD_GROUP_IS_HIGH'] = \
                (df['NAME_YIELD_GROUP']=='high').astype('int64')
        df['NAME_YIELD_GROUP_IS_MIDDLE'] = \
                (df['NAME_YIELD_GROUP']=='middle').astype('int64')
        df['NAME_YIELD_GROUP_IS_LOWNORMAL'] = \
                (df['NAME_YIELD_GROUP']=='low_normal').astype('int64')
        df['NAME_YIELD_GROUP_IS_LOWACTION'] = \
                (df['NAME_YIELD_GROUP']=='low_action').astype('int64')
        
        df['AMT_CREDIT_LT_APPLICATION'] = \
            (df['AMT_CREDIT'] < df['AMT_APPLICATION']).astype('int64')
        df['DAYS_LAST_DUE_DUE'] = \
            (df['DAYS_LAST_DUE_1ST_VERSION'] < df['DAYS_LAST_DUE']).astype('int64')
        df['DAYS_LAST_DUE_TERMINATION'] = \
            (df['DAYS_LAST_DUE'] < df['DAYS_TERMINATION']).astype('int64')
        df['RATE_HAS_PRIVILEGED'] = \
            (df['RATE_INTEREST_PRIVILEGED'].notna()).astype('int64')
        # num_feature
        df['AMT_CREDIT_APPLICATION_RATIO'] = \
            df['AMT_CREDIT']/df['AMT_APPLICATION']
        df['AMT_RATE_OF_CREDIT'] = \
            df['AMT_ANNUITY']*df['CNT_PAYMENT']/df['AMT_CREDIT']
        return df
    
    def fit(self, df):
        df_groupby_sk_id_curr = df.groupby(['SK_ID_CURR'])
        df_feature = df_groupby_sk_id_curr.size().reset_index().\
                rename(columns = {0:'PREV_NUMS_OF_APPLICATION'})
                
        count_feature = ['NAME_CONTRACT_TYPE','NAME_CASH_LOAN_PURPOSE',
                         'NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE','NAME_TYPE_SUITE',
                         'NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO',
                         'NAME_PRODUCT_TYPE','CHANNEL_TYPE','SELLERPLACE_AREA',
                         'NAME_SELLER_INDUSTRY','CNT_PAYMENT','NAME_YIELD_GROUP',
                         'PRODUCT_COMBINATION']
        df_feature_count = df_groupby_sk_id_curr[count_feature].agg('nunique').\
                reset_index().\
                rename(columns = dict(zip(count_feature,
                                          ['PREV_%s_NUNIQUE'%name for name in count_feature])))
        df_feature = df_feature.merge(df_feature_count, on = 'SK_ID_CURR', how = 'left')
        
        cat_feature = list(df.columns)[37:110] + \
                ['NFLAG_LAST_APPL_IN_DAY','NFLAG_INSURED_ON_APPROVAL']
        df_feature_sum = df_groupby_sk_id_curr[cat_feature].agg('sum').\
                reset_index().\
                rename(columns = dict(zip(cat_feature,
                                          ['PREV_%s_SUM'%name for name in cat_feature])))
        df_feature = df_feature.merge(df_feature_sum, on = 'SK_ID_CURR', how = 'left')
        
        df_feature_avg = df_groupby_sk_id_curr[cat_feature].agg('mean').\
                reset_index().\
                rename(columns = dict(zip(cat_feature,
                                          ['PREV_%s_MEAN'%name for name in cat_feature])))
        df_feature = df_feature.merge(df_feature_avg, on = 'SK_ID_CURR', how = 'left')
        
        num_feature = ['AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_DOWN_PAYMENT',
                       'AMT_GOODS_PRICE','HOUR_APPR_PROCESS_START','RATE_DOWN_PAYMENT',
                       'DAYS_DECISION','CNT_PAYMENT','DAYS_FIRST_DRAWING',
                       'DAYS_FIRST_DUE','DAYS_LAST_DUE','DAYS_TERMINATION',
                       'AMT_CREDIT_APPLICATION_RATIO','AMT_RATE_OF_CREDIT']
        df_feature_num = df_groupby_sk_id_curr[num_feature].\
                agg(['sum','mean','median','max','min','std']).reset_index()
        df_feature_num.columns = ['SK_ID_CURR']+ ['PREV_%s_%s' % (var,fun) \
                                   for fun in ['sum','mean','median','max','min','std'] 
                                   for var in num_feature]
        df_feature = df_feature.merge(df_feature_num, on = 'SK_ID_CURR', how = 'left')
        self.data = df_feature
        return df_feature
    
    def feature_extract(self, df):
        df = self.transform(df)
        df = self.fit(df)
        return df