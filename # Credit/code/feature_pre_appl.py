# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 10:12:22 2018

@author: Franc
"""
import pandas as pd
from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()

class ApplPreFeatures(BaseTransformer):
    def __init__(self, na_value, **kwargs):
        self.na_value = na_value
        self.df_feature = None
        
    def transform(self, df):
        df = df.replace({'XNA':self.na_value, 365243:self.na_value}).\
                sort_values(['SK_ID_CURR','SK_ID_PREV'])
        df['NAME_CONTRACT_IS_CASH'] = \
            (df['NAME_CONTRACT_TYPE'] == 'Cash loans').astype('int64')
        df['NAME_CONTRACT_IS_CONSUMER'] = \
            (df['NAME_CONTRACT_TYPE'] == 'Consumer loans').astype('int64')
        df['NAME_CONTRACT_IS_REVOLVING'] = \
            (df['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype('int64')
        df['AMT_CREDIT_LT_APPLICATION'] = \
            (df['AMT_CREDIT'] < df['AMT_APPLICATION']).astype('int64')
        df['AMT_CREDIT_APPLICATION_RATIO'] = \
            df['AMT_CREDIT']/df['AMT_APPLICATION']
        df['AMT_RATE_OF_CREDIT'] = \
            df['AMT_ANNUITY']*df['CNT_PAYMENT']/df['AMT_CREDIT']
        df['DATE_WEEKDAY_IS_WEEKEND'] = \
            (df['WEEKDAY_APPR_PROCESS_START'].\
                 isin(['SATURDAY','SUNDAY','MONDAY'])).astype('int64')
        df['DATE_HOUR_IS_DAY'] = \
            ((df['HOUR_APPR_PROCESS_START']>7)&
             (df['HOUR_APPR_PROCESS_START']<20)).astype('int64')
        df['NAME_SELLER_INDUSTRY_CONSUMING'] = \
            (df['NAME_SELLER_INDUSTRY'].\
                 isin(['Connectivity','Consumer electronics','Clothing',
                       'Tourism','MLM Partners'])).astype('int64')
        df['NAME_YIELD_GROUP_HIGH'] = \
            (df['NAME_YIELD_GROUP'] == 'high').astype('int64')
        
        df['DAYS_LAST_DUE_DUE'] = \
            (df['DAYS_LAST_DUE_1ST_VERSION'] < df['DAYS_LAST_DUE']).astype('int64')
        df['DAYS_LAST_DUE_TERMINATION'] = \
            (df['DAYS_LAST_DUE'] < df['DAYS_TERMINATION']).astype('int64')
        
        df['RATE_HAS_PRIVILEGED'] = \
            (df['RATE_INTEREST_PRIVILEGED'].notna()).astype('int64')
        df['NAME_LOAN_PURPOSE_IS_URGENT'] = \
            (df['NAME_CASH_LOAN_PURPOSE'].\
                 isin(['Payments on other loans','Urgent needs',
                       'Refusal to name the goal'])).astype('int64')
        df['NAME_LOAN_PURPOSE_IS_REGULAR'] = \
            (df['NAME_CASH_LOAN_PURPOSE'].\
                 isin(['Everyday expenses', 'Hobby', 'Gasification / water supply',
                       'Repairs','Car Repairs','Journey','Medichine',
                       'Education','Gasification / water supply'])).astype('int64')
        df['NAME_LOAN_PURPOSE_IS_BUSINESS'] = \
            (df['NAME_CASH_LOAN_PURPOSE'].\
                 isin(['Business development',
                       'Money for a third person'])).astype('int64')
        df['NAME_CONTRACT_IS_UNUSED_CANCEL'] = \
            (df['NAME_CONTRACT_STATUS'].\
                 isin(['Cancelled','Unused offer'])).astype('int64')
        df['NAME_CONTRACT_IS_REFUSED'] = \
            (df['NAME_CONTRACT_STATUS']=='Refused').astype('int64')
        
        df['NAME_PAYMENT_FROM_BANK'] = \
            (df['NAME_PAYMENT_TYPE']=='Cash through the bank').astype('int64')
        df['NAME_REJECT_VERIF_LIMIT'] = \
            (df['CODE_REJECT_REASON'].isin(['VERIF','LIMIT','HC'])).astype('int64')
        df['NAME_REJECT_SCOFR_SCO'] = \
            (df['CODE_REJECT_REASON'].isin(['SCO','SCOFR','SYSTEM'])).astype('int64')
        df['NAME_TYPE_SUITE_FAMILY'] = \
            (df['NAME_TYPE_SUITE'].isin(['Family','Children'])).astype('int64')
        df['NAME_TYPE_SUITE_OTHER'] = \
            (df['NAME_TYPE_SUITE'].isin(['Spouse, partner','Other_B',
                 'Other_A', 'Group of people'])).astype('int64')
        df['NAME_PRODUCT_TYPE_XSHELL'] = \
            (df['NAME_PRODUCT_TYPE']=='x-shell').astype('int64')
            
        df['NAME_CHANNEL_DEALER'] = \
            (df['CHANNEL_TYPE'].\
                 isin(['Channel of corporate sales','Car dealer'])).astype('int64')
        df['NAME_CHANNEL_CENTER'] = \
            (df['CHANNEL_TYPE'].\
                 isin(['Contact','Credit and cash offices',
                       'AP+ (Cash loan)'])).astype('int64')
        df['NAME_SELLERAREA'] = \
            (df['SELLERPLACE_AREA'] == -1).astype('int64')
        return df
    
    def fit(self, df):
        df_feature = pd.DataFrame({'SK_ID_CURR':df['SK_ID_CURR'].unique()})
        
        df_groupby_curr = df.groupby(['SK_ID_CURR'])
        df_feature['POS_num_of_application'] = \
            df_groupby_curr.size()
        df_feature['POS_num_of_reject_type'] = \
            df_groupby_curr['CODE_REJECT_REASON'].nunique()
        df_feature['POS_num_of_goods_type'] = \
            df_groupby_curr['NAME_GOODS_CATEGORY'].nunique()
        df_feature['POS_num_of_loan_purpose_type'] = \
            df_groupby_curr['NAME_CASH_LOAN_PURPOSE'].nunique()
        df_feature['POS_num_of_suite_type'] = \
            df_groupby_curr['NAME_TYPE_SUITE'].nunique()
        df_feature['POS_num_of_channel_type'] = \
            df_groupby_curr['CHANNEL_TYPE'].nunique()
        var_sumlist = ['NAME_CONTRACT_IS_CASH','NAME_CONTRACT_IS_CONSUMER',
                       'NAME_CONTRACT_IS_REVOLVING','AMT_CREDIT_LT_APPLICATION',
                       'DATE_WEEKDAY_IS_WEEKEND','DATE_HOUR_IS_DAY',
                       'NAME_SELLER_INDUSTRY_CONSUMING','NAME_YIELD_GROUP_HIGH',
                       'DAYS_LAST_DUE_DUE','DAYS_LAST_DUE_TERMINATION',
                       'RATE_HAS_PRIVILEGED','NAME_LOAN_PURPOSE_IS_URGENT',
                       'NAME_LOAN_PURPOSE_IS_REGULAR','NAME_LOAN_PURPOSE_IS_BUSINESS',
                       'NAME_CONTRACT_IS_UNUSED_CANCEL','NAME_CONTRACT_IS_REFUSED',
                       'NAME_PAYMENT_FROM_BANK','NAME_REJECT_VERIF_LIMIT',
                       'NAME_REJECT_SCOFR_SCO','NAME_TYPE_SUITE_FAMILY',
                       'NAME_TYPE_SUITE_OTHER','NAME_PRODUCT_TYPE_XSHELL',
                       'NAME_CHANNEL_DEALER','NAME_CHANNEL_CENTER','NAME_SELLERAREA'
                       ]
        var_meanlist = ['NAME_CONTRACT_IS_CASH','NAME_CONTRACT_IS_CONSUMER',
                        'NAME_CONTRACT_IS_REVOLVING','AMT_CREDIT_LT_APPLICATION',
                        'AMT_CREDIT_APPLICATION_RATIO','AMT_RATE_OF_CREDIT',
                        'DATE_WEEKDAY_IS_WEEKEND','DATE_HOUR_IS_DAY',
                        'NAME_SELLER_INDUSTRY_CONSUMING','NAME_YIELD_GROUP_HIGH',
                        'DAYS_LAST_DUE_DUE','DAYS_LAST_DUE_TERMINATION',
                        'RATE_HAS_PRIVILEGED','NAME_LOAN_PURPOSE_IS_URGENT',
                        'NAME_LOAN_PURPOSE_IS_REGULAR','NAME_LOAN_PURPOSE_IS_BUSINESS',
                        'NAME_CONTRACT_IS_UNUSED_CANCEL','NAME_CONTRACT_IS_REFUSED',
                        'NAME_PAYMENT_FROM_BANK','NAME_REJECT_VERIF_LIMIT',
                        'NAME_REJECT_SCOFR_SCO','NAME_TYPE_SUITE_FAMILY',
                        'NAME_TYPE_SUITE_OTHER','NAME_PRODUCT_TYPE_XSHELL',
                        'NAME_CHANNEL_DEALER','NAME_CHANNEL_CENTER','NAME_SELLERAREA',
                        'AMT_ANNUITY','AMT_APPLICATION','AMT_DOWN_PAYMENT',
                        'DAYS_DECISION']
        var_mlist = ['AMT_CREDIT_APPLICATION_RATIO','AMT_RATE_OF_CREDIT',
                    'NAME_SELLER_INDUSTRY_CONSUMING',
                    'AMT_ANNUITY','AMT_APPLICATION','AMT_DOWN_PAYMENT',
                    'RATE_INTEREST_PRIMARY','DAYS_DECISION']
        for var in var_sumlist:
            df_feature['POS_{}_sum'.format(var.lower())] = \
                df_groupby_curr[var].sum()
        for var in var_meanlist:
            df_feature['POS_{}_avg'.format(var.lower())] = \
                df_groupby_curr[var].mean()
        for var in var_mlist:
            df_feature['POS_{}_max'.format(var.lower())] = \
                df_groupby_curr[var].max()
        for var in var_mlist:
            df_feature['POS_{}_min'.format(var.lower())] = \
                df_groupby_curr[var].min()
        return df_feature
    def outcome(self,df):
        self.df_feature = self.transform(df).pipe(self.fit)
        return self.df_feature

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