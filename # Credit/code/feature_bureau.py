# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 08:56:04 2018

@author: Franc
"""
import pandas as pd
from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()

class BureauFeatures(BaseTransformer):
    def __init__(self, fill_value, df_bal, **kwargs):
        self.fill_value = fill_value
        self.df_feature = None
        self.df_bal = df_bal
    
    def transform(self, df):
        df['BUREAU_IS_DPD'] = \
            (df['STATUS'].isin(['1','2','3','4','5'])).astype('int64')
        df['BUREAU_DPD_STATUS'] = df['STATUS'].\
            apply(lambda x: int(x) if x in ['1','2','3','4','5'] else 0)
        df_dpd = \
            df.groupby('SK_ID_BUREAU')[['BUREAU_IS_DPD',
                      'BUREAU_DPD_STATUS']].agg('max')
        return df_dpd
        
    def fit(self, df):
        df = df.drop('CREDIT_CURRENCY',axis=1).\
            sort_values(['SK_ID_CURR']).reset_index(drop=True)
        
        df_dpd = self.transform(self.df_bal)
        df = df.join(df_dpd,on='SK_ID_BUREAU',how='left')
        df['BUREAU_CREDIT_ACTIVE_IS_ACTIVE'] = \
            (df['CREDIT_ACTIVE'] == 'Active').astype('int64')
        df['BUREAU_CREDIT_ACTIVE_IS_BAD'] = \
            (df['CREDIT_ACTIVE'] == 'Bad debt').astype('int64')
        df['BUREAU_CREDIT_ACTIVE_IS_CLOSED'] = \
            (df['CREDIT_ACTIVE'] == 'Closed').astype('int64')
        df['BUREAU_CREDIT_ACTIVE_IS_SOLD'] = \
            (df['CREDIT_ACTIVE'] == 'Sold').astype('int64')
        
        
        df['BUREAU_CREDIT_TYPE_IS_REVOLVING'] = \
            (df['CREDIT_TYPE'] == 'Credit card').astype('int64')
        df['BUREAU_CREDIT_TYPE_IS_MICRO'] = \
            (df['CREDIT_TYPE'] == 'Microloan').astype('int64')
        df['BUREAU_CREDIT_TYPE_IS_CASH'] = \
            (df['CREDIT_TYPE'] == 'Cash loan (non-earmarked)').astype('int64')
        df['BUREAU_CREDIT_TYPE_IS_CAPITAL'] = \
            (df['CREDIT_TYPE'].isin(['Loan for business development',
                                     'Loan for working capital replenishment',
                                     'Loan for the purchase of equipment',
                                     'Loan for purchase of shares (margin lending)',
                                     'Mobile operator loan','Mortgage',
                                     'Interbank credit'])).astype('int64')
        df['BUREAU_CREDIT_TYPE_IS_INDU'] = \
            (df['CREDIT_TYPE'].isin(['Loan for purchase of shares (margin lending)',
                                     'Car loan','Real estate loan'])).astype('int64')
        
        df['BUREAU_AMT_RATIO_DEBT_SUM'] = \
            df['AMT_CREDIT_SUM_DEBT']/df['AMT_CREDIT_SUM']
        df['BUREAU_AMT_RATIO_OVERDUE_SUM'] = \
            df['AMT_CREDIT_SUM_OVERDUE']/df['AMT_CREDIT_SUM']
        df['BUREAU_AMT_RATIO_MAXOVERDUE_SUM'] = \
            df['AMT_CREDIT_MAX_OVERDUE']/df['AMT_CREDIT_SUM']
        return df
        
    def feature_extract(self, df):
        df_feature = pd.DataFrame({'SK_ID_CURR':df['SK_ID_CURR'].unique()})
        df_groupby_id = df.groupby(by=['SK_ID_CURR'])
        
        df_feature['BUREAU_nums_card_total'] = \
            df_groupby_id.size()
        df_feature['BUREAU_nums_card_active'] = \
            df_groupby_id['BUREAU_CREDIT_ACTIVE_IS_ACTIVE'].sum()
        df_feature['BUREAU_nums_card_bad'] = \
            df_groupby_id['BUREAU_CREDIT_ACTIVE_IS_BAD'].sum()
        df_feature['BUREAU_nums_card_closed'] = \
            df_groupby_id['BUREAU_CREDIT_ACTIVE_IS_CLOSED'].sum()
        df_feature['BUREAU_nums_card_sold'] = \
            df_groupby_id['BUREAU_CREDIT_ACTIVE_IS_SOLD'].sum()
        df_feature['BUREAU_dpd_card_nums'] = \
            df_groupby_id['BUREAU_IS_DPD'].sum()
        df_feature['BUREAU_dpd_card_max'] = \
            df_groupby_id['BUREAU_DPD_STATUS'].max()
        df_feature['BUREAU_dpd_card_avg'] = \
            df_groupby_id['BUREAU_DPD_STATUS'].mean()
        
        df_feature['BUREAU_type_card_total'] = \
            df_groupby_id['CREDIT_TYPE'].nunique()
        df_feature['BUREAU_type_card_revolving'] = \
            df_groupby_id['BUREAU_CREDIT_TYPE_IS_REVOLVING'].sum()
        df_feature['BUREAU_type_card_micro'] = \
            df_groupby_id['BUREAU_CREDIT_TYPE_IS_MICRO'].sum()
        df_feature['BUREAU_type_card_capital'] = \
            df_groupby_id['BUREAU_CREDIT_TYPE_IS_CAPITAL'].sum()
        df_feature['BUREAU_type_card_indu'] = \
            df_groupby_id['BUREAU_CREDIT_TYPE_IS_INDU'].sum()
        for var in ['AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_LIMIT',
                    'AMT_CREDIT_SUM_OVERDUE','AMT_CREDIT_MAX_OVERDUE','AMT_ANNUITY',
                    'BUREAU_AMT_RATIO_DEBT_SUM','BUREAU_AMT_RATIO_OVERDUE_SUM',
                    'BUREAU_AMT_RATIO_MAXOVERDUE_SUM', 'CNT_CREDIT_PROLONG']:
            for funcname in ['sum','median','max','min']:
                df_feature['Bureau_{}_{}'.format(var.lower(),funcname)] = \
                    df_groupby_id[var].agg(funcname)
        for var in ['DAYS_CREDIT', 'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT_ENDDATE',
                    'DAYS_ENDDATE_FACT','DAYS_CREDIT_UPDATE']:
            df_feature['Bureau_{}_avg'.format(var.lower())] = \
                    df_groupby_id[var].agg('mean')
        return df_feature
    def outcome(self,df):
        self.df_feature = self.fit(df).pipe(self.feature_extract)
        return self.df_feature
