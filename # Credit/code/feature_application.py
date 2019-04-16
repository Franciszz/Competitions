# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:20:20 2018

@author: Franc
"""

import pandas as pd
from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()

class ApplicationFeatures(BaseTransformer):
    def __init__(self, na_value, **kwargs):
        self.na_value = na_value
        self.df = None
        self.categorical_feature = \
            ['NAME_FAMILY_STATUS','NAME_TYPE_SUITE','NAME_INCOME_TYPE',
             'NAME_EDUCATION_TYPE','NAME_HOUSING_TYPE',
             'BASE_OCCUPATION_TYPE','BASE_ORGANIZATION_TYPE',
             'ASSET_WALLSMATERIAL_MODE','ASSET_HOUSETYPE_MODE']
        
    def transform(self, df):
        df['NAME_CONTRACT_TYPE'].\
            replace({'Cash loans':0, 'Revolving loans':1},inplace=True)
            
        df['CODE_GENDER'].\
            replace({'F':0,'M':1, 'XNA':self.na_value},inplace=True)
        df['FLAG_OWN_CAR'].\
            replace({'N':0, 'Y':1},inplace=True)
        df['FLAG_OWN_REALTY'].\
            replace({'N':0, 'Y':1},inplace=True)
        df['EMERGENCYSTATE_MODE'].\
            replace({'No':0, 'Yes':1}, inplace=True)
        df['WEEKDAY_APPR_PROCESS_START'].\
            replace({'TUESDAY':0, 'WEDNESDAY':0,'THURSDAY':0,'FRIDAY':1, 
                     'MONDAY':1,'SATURDAY':1,'SUNDAY':1}, inplace=True)
        df['FONDKAPREMONT_MODE'].\
            replace({'reg oper account':0,'reg oper spec account':0, 
                     'not specified':0, 'org spec account':1},inplace=True)
        df['NAME_FAMILY_STATUS'].\
            replace('Unknown', self.na_value, inplace=True)
            
        df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].\
            replace({'Children':0, 'Family':0, 
                     'Group of people':1, 'Other_A':1, 'Spouse, partner':1,
                     'Other_B':2, 'Unaccompanied':3}).astype('category')
    
        df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].\
            replace({'Commercial associate':1, 'Working':1, 'State servant':1,
                     'Unemployed':2, 'Maternity leave':2, 'Pensioner':2,
                     'Businessman':0, 'Student':0}).astype('category')
    
        df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].\
            replace({'Secondary / secondary special':0, 
                     'Incomplete higher':1, 'Lower secondary':0,
                     'Higher education':2, 'Academic degree':3}).astype('category')
    
        df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].\
            replace({'Single / not married':0, 
                     'Widow':1, 'Separated':1,
                     'Married':2, 'Civil marriage':2, 
                     'Unknown':self.na_value}).astype('category')
    
        df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].\
            replace({'House / apartment':0, 
                     'Rented apartment':1, 'With parents':1,
                     'Municipal apartment':2, 'Co-op apartment':2,
                     'Office apartment':3}).astype('category')
    
        df['HOUSETYPE_MODE'] = df['HOUSETYPE_MODE'].\
            replace({'block of flats':0, 'terraced house':1,
                     'specific housing':2}).astype('category')
    
        df['WALLSMATERIAL_MODE'] = df['WALLSMATERIAL_MODE'].\
            replace({'Wooden':0, 'Others':0, 'Block':1, 'Mixed':1,
                     'Panel':2, 'Stone, brick':2, 
                     'Monolithic':3}).astype('category')
    
        df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].\
            replace({'Accountants':0, 'Secretaries':0, 'Medicine staff':0,
                     'High skill tech staff':0, 'IT staff':0, 'Managers':0,
                     'Private service staff':0, 'Core staff':0, 'HR staff':0,
                     'Cleaning staff':1, 'Cooking staff':1, 'Drivers':1,
                     'Laborers':1, 'Realty agents':1, 'Sales staff':1,
                     'Security staff':1, 'Waiters/barmen staff':1,
                     'Low-skill Laborers':2}).astype('category')

        df['ORGANIZATION_TYPE'] = df['ORGANIZATION_TYPE'].\
            replace({'Industry: type 12':0, 'Trade: type 4':0,
                     'Police':1, 'Security Ministries':1, 
                     'Trade: type 6':1, 'Transport: type 1':1,
                     'Trade: type 5':2, 'University':2,
                     'Bank':2, 'Culture':2, 'Electricity':2, 'Hotel':2, 
                     'Industry: type 10':2, 'Insurance':2, 'Medicine':2, 
                     'Military':2, 'Religion':2, 'School':2, 'Services':2,
                     'Business Entity Type 3':3, 'Emergency':3, 'Other':3,
                     'Industry: type 7':3, 'Industry: type 9':3, 'Mobile':3,
                     'Business Entity Type 2':3, 'Government':3, 'Housing':3,
                     'Business Entity Type 1':3, 'Advertising':3, 'Postal':3,
                     'Industry: type 11':3, 'Industry: type 2':3, 'Kindergarten':3,
                     'Industry: type 5':3, 'Industry: type 6':3, 'Legal Services':3,
                     'Security':3, 'Telecom':3, 'Trade: type 1':3, 'Trade: type 2':3,
                     'Trade: type 7':3, 'Transport: type 2':3, 'Transport: type 4':3,
                     'Agriculture':4, 'Cleaning':4, 'Restaurant':4, 'Realtor':4,
                     'Self-employed':4, 'Trade: type 3':4, 'Transport: type 3':4,
                     'Construction':4, 'Industry: type 1':4, 'Industry: type 13':4,
                     'Industry: type 4':4, 'Industry: type 8':4, 'Industry: type 3':4,
                     'XNA':self.na_value}).astype('category')
        df.replace(365243, self.na_value, inplace = True)
        return df
    
    def shrink(self, df):
        var_docs = ['FLAG_DOCUMENT_%s'%i for i in range(2,22)]
        
        var_docs_etc = ['FLAG_DOCUMENT_%s'% i for i in [3,6,13,16]]
        
        var_aparts = [name + '_' + measure 
                      for name in
                      ['BASEMENTAREA','ELEVATORS','ENTRANCES','FLOORSMAX','FLOORSMIN',
                       'LANDAREA','LIVINGAPARTMENTS','LIVINGAREA','NONLIVINGAPARTMENTS',
                       'COMMONAREA','APARTMENTS','YEARS_BEGINEXPLUATATION',
                       'NONLIVINGAREA','YEARS_BUILD'] 
                      for measure in 
                      ['AVG','MEDI']]
        
        var_phone = ['FLAG_%s'%name for name in 
                     ['CONT_MOBILE','EMAIL','MOBIL','PHONE','WORK_PHONE']]
        
        var_query = ['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',
                     'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
                     'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR',]
        
        var_place = ['LIVE_CITY_NOT_WORK_CITY','LIVE_REGION_NOT_WORK_REGION',
                     'REG_CITY_NOT_WORK_CITY','REG_REGION_NOT_WORK_REGION',
                     'REG_CITY_NOT_LIVE_CITY','REG_REGION_NOT_LIVE_REGION']
        
        df['FLAG_DOCUMENT'] = df[var_docs].apply('sum',axis=1)
        
        df['FLAG_DOCUMENT_EMP'] = \
            df[var_docs_etc].rmul([4,2,1,1],axis=1).apply('sum',axis=1)
        
        df['FLAG_PHONE'] = \
            df[var_phone].rmul([1,1,1,2,2],axis=1).apply('sum',axis=1)
            
        df['FLAG_PLACE'] = \
            df[var_place].rmul([3,1,6,1,5,1],axis=1).apply('sum',axis=1)
        
        df['AMT_QUERY'] = \
            df[var_query].rmul([1,4,4,2,2,1],axis=1).apply('sum',axis=1)
        
        df = df.drop(var_docs + var_phone + var_aparts + var_place + var_query,
                axis=1).\
            rename(columns = {
                    'CODE_GENDER':'BASE_CODE_GENDER',
                    'CNT_CHILDREN':'BASE_CNT_CHILDREN',
                    'CNT_FAM_MEMBERS':'BASE_CNT_FAM_MEMBERS',
                    'OCCUPATION_TYPE':'BASE_OCCUPATION_TYPE',
                    'ORGANIZATION_TYPE':'BASE_ORGANIZATION_TYPE',
                    'DEF_30_CNT_SOCIAL_CIRCLE':'BASE_DEF_30_CNT_SOCIAL_CIRCLE',
                    'DEF_60_CNT_SOCIAL_CIRCLE':'BASE_DEF_60_CNT_SOCIAL_CIRCLE',
                    'OBS_30_CNT_SOCIAL_CIRCLE':'BASE_OBS_30_CNT_SOCIAL_CIRCLE',
                    'OBS_60_CNT_SOCIAL_CIRCLE':'BASE_OBS_60_CNT_SOCIAL_CIRCLE',
                    
                    'REGION_POPULATION_RELATIVE':'EXT_REGION_POPULATION_RELATIVE',
                    'REGION_RATING_CLIENT':'EXT_REGION_RATING_CLIENT',
                    'REGION_RATING_CLIENT_W_CITY':'EXT_REGION_RATING_CLIENT_W_CITY',  
                  
                    'OWN_CAR_AGE':'DAYS_OWN_CAR_AGE',
                    
                    'HOUR_APPR_PROCESS_START':'NAME_HOUR_APPR_PROCESS_START',
                    'WEEKDAY_APPR_PROCESS_START':'NAME_WEEKDAY_APPR_PROCESS_START',
                    
                    'APARTMENTS_MODE':'ASSET_APARTMENTS_MODE',
                    'BASEMENTAREA_MODE':'ASSET_BASEMENTAREA_MODE',
                    'COMMONAREA_MODE':'ASSET_COMMONAREA_MODE',
                    'ELEVATORS_MODE':'ASSET_ELEVATORS_MODE',
                    'EMERGENCYSTATE_MODE':'ASSET_EMERGENCYSTATE_MODE',
                    'ENTRANCES_MODE':'ASSET_ENTRANCES_MODE',
                    'WALLSMATERIAL_MODE':'ASSET_WALLSMATERIAL_MODE',
                    'HOUSETYPE_MODE':'ASSET_HOUSETYPE_MODE',
                    'FLOORSMAX_MODE':'ASSET_FLOORSMAX_MODE',
                    'FLOORSMIN_MODE':'ASSET_FLOORSMIN_MODE',
                    'FONDKAPREMONT_MODE':'ASSET_FONDKAPREMONT_MODE',
                    'LANDAREA_MODE':'ASSET_LANDAREA_MODE',
                    'LIVINGAPARTMENTS_MODE':'ASSET_LIVINGAPARTMENTS_MODE',
                    'LIVINGAREA_MODE':'ASSET_LIVINGAREA_MODE',
                    'NONLIVINGAPARTMENTS_MODE':'ASSET_NONLIVINGAPARTMENTS_MODE',
                    'NONLIVINGAREA_MODE':'ASSET_NONLIVINGAREA_MODE',
                    'TOTALAREA_MODE':'ASSET_TOTALAREA_MODE',
                    'YEARS_BUILD_MODE':'ASSET_YEARS_BUILD_MODE',
                    'YEARS_BEGINEXPLUATATION_MODE':'ASSET_YEARS_BEGINEXPLUATATION_MODE'
                    })
        return df
    
    def fit(self, df):
        
        df['BASE_child_income_ratio'] = \
            df['BASE_CNT_CHILDREN']/df['AMT_INCOME_TOTAL']
        df['BASE_income_members_ratio'] = \
            df['BASE_CNT_FAM_MEMBERS']/df['AMT_INCOME_TOTAL']
        df['BASE_child_area_ratio'] = \
            df['BASE_CNT_CHILDREN']/df['ASSET_LIVINGAREA_MODE']
        df['BASE_members_area_ratio'] = \
            df['BASE_CNT_FAM_MEMBERS']/df['ASSET_LIVINGAREA_MODE']
        df['BASE_child_common_ratio'] = \
            df['BASE_CNT_CHILDREN']/df['ASSET_COMMONAREA_MODE']
        df['BASE_members_common_ratio'] = \
            df['BASE_CNT_FAM_MEMBERS']/df['ASSET_COMMONAREA_MODE']
        df['BASE_family_child_ratio'] = \
            df['BASE_CNT_CHILDREN']/df['BASE_CNT_FAM_MEMBERS']
        df['BASE_def_obs_ratio_30'] = \
            df['BASE_DEF_30_CNT_SOCIAL_CIRCLE']/df['BASE_OBS_30_CNT_SOCIAL_CIRCLE']
        df['BASE_def_obs_ratio_60'] = \
            df['BASE_DEF_60_CNT_SOCIAL_CIRCLE']/df['BASE_OBS_60_CNT_SOCIAL_CIRCLE']
        
        df['AMT_annuity_income_ratio'] = \
            df['AMT_ANNUITY']/df['AMT_INCOME_TOTAL']
        df['AMT_annuity_credit_ratio'] = \
            df['AMT_ANNUITY']/df['AMT_CREDIT']
        df['AMT_annuity_price_ratio'] = \
            df['AMT_ANNUITY']/df['AMT_GOODS_PRICE']
        df['AMT_credit_price_ratio'] = \
            df['AMT_CREDIT']/df['AMT_GOODS_PRICE']
        df['AMT_credit_income_ratio'] = \
            df['AMT_CREDIT']/df['AMT_INCOME_TOTAL']
        df['AMT_price_income_ratio'] = \
            df['AMT_GOODS_PRICE']/df['AMT_INCOME_TOTAL']
        
        df['DAYS_employed_birth_ratio'] = \
            df['DAYS_EMPLOYED']/df['DAYS_BIRTH']
        df['DAYS_phonechange_birth_ratio'] = \
            df['DAYS_LAST_PHONE_CHANGE']/df['DAYS_BIRTH']
        df['DAYS_id_birth_ratio'] = \
            df['DAYS_ID_PUBLISH']/df['DAYS_BIRTH']
        df['DAYS_regis_birth_ratio'] = \
            df['DAYS_REGISTRATION']/df['DAYS_BIRTH']
        df['DAYS_car_birth_ratio'] = \
            df['DAYS_OWN_CAR_AGE']/df['DAYS_BIRTH']
        df['DAYS_id_employed_ratio'] = \
            df['DAYS_ID_PUBLISH']/df['DAYS_EMPLOYED']
        df['DAYS_regis_employed_ratio'] = \
            df['DAYS_REGISTRATION']/df['DAYS_EMPLOYED']
        df['DAYS_car_employed_ratio'] = \
            df['DAYS_OWN_CAR_AGE']/df['DAYS_EMPLOYED']
        df['DAYS_car_regis_ratio'] = \
            df['DAYS_OWN_CAR_AGE']/df['DAYS_REGISTRATION']
        
        df['EXT_SOURCE_MAX'] = \
            df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].max(axis=1)
        df['EXT_SOURCE_MIN'] = \
            df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].min(axis=1)
        df['EXT_SOURCE_MEDIAN'] = \
            df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].median(axis=1)
        
        df['ASSET_COMMON_TOTAL_AREA'] = \
            df['ASSET_COMMONAREA_MODE']/df['ASSET_TOTALAREA_MODE']
        df['ASSET_LIVING_TOTAL_AREA'] = \
            df['ASSET_LIVINGAREA_MODE']/df['ASSET_TOTALAREA_MODE']
        df['ASSET_NONLIVING_TOTAL_AREA'] = \
            df['ASSET_NONLIVINGAREA_MODE']/df['ASSET_TOTALAREA_MODE']
        df['ASSET_LANDAREA_TOTAL_AREA'] = \
            df['ASSET_LANDAREA_MODE']/df['ASSET_TOTALAREA_MODE']
        df['ASSET_BASEMENTAREA_TOTAL_AREA'] = \
            df['ASSET_BASEMENTAREA_MODE']/df['ASSET_TOTALAREA_MODE']
        df['ASSET_NONLIVING_LIVING_AREA'] = \
            df['ASSET_NONLIVINGAREA_MODE']/df['ASSET_LIVINGAREA_MODE']
        df['ASSET_NONLIVIN_LIVING_APARTS'] = \
            df['ASSET_NONLIVINGAPARTMENTS_MODE']/df['ASSET_LIVINGAPARTMENTS_MODE']
        return df

    def outcome(self, df):
        self.df = self.transform(df).pipe(self.shrink).pipe(self.fit)
        return self.df
        
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
            df.groupby('SK_ID_BUREAU')[['BUREAU_IS_DPD','BUREAU_DPD_STATUS']].agg('max')
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