# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:13:51 2018

@author: Franc
"""

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
             'ASSET_WALLSMATERIAL_MODE','ASSET_HOUSETYPE_MODE',
             'WEEKDAY_APPR_PROCESS_START','FONDKAPREMONT_MODE']
        
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
        df['WEEKDAY_APPR_PROCESS_START'] = df['WEEKDAY_APPR_PROCESS_START'].\
            replace({'TUESDAY':1, 'WEDNESDAY':2,'THURSDAY':3,'FRIDAY':4, 
                     'MONDAY':5,'SATURDAY':6,'SUNDAY':7}).astype('category')
        df['FONDKAPREMONT_MODE'] = df['FONDKAPREMONT_MODE'].\
            replace({'reg oper account':0,'reg oper spec account':1, 
                     'not specified':2, 'org spec account':3}).astype('category')
            
        df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].\
            replace({'Children':0, 'Family':1, 
                     'Group of people':2, 'Other_A':3, 'Spouse, partner':4,
                     'Other_B':5, 'Unaccompanied':6}).astype('category')
    
        df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].\
            replace({'Commercial associate':1, 'Working':6, 'State servant':5,
                     'Unemployed':2, 'Maternity leave':3, 'Pensioner':4,
                     'Businessman':1, 'Student':0}).astype('category')
    
        df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].\
            replace({'Secondary / secondary special':0, 
                     'Incomplete higher':2, 'Lower secondary':1,
                     'Higher education':3, 'Academic degree':4}).astype('category')
    
        df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].\
            replace({'Single / not married':0, 
                     'Widow':1, 'Separated':2,
                     'Married':3, 'Civil marriage':4, 
                     'Unknown':self.na_value}).astype('category')
    
        df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].\
            replace({'House / apartment':0, 
                     'Rented apartment':1, 'With parents':2,
                     'Municipal apartment':3, 'Co-op apartment':4,
                     'Office apartment':5}).astype('category')
    
        df['HOUSETYPE_MODE'] = df['HOUSETYPE_MODE'].\
            replace({'block of flats':0, 'terraced house':1,
                     'specific housing':2}).astype('category')
    
        df['WALLSMATERIAL_MODE'] = df['WALLSMATERIAL_MODE'].\
            replace({'Wooden':0, 'Others':1, 'Block':2, 'Mixed':3,
                     'Panel':4, 'Stone, brick':5, 
                     'Monolithic':6}).astype('category')
    
        df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].\
            replace({'Accountants':0, 'Secretaries':1, 'Medicine staff':2,
                     'High skill tech staff':5, 'IT staff':4, 'Managers':3,
                     'Private service staff':6, 'Core staff':7, 'HR staff':8,
                     'Cleaning staff':11, 'Cooking staff':10, 'Drivers':9,
                     'Laborers':12, 'Realty agents':13, 'Sales staff':14,
                     'Security staff':15, 'Waiters/barmen staff':16,
                     'Low-skill Laborers':17}).astype('category')

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
        
        df = df.drop(var_docs + var_aparts, #+ var_place + var_query + var_phone,
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
        df['EXT_SOURCE_AVG'] = \
            df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
        df['EXT_SOURCE_STD'] = \
            df[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].std(axis=1)
        
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