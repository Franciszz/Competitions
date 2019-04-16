# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:48:08 2018

@author: Franc
"""

import numpy as np
import pandas as pd

from feature_vertical import googleVertical
from datetime import timedelta
import holidays

class googleExternal(googleVertical):
    
    def __init__(self):
        super(googleVertical,self).__init__()
        
    def featureExternal(self, df, train_mode=True):
        df = df.copy()
        self.us_holidays = list(holidays.UnitedStates(
                years=[2016,2017,2018]).keys())
        df['external_us_holiday'] = (df['visitId'].dt.date.isin(
                self.us_holidays)).astype(int)
        df['external_us_holiday_delta_1'] = \
            df['external_us_holiday'] + self.judge_us_holiday(df, 1)
        df['external_us_holiday_delta_2'] = \
            df['external_us_holiday_delta_1'] + self.judge_us_holiday(df, 2)
        df['external_us_holiday_delta_3'] = \
            df['external_us_holiday_delta_2'] + self.judge_us_holiday(df, 3)
        df['external_us_holiday_delta_4'] = \
            df['external_us_holiday_delta_3'] + self.judge_us_holiday(df, 4)
        df['external_us_holiday_delta_5'] = \
            df['external_us_holiday_delta_4'] + self.judge_us_holiday(df, 5)
            
        df['external_visit_is_local_holiday'] = self.judge_local_holiday(df)
        df['date'] = df['date'].astype(int)
        component = self.UsdxIndexInput()
        df = df.merge(component, on='date', how='left')
        component = self.EconomicIndexInput()
        df = df.merge(component, on='date', how='left')
        
        df[['external_usdx_index','external_employment',
              'external_rate','external_unemployment']] = \
            df[['external_usdx_index','external_employment','external_rate',
                  'external_unemployment']].fillna(method='ffill')
        return df
    
    def UsdxIndexInput(self):
        df = pd.read_csv('data/usdxIndex.csv', 
                         parse_dates=[0]).fillna(method='ffill')
        df['date'] = df['date'].apply(
                lambda x: x.strftime('%Y%m%d')).astype(int)
        return df
    
    def EconomicIndexInput(self):
        df = pd.read_csv('data/economicsIndex.csv', 
                         parse_dates=[0]).fillna(method='ffill')
        df['date'] = df['date'].apply(
                lambda x: x.strftime('%Y%m%d')).astype(int)
        return df
        
    def judge_us_holiday(self, df, delta):
        judge_1 = \
            (df['visitId']+timedelta(days=delta)).dt.date.isin(
                    self.us_holidays)
        judge_2 = \
            (df['visitId']+timedelta(days=-delta)).dt.date.isin(
                    self.us_holidays)
        judge_holiday_delta = (judge_1|judge_2).astype(int)
        return judge_holiday_delta
    
    def judge_local_holiday(self, df):
        country = df['geoNetwork_country']
        date = df['visitId'].apply(lambda x: x.date())
        judge_holiday = \
            np.where(country.isin(
                    ['United States','India','Canada','Germany',
                     'Japan','France','Mexico','Australia',
                     'Spain','Netherlands','Italy','Ireland',
                     'Sweden','Argentina','Colombia','Belgium',
                     'Switzerland','Czechia','Colombia','Belgium',
                     'New Zealand','South Africa','South Africa']),\
            np.where((country=='United States')&
                     (date.isin(holidays.US())),1,
                     np.where((country=='India')&
                              (date.isin(holidays.India())),1,
                              np.where((country=='Canada')&
                                       (date.isin(holidays.CA())),1,
                                       np.where((country=='Germany')&
                                                (date.isin(holidays.DE())),1,\
            np.where((country=='Japan')&
                     (date.isin(holidays.JP())),1,
                     np.where((country=='France')&
                              (date.isin(holidays.FRA())),1,
                              np.where((country=='Mexico')&
                                       (date.isin(holidays.MX())),1,
                                       np.where((country=='Australia')&
                                                (date.isin(holidays.AU())),1,\
            np.where((country=='Spain')&
                     (date.isin(holidays.ES())),1,
                     np.where((country=='Netherlands')&
                              (date.isin(holidays.NL())),1,
                              np.where((country=='Italy')&
                                       (date.isin(holidays.IT())),1,
                                       np.where((country=='Ireland')&
                                                (date.isin(holidays.IE())),1,\
            np.where((country=='Sweden')&
                     (date.isin(holidays.SE())),1,
                     np.where((country=='Argentina')&
                              (date.isin(holidays.AR())),1,
                              np.where((country=='Colombia')&
                                       (date.isin(holidays.CO())),1,
                                       np.where((country=='Belgium')&
                                                (date.isin(holidays.BE())),1,\
            np.where((country=='Switzerland')&
                     (date.isin(holidays.CH())),1,
                     np.where((country=='Czechia')&
                              (date.isin(holidays.CZ())),1,
                              np.where((country=='Denmark')&
                                       (date.isin(holidays.DK())),1,
                                       np.where((country=='Austria')&
                                                (date.isin(holidays.AT())),1,\
            np.where((country=='Hungary')&
                     (date.isin(holidays.HU())),1,
                     np.where((country=='Portugal')&
                              (date.isin(holidays.PT())),1,
                              np.where((country=='Norway')&
                                       (date.isin(holidays.NO())),1,
                                       np.where((country=='Portugal')&
                                                (date.isin(holidays.PT())),1,\
            np.where((country=='New Zealand')&
                     (date.isin(holidays.NZ())),1,
                     np.where((country=='South Africa')&
                              (date.isin(holidays.ZA())),1,
                              np.where((country=='South Africa')&
                                       (date.isin(holidays.ZA())),1,\
            0))))))))))))))))))))))))))),np.nan).astype(int)
        return judge_holiday