# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 07:46:25 2018

@author: Franc
"""
from datetime import datetime

import pandas as pd
from pandas.io.json import json_normalize
import numpy as np

import json

from steppy.base import BaseTransformer

class googleDataInput(BaseTransformer):
    
    def __init__(self):
        super(googleDataInput,self).__init__()
        self.json_columns = ['device','geoNetwork','totals','trafficSource']
        
    def jsonInput(self, path = 'data/train.csv', train_mode = True):
        df = pd.read_csv(
                path, dtype = {'fullVisitorId':str},
                converters = {col: json.loads for col in self.json_columns})
        for col in self.json_columns:
            col_df = json_normalize(df[col]).pipe(self.filterCol)
            col_df.columns = [f'{col}_{subcol}' for subcol in col_df.columns]
            df = df.join(col_df)
        df = self.transform(df, train_mode)
        df.drop(self.json_columns + ['socialEngagementType','device_isMobile'],
                axis=1, inplace=True)
        df = df.sort_values(['fullVisitorId'])
        return df
        
    def transform(self, df, train_mode = True):
        """ totals_transactionRevenue """
        if train_mode:
            cols = df.columns.tolist()
            cols.remove('totals_transactionRevenue')
            df = df.reindex(columns = ['totals_transactionRevenue'] + sorted(cols))
            df.insert(1,'validRevenue',
                        (df['totals_transactionRevenue'].notnull()).astype('int32'))
            df['totals_transactionRevenue'] = \
                df['totals_transactionRevenue'].fillna(0).astype('float').\
                apply(lambda x: round(np.log(x+1),2) if x>0 else 0)
        """ totals_hits&views """
        df['totals_hits'] = df['totals_hits'].astype('float')
#        df.loc[df['totals_hits']>=30,'totals_hits'] = 30
        
        df['totals_pageviews'] = df['totals_pageviews'].astype('float')
#        df.loc[df['totals_pageviews']>=30,'totals_pageviews'] = 30
        
        """ visitStartTime """
        visitStartTime_parse = df['visitStartTime'].apply(datetime.fromtimestamp)
        df['visitDate'] = \
            visitStartTime_parse.apply(lambda x: int(x.date().strftime('%Y%m%d')))
        df['visitTime'] = \
            visitStartTime_parse.apply(lambda x: int(x.time().strftime('%H%M%S')))
        df['visitMonth'] = \
            visitStartTime_parse.apply(lambda x: x.month)
        df['visitHour'] = \
            visitStartTime_parse.apply(lambda x: x.hour)
        df['visitWeekday'] = \
            visitStartTime_parse.apply(lambda x: x.isoweekday())
        df.drop(['visitStartTime'], axis=1, inplace=True)
        df.replace({'not available in demo dataset':'not available'},inplace=True)
        return df
    
    def RemoveConstantColumns(self, df):
        columnNuniques = df.apply('nunique', axis=0)
        df = df[df.columns[columnNuniques > 1]]
        return df
    
    def stupidJsonInput(self, path = 'data/train.csv', train_mode=True):
        df = pd.read_csv(path, dtype = {'fullVisitorId':str})
        
        component = pd.read_csv(
                path, usecols=[2], converters = {'device':json.loads})
        component = json_normalize(component['device']).pipe(self.filterCol)
        component.columns = ['device_' + col for col in component.columns]
        df = df.concat([df, component], axis=1)
        
        component = pd.read_csv(
                path, usecols=[4], converters = {'geoNetwork':json.loads})
        component = json_normalize(component['geoNetwork']).pipe(self.filterCol)
        component.columns = ['geoNetwork_'+col for col in component.columns]
        
        df_total = pd.read_csv(
                path, usecols=[7], converters = {'totals':json.loads})
        df_total = json_normalize(df_total['totals']).pipe(self.filterCol)
        df_total.columns = ['totals_'+col for col in df_total.columns]
        
        df_traffic = pd.read_csv(path, usecols=[8], 
                                   converters = {'trafficSource':json.loads})
        df_traffic = json_normalize(df_traffic['trafficSource']).pipe(self.filterCol)
        df_traffic.columns = ['trafficSource_'+col for col in df_traffic.columns]
        
        df = self.transform(df, train_mode)
        
        df.drop(self.json_columns+['socialEngagementType','device_isMobile'],
                  axis=1, inplace=True)
        return df