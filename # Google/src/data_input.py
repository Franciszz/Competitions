# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 22:45:41 2018

@author: Franc
"""
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from tqdm import tqdm
from steppy.base import BaseTransformer

class googleInput(BaseTransformer):
    
    def __init__(self):
        super(BaseTransformer).__init__()
        
    def JsonInput(self, train_mode = True):
        path = 'data/train.csv' if train_mode else 'data/test.csv'
        json_columns = ['device','geoNetwork','totals','trafficSource']
        df = pd.read_csv(
                path, dtype = {'fullVisitorId':str}, 
                usecols = [0,1,3,5,6,9,10,11])
        for (col, col_index) in tqdm(zip(json_columns, [2,4,7,8])):
            component = json_normalize(pd.read_csv(
                    path, usecols = [col_index], 
                    converters = {col: json.loads})[col])
            component.columns = [
                    f'{col}_{subcol}' for subcol in component.columns]
            df = df.join(component)
        df = df.sort_values(['fullVisitorId','visitId',
                             'date']).reset_index()
        nunique_cols = df.nunique(axis=0)
        df.drop(nunique_cols[nunique_cols==1].index.tolist(),
                axis=1,inplace=True)
        leak_df = self.LeakDataInput(train_mode=train_mode)
        df = df.merge(leak_df, on=['visitId'], how='left')
        df.drop(['index'],axis=1,inplace=True)
        return df
    
    def ExtractInput(self, train_mode = True):
        path = 'train' if train_mode else 'test'
        df = pd.read_csv(
                f'data/extracted_fields_{path}.gz',
                dtype={'date': str, 'fullVisitorId': str,
                       'sessionId':str, 'visitId': np.int64}).\
            sort_values(['sessionId','visitId'])
        df.columns = df.columns.str.replace('.','_')
        leak_df = self.LeakDataInput(train_mode=train_mode)
        df = df.merge(leak_df, on=['visitId'], how='left')
        return df
        
    def LeakDataInput(self, train_mode):
        path = 'Train_external_data' if train_mode else 'Test_external_data'
        leak_df_1 = pd.read_csv(
                f'data/{path}.csv', low_memory=False, 
                skiprows=6, dtype={'Client Id':'str'})
        leak_df_2 = pd.read_csv(
                f'data/{path}_2.csv', low_memory=False, 
                skiprows=6, dtype={'Client Id':'str'})
        leak_df = pd.concat([leak_df_1, leak_df_2], sort=False)
        leak_df['visitId'] = leak_df['Client Id'].apply(
                lambda x: x.split('.', 1)[1]).astype('int64')
        leak_df.drop(['Client Id'], axis=1, inplace=True)
        return leak_df