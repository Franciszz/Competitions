# -*- coding: utf-8 -*-
'''
Created on Sun Oct 14 21:36:47 2018

@author: Franc
'''

import pandas as pd
from data_input import googleDataInput

class dataLeakInput(googleDataInput):
    
    def __init__(self):
        super(dataLeakInput, self).__init__()
    
    def leakDataInput(self, train_mode = True):
        filepath = 'Train_external_data' if train_mode else 'Test_external_data'
        leak_df_1 = pd.read_csv(
                f'data/{filepath}.csv', low_memory=False, 
                skiprows=6, dtype={'Client Id':'str'})
        leak_df_2 = pd.read_csv(
                f'data/{filepath}_2.csv', low_memory=False, 
                skiprows=6, dtype={'Client Id':'str'})
        leak_df = pd.concat([leak_df_1, leak_df_2], sort=False)
        leak_df['visitId'] = leak_df['Client Id'].apply(
                lambda x: x.split('.', 1)[1]).astype('int64')
        leak_df.drop(['Client Id'], axis=1, inplace=True)
        return leak_df
    
    def 
        
        
        

        
        
        