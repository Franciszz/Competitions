# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 23:12:25 2018

@author: Franc
"""
import pandas as pd
from data_input import googleInput

class googleCleaner(googleInput):
    
    def __init__(self):
        super(googleInput, self).__init__()
        
    def DataCleaner(self, df, train_mode = True):
        df = df.copy()
        
        """ totals_transactionRevenue """
        if train_mode:
            df['totals_transactionRevenue'] = \
                df['totals_transactionRevenue'].fillna(0).astype(float)
            self.target = df['totals_transactionRevenue']
            self.target_sums = df.groupby('fullVisitorId')[
                    'totals_transactionRevenue'].sum().reset_index()
            self.validTarget = (self.target>0).astype(int)
        
        """ totals_pageviews & totals_hits """
        df['totals_hits'] = df['totals_hits'].astype(float)
        df['totals_pageviews'] = df['totals_pageviews'].astype(float)
        
        """ visitStartTime """
        df['visitStartTime'] = pd.to_datetime(df['visitStartTime'], unit='s')
        
        """ date """
#        df['date'] = df['visitStartTime']
        
        """ Revenue """
        df['Revenue'] = df['Revenue'].str.replace('[$,]','').\
            fillna(0).astype(float)  
            
        """ Sessions """
        df['Sessions'].fillna(0, inplace=True)
        
        """ Avg. Session Duration """
        df['Avg. Session Duration'] = pd.to_timedelta(
                df['Avg. Session Duration']).map(lambda x:x.seconds).\
                fillna(0).astype(int)
                
        """ Bounce Rate """
        df['Bounce Rate'] = df['Bounce Rate'].str.\
            replace('%','').fillna(0).astype(float)
            
        """ Goal Conversion Rate """
        df['Goal Conversion Rate'] = df['Goal Conversion Rate'].str.\
            replace('%','').fillna(0).astype(float)
            
        """ Transactions """
        df['Transactions'].fillna(0, inplace=True)
        return df