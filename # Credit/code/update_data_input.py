# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 18:10:27 2018

@author: Franc
"""

import pandas as pd

from steppy.base import BaseTransformer
from steppy.utils import get_logger

logger = get_logger()

class DataInput(BaseTransformer):
    def __init__(self,**kwargs):
        self.path = './data'
        self.sk_id_curr_category = None
        self.sk_id_bureau_category = None
    
    def application_input(self):
        data_train = pd.read_csv(self.path+'/application_train.csv')
        data_test = pd.read_csv(self.path+'/application_test.csv')    
        data_test.insert(1,'TARGET',2)
        data = data_train.append(data_test)
        self.sk_id_curr_category = data.SK_ID_CURR
        return data
    
    def previous_input(self):
        data = pd.read_csv(self.path+'/previous_application.csv')
        data = data[data.SK_ID_CURR.isin(self.sk_id_curr_category)].\
                reset_index(drop=True)
        return data
    
    def bureau_input(self):
        data = pd.read_csv(self.path+'/bureau.csv')
        data = data[data.SK_ID_CURR.isin(self.sk_id_curr_category)].\
                reset_index(drop=True)
        self.sk_id_bureau_category = data.SK_ID_BUREAU
        return data
    
    def bureau_bal_input(self):
        data = pd.read_csv(self.path+'/bureau_balance.csv')
        data = data[data.SK_ID_BUREAU.isin(self.sk_id_bureau_category)].\
                reset_index(drop=True)
        return data
    
    def credit_input(self):
        data = pd.read_csv(self.path+'/credit_card_balance.csv')
        data = data[data.SK_ID_CURR.isin(self.sk_id_curr_category)].\
                reset_index(drop=True)
        return data
    def poscash_input(self):
        data = pd.read_csv(self.path+'/POS_CASH_balance.csv')
        data = data[data.SK_ID_CURR.isin(self.sk_id_curr_category)].\
                reset_index(drop=True)
        return data
    def instalment_input(self):
        data = pd.read_csv(self.path+'/installments_payments.csv')
        data = data[data.SK_ID_CURR.isin(self.sk_id_curr_category)].\
                reset_index(drop=True)
        return data