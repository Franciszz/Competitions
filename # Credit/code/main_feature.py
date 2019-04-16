# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 20:20:09 2018

@author: Franc
"""

import os
os.chdir('C:/Users/Franc/Desktop/Dir/Competitions/#Kaggle/#Credit')
         
import sys
sys.path.append('code')

import pandas as pd
import numpy as np
from feature_application import ApplicationFeatures
from feature_bureau import BureauFeatures
from feature_instalment import InstalmentFeatures
from feature_pos_bal import PosBalFeatures
from feature_pre_appl import ApplPreFeatures
from feature_credit_bal import CreditCardBalFeatures
# ======================== Application Data =================================
def appl_input():
    df_test = pd.read_csv('data/application_test.csv')
    df_test.insert(1,'TARGET',2)
    df = pd.read_csv('data/application_train.csv').append(df_test)
    return df
df_appl = appl_input()
#var_obj = df_appl.columns[df_appl.dtypes=='object']
#corrs = df_appl.query('TARGET in [0,1]').corr().TARGET.abs().sort_index().\
#    reindex(df_appl.columns).sort_index()
AppClean = ApplicationFeatures(na_value = np.nan)
df_appl = AppClean.outcome(df_appl)
# ======================== Bureau Data ======================================
df_bureau = pd.read_csv('data/bureau.csv')
df_bureau_bal = pd.read_csv('data/bureau_balance.csv')

BuClean = BureauFeatures(fill_value = 0, df_bal = df_bureau_bal)
df_bureau = BuClean.outcome(df_bureau)

df_appl = df_appl.merge(df_bureau, on='SK_ID_CURR', how='left')
# ======================== Credit Card Data =================================
df_credit_bal = pd.read_csv('data/credit_card_balance.csv')

CardBalClean = CreditCardBalFeatures(fill_value = 0)
df_credit_bal = CardBalClean.outcome(df = df_credit_bal)

df_appl = df_appl.merge(df_credit_bal, on='SK_ID_CURR', how='left')
# ======================== Installment Data =================================
df_instl_pay = pd.read_csv('data/installments_payments.csv')

InstalmentClean = InstalmentFeatures()
df_instl_pay = InstalmentClean.outcome(df = df_instl_pay)

df_appl = df_appl.merge(df_instl_pay, on='SK_ID_CURR', how='left')
# ======================== Pos Card Data ====================================
df_pos_bal = pd.read_csv('data/POS_CASH_balance.csv')

PosBalClean = PosBalFeatures()
df_pos_bal = PosBalClean.outcome(df = df_pos_bal)

df_appl = df_appl.merge(df_pos_bal, on='SK_ID_CURR', how='left')
# ======================== Previous Application Data ========================
df_pre_appl = pd.read_csv('data/previous_application.csv',nrows=10000)

ApplPreClean = ApplPreFeatures(na_value = np.nan)
df_pre_appl = ApplPreClean.outcome(df = df_pre_appl)

# df_appl.to_csv('data/applications_features.csv',index=False)
# df_appl = pd.read_csv('data/applications_features.csv')
df_appl = df_appl.merge(df_pre_appl, on='SK_ID_CURR', how='left')
# ======================== Features Added ===================================
df_feature = df_appl.drop(list(df_appl.columns)[2:94],axis=1)
# df_feature.to_csv('data/feature_append.csv',index=False)
