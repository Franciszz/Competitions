# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 18:13:23 2018

@author: Franc
"""

import os
os.chdir('C:/Users/Franc/Desktop/Dir/Competitions/#Kaggle/#Credit')
         
import sys
sys.path.append('code')

from update_data_input import DataInput

from update_feature_application import ApplicationFeature
from update_feature_previous import PreviousFeature
from update_feature_bureau import BureauFeature
from update_feature_credit import CreditFeature
from update_feature_pos import PosCashFeature
from update_feature_instalment import InstalFeature
# ===================================================================================
data_input = DataInput()
### ====== Application Feature 
df_application = data_input.application_input()
application_feature = ApplicationFeature()
df_application = application_feature.feature_extract(df_application)
# 53个变量缺失率>0.5, 大部分为房子信息和拥有车的年龄的比率数据
df_application.to_csv('./data/feature_application_update.csv',index=False)

### ====== Previous Application Feature
df_previous = data_input.previous_input()
previous_feature = PreviousFeature()
df_previous = previous_feature.feature_extract(df_previous)
df_previous.to_csv('./data/feature_previous_update.csv',index=False)

### ====== Bureau Feature
df_bureau = data_input.bureau_input()
df_bureau_bal = data_input.bureau_bal_input()
bureau_feature = BureauFeature()
df_bureau = bureau_feature.feature_extract(df_bureau, df_bureau_bal)
df_bureau.to_csv('./data/feature_bureau_update.csv',index=False)

### ====== Credit Cards Balance Feature
df_credit = data_input.credit_input()
credit_feature = CreditFeature()
df_credit = credit_feature.feature_extract(df_credit)
df_credit.to_csv('./data/feature_credit_update.csv',index=False)

### ====== Pos Cash Balance Feature
df_pos = data_input.poscash_input()
pos_feature = PosCashFeature()
df_pos = pos_feature.feature_extract(df_pos)
df_pos.to_csv('./data/feature_pos_update.csv',index=False)

### ====== Instalment Feature
df_instal = data_input.instalment_input()
instal_feature = InstalFeature()
df_instal = instal_feature.feature_extract(df_instal)
df_instal.to_csv('./data/feature_instal_update.csv',index=False)

