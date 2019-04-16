#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 21:03:03 2018

@author: Franc, Jimmy
"""

import pandas as pd
import sys
import datetime

### path
path = '../../%s' 
sys.path.append(path % '/47_152/code')
from data_input import DataInput
from data_transform import GroupSeries
from data_model import AddMacroData, AddInduData
from model_train import RevenuePre, RevenuePreIndu , RevenueTran, RevenueCom

### 预测列表
pre_list = pd.read_csv(path%'fddc1_data/predict_list.csv', header=None)[0].\
    str.split('.',expand=True).rename(columns={0:'TICKER_SYMBOL',1:'EXCHANGE_CODE'})
    
### 交易及行业数据
market_info = pd.read_excel(path%'fddc1_data/Market Data.xlsx', sheet_name = 'DATA',
                       nrows=3629, encoding = 'utf-8', dtype = {'TICKER_SYMBOL':'str'},
                       usecols = [1,6,8], parse_dates=[2])
market_info = market_info[market_info.TICKER_SYMBOL.isin(pre_list.TICKER_SYMBOL)].\
    assign(MARKET_VALUE = market_info.MARKET_VALUE/10e7).reset_index(drop=True)
    
### 宏观经济数据
macro_info = pd.read_excel(path%'fddc1_data/Macro&Industry.xlsx',header=0,\
              sheet_name='INDIC_DATA',encoding='utf8',dtype = {'indic_id': 'str'},\
              parse_dates=[1],usecols=[0,4,5],index_col='PERIOD_DATE')['20101231':]
macro_info.index =  macro_info.index+ datetime.timedelta(85)
### 财务数据
df_lst = DataInput(path = '../../fddc1_data/financial_data/%s',
                   comlist = pre_list.TICKER_SYMBOL, last=False)
#df_lst.to_csv(path%'/47_152/data/df_fst.csv',index=False)
#df_lst = pd.read_csv(path%'/47_152/data/df_lst.csv',dtype = {'TICKER_SYMBOL':'str'})
### 季度化数据
df_normal = GroupSeries(df_lst,dropnan = True)
### 添加宏观数据
df_model_macro = AddMacroData(df_normal, macro_info)
### 模型训练
df_xgb_ind = RevenuePre(df_model_macro)

df_predict_ind = RevenueTran(df_model_macro, df_xgb_ind, market_info,
                             [1.24, 1.07, 1.00, 1.15], 0.5, 0.75, 0.24)
df_submit_ind = RevenueCom(df_predict_ind, pre_list)

### 行业营收模型
df_model_indu = AddInduData(df_normal, market_info[['TICKER_SYMBOL','TYPE_NAME_EN']])
### 模型训练
df_xgb_indu = RevenuePreIndu(df_model_indu)
df_predict_indu = RevenueTran(df_model_macro, df_xgb_indu, market_info,
                             [1.24, 1.07, 1.00, 1.15], 0.5, 0.75, 0.24)
df_submit_indu = RevenueCom(df_predict_indu, pre_list)

df_submit_indu.PREDICT = df_submit_ind.PREDICT*0.45+df_submit_indu.PREDICT*0.55
df_submit_indu.to_csv(path%'/47_152/submit/submit_%s_%s.csv'%(
        datetime.datetime.now().strftime('%Y%m%d'),
        datetime.datetime.now().strftime('%H%M%S')),
    encoding='utf-8',header=False,index=False)

