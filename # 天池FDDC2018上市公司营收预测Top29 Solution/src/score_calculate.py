#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 12:06:34 2018

@author: Jimmy, Francis
"""

import tushare as ts
import os
import pandas as pd
import numpy as np

path = '../../%s'
### 预测列表
pre_list = pd.read_csv(path % 'fddc1_data/predict_list.csv', header=None)[0].\
    str.split('.',expand=True).rename(columns={0:'TICKER_SYMBOL',1:'EXCHANGE_CODE'})

### 交易及行业数据
market_info = pd.read_excel(path % 'fddc1_data/Market Data.xlsx', sheet_name = 'DATA',
                       nrows=3629, encoding = 'utf-8', dtype = {'TICKER_SYMBOL':'str'},
                       usecols = [1,6])
df_market = market_info[market_info.TICKER_SYMBOL.isin(pre_list.TICKER_SYMBOL)].\
    assign(MARKET_VALUE = market_info.MARKET_VALUE/10e7).reset_index(drop=True)
    
### 读取最新的营业收入数据
df_revenue = ts.get_profit_data(2018,2)[['code','business_income']].dropna().\
    rename(columns={'code':'TICKER_SYMBOL','business_income':'revenue'})

def cal_score(df_revenue, df_pre, df_market):
    """ 计算得分及用于计算得分的公司数目 """
    df = pd.merge(pd.merge(df_revenue, df_pre, how='inner'), df_market, how='left')
    df['bias'] = (df.predict/df.revenue-1).abs().apply(lambda x : x if x<0.8 else 0.8)
    df['score'] = df.bias*np.log2(df.MARKET_VALUE) 
    return df.bias.mean(),df.score.mean(),len(df)

file_list = os.listdir(path % '47_152/submit')
print('\n')
### 遍历submit文件夹下的csv文件，计算score
for filename in file_list:
    df_submit = pd.read_csv(path % '47_152/submit/'+filename, header=None).\
        rename(columns={0:'TICKER_SYMBOL',1:'predict'})
    df_pre = df_submit.assign(TICKER_SYMBOL=df_submit.TICKER_SYMBOL.\
                apply(lambda x:x[:6]))
    bias, score, num = cal_score(df_revenue, df_pre, df_market)
    print('File:'+filename+'\tBias:'+str(np.round(bias,4))+
          '\tScore:'+str(np.round(score,4))+'\tNums:'+str(num)+'\n')
