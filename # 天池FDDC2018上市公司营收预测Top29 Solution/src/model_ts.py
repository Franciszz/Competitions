# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 09:34:08 2018

@author: jimmy
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

def DataTs(data):
    """ 从df_final数据转化为时间序列数据 """
    df_ts = data[['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD','REVENUE']]
    df_ts['date'] = sm.tsa.datetools.\
        dates_from_str([str(year)+'Q'+str(int(quarter)) for 
                        year,quarter in zip(df_ts.END_DATE,df_ts.FISCAL_PERIOD)])
    df_ts = df_ts.set_index('date').drop(['END_DATE','FISCAL_PERIOD'],axis=1)   
    ### 筛选出数据大于10期的公司
    bool_g10 = df_ts.groupby('TICKER_SYMBOL').size()>10
    TICKER_SYMBOL_g10 = bool_g10.index[bool_g10.values]
    df_ts = df_ts[df_ts.TICKER_SYMBOL.isin(TICKER_SYMBOL_g10)]
    df_ts.REVENUE = df_ts.REVENUE/1e6
    return df_ts

def p_i_q_select(timeseries,pmax=3,qmax=3,diffmax=3):
    """ 识别输入的时间序列的最优p,d,q """
    ts=timeseries.dropna(axis=0) 
    result = []
    pq_index = []
    for i in range(pmax):
        for j in range(qmax):
            for d in range(diffmax):
                pq_index.append((i,d,j))
                try:
                    arma = sm.tsa.ARMA(ts,(i,d,j)).fit(disp=-1,method='mle')
                    average=np.mean([arma.aic,arma.bic,arma.hqic])
                    result.append(average)
                except:
                    result.append(np.inf)
    final_result = pd.Series(result,index=pq_index)[1:]
    return list(final_result.index[final_result==final_result.min()])[0]

def predict(data):
    """ 输入一个公司的营收序列数据，返回预测营收 """
    ts=data.REVENUE[:'20180401']
    order = p_i_q_select(ts,pmax=3,qmax=3,diffmax=3)
    arma = sm.tsa.ARMA(ts,order).fit(disp=-1,method='mle')
    try:
        result = arma.forecast(1)[0][0]+ts['20180331']
    except:
        result = np.nan
    return result

def RevenuePreTs(data):
    df=data.groupby('TICKER_SYMBOL').apply(predict).\
        dropna().reset_index().\
        rename(columns={0:'predicted'})
    return df[['TICKER_SYMBOL','predicted']]
