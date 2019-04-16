#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 09:18:26 2018

@author: Franc, Jimmy
"""

import pandas as pd
import numpy as np

def ColReorder(data, normal = True):
    """ 重排序 """
    if normal:
        col_index = ['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD','MARKET_VALUE','REVENUE']
        col_revenue = ['REVENUE'+arg for arg in ['_S1','_S2','_S3','_Y1','_Y2','_Y3']]
        col_data = col_index+col_revenue+list(set(data.columns)-set(col_index)-set(col_revenue))
    else:
        col_index = ['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD','REVENUE']
        col_data = col_index+list(set(data.columns)-set(col_index))
    df = data.reindex(columns = col_data)
    return df

def ZeroRatio(data, pos, threshold):
    """ 判断变量为0的比例 """
    df = data.copy()
    df.drop(['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD','REVENUE'], 
            axis=1, inplace=True)
    lens = len(df)
    ratio = df.apply(lambda x: sum(x==0)/lens*100, axis=0)
    cols = ratio.index[ratio>threshold] if pos else ratio.index[(ratio<threshold)|(ratio==100)]
    return cols

def ColBinary(data, pos, threshold):
    """ 变量分类为正常和异常 """
    df = data.copy()
    cols = ZeroRatio(df, pos=pos, threshold=threshold)
    df.drop(cols, axis=1, inplace=True)
    return df

def DataSeries(data, pos, threshold):
    """ 变量完整季度化 """
    df = data.copy()
    df = ColBinary(df, pos, threshold)
    symbol,yearlist = list(df.TICKER_SYMBOL)[0], list(df.END_DATE)
    periodlist = list(df.FISCAL_PERIOD)
    series_ori = set(zip(yearlist,periodlist))
    series_imp = list(set([(year,quar) for year in set(yearlist)\
                           for quar in set(periodlist)])-series_ori)
    df = pd.merge(df, pd.DataFrame(series_imp, columns=['END_DATE','FISCAL_PERIOD']),
                  how='outer', on=['END_DATE','FISCAL_PERIOD']).\
                  sort_values(['END_DATE','FISCAL_PERIOD']).reset_index(drop=True)
    df.TICKER_SYMBOL = df.TICKER_SYMBOL.fillna(symbol)
    df.loc[(df.END_DATE==2018)&(df.FISCAL_PERIOD==2),'REVENUE'] = 0
    return df

def DataDiffNeg(data, pos=False, threshold=20):
    """ 异常数据季度化 """
    df = data.copy()
    df = DataSeries(data, pos=pos, threshold=threshold)
    np_ori = df.drop(['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD'], 
                     axis=1,inplace=False).values
    lens = np_ori.shape[1]
    np_full = np.zeros([np_ori.shape[0],lens*2])+np.nan
    np_full[1:,0] = np_ori[:-1,0]
    np_full[1:,1:lens] = np_ori[1:,1:] - np_ori[:-1,1:]
    np_full[4:,lens] = np_ori[:-4,0]
    np_full[4:,lens+1 : 2*lens] = np_ori[4:,1:] - np_ori[:-4,1:]
    colnames = [col+lag for lag in ['_S1','_Y1'] for col in list(df.columns)[3:]]
    df = pd.concat([df[['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD','REVENUE']],
                   pd.DataFrame(np_full, columns=colnames).sort_index(axis=1)],
         axis=1,sort=False).dropna()
    return df

def DataDiffPos(data, pos=True, threshold=20, dropnan = True):
    """ 正常数据季度化 """
    df = data.copy().pipe(DataSeries, pos=pos, threshold=threshold)
    np_ori = df.drop(['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD'], axis=1).values
    qlen, lens = np_ori.shape
    if qlen >= 32:
        np_full = np.zeros([np_ori.shape[0],lens*6])+np.nan
        np_full[1:,:lens] = np_ori[:-1]
        np_full[2:,lens:lens*2] = np_ori[:-2]
        np_full[3:,lens*2:lens*3] = np_ori[:-3]
        np_full[4:,lens*3:lens*4] = np_ori[:-4]
        np_full[8:,lens*4:lens*5] = np_ori[:-8]
        np_full[12:,lens*5:lens*6] = np_ori[:-12]
        colnames = [col+lag for lag in ['_S1','_S2','_S3','_Y1','_Y2','_Y3'] \
                    for col in list(df.columns)[3:]]
    elif qlen == 28:
        np_full = np.zeros([np_ori.shape[0],lens*5])+np.nan
        np_full[1:,:lens] = np_ori[:-1]
        np_full[2:,lens:lens*2] = np_ori[:-2]
        np_full[3:,lens*2:lens*3] = np_ori[:-3]
        np_full[4:,lens*3:lens*4] = np_ori[:-4]
        np_full[8:,lens*4:lens*5] = np_ori[:-8]
        colnames = [col+lag for lag in ['_S1','_S2','_S3','_Y1','_Y2'] \
                    for col in list(df.columns)[3:]]
    else:
        np_full = np.zeros([np_ori.shape[0],lens*4])+np.nan
        np_full[1:,:lens] = np_ori[:-1]
        np_full[2:,lens:lens*2] = np_ori[:-2]
        np_full[3:,lens*2:lens*3] = np_ori[:-3]
        np_full[4:,lens*3:lens*4] = np_ori[:-4]
        colnames = [col+lag for lag in ['_S1','_S2','_S3','_Y1'] \
                    for col in list(df.columns)[3:]]
    df = pd.concat([df[['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD','REVENUE']],
                   pd.DataFrame(np_full, columns=colnames).sort_index(axis=1)],
        axis = 1,sort=False)
    return df.dropna() if dropnan else df

def GroupSeries(data,dropnan):
    """ 分组季度化 """
    df = data.copy()
    df_normal = df.groupby(['TICKER_SYMBOL'],as_index=False, sort=False).\
                apply(DataDiffPos,dropnan=dropnan).\
                pipe(ColReorder, normal = False).reset_index(drop=True)
    return df_normal



