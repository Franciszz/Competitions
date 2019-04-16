#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 21:04:08 2018

@author: Franc, Jimmy
"""

import pandas as pd
import numpy as np

def RecordLast(data, comlist, last):
    """ 重复报表的选择 """
    df = data.copy()
    df = df[df.TICKER_SYMBOL.isin(comlist)].fillna(0).\
         drop(['PARTY_ID','END_DATE_REP','REPORT_TYPE','MERGED_FLAG','EXCHANGE_CD'], 
              axis=1).drop_duplicates()
    dfs = df.assign(END_DATE = df.END_DATE.map(lambda x: int(x.year))).\
          sort_values(['TICKER_SYMBOL','END_DATE','PUBLISH_DATE']).\
          groupby(['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD'])
    df = dfs.last().reset_index() if last else dfs.first().reset_index()
    df = df.drop('PUBLISH_DATE',axis=1).assign(FISCAL_PERIOD = df.FISCAL_PERIOD/3)
    return df

def Quarterize(data):
    """ 数据季度化 """
    df = data.copy()
    period = list(df.FISCAL_PERIOD)
    lens = len(period)
    df_np = np.r_[np.zeros([1,df.shape[1]-3]),df.iloc[:,3:].values]
    df_np = df_np[1:] - df_np[:-1]
    if lens == 4:
        pass
    elif lens == 3:
        if 4 not in period:
            pass
        elif 3 not in period:
            df_np[2] /= 2
        elif 2 not in period:
            df_np[1] /= 2
        else:
            df_np[0] /= 2
    elif lens == 2:
        if 4 in period:
            if 3 in period:
                df_np[0] /= 3
            elif 2 in period:
                df_np /= 2
            else:
                df_np[1] /= 3
        elif 3 in period:
            if 2 in period:
                df_np[0] /= 2
            else:
                df_np[1] /= 2
        else:
            pass
    else:
        df_np /= period[0]
    df.iloc[:,3:] = df_np
    return df 

def QuarterProcess(data):
    """ 数据分组季度化 """
    df = data.groupby(['TICKER_SYMBOL','END_DATE'], as_index=False).\
        apply(Quarterize).reset_index(drop=True)
    return df

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

def ExcelData(path, filename, sheetname):
    df = pd.read_excel(path % filename, sheet_name=sheetname, header=0, 
                       dtype = {'TICKER_SYMBOL': 'str'}, parse_dates=[3,4,5])
    return df

def DataInput(path, comlist, last):
    """ 数据读取及合并 """
    bs_indu = ExcelData(path, 'Balance Sheet.xls', 'General Business').\
        pipe(RecordLast, comlist, last)
    bs_bank = ExcelData(path, 'Balance Sheet.xls', 'Bank').\
        pipe(RecordLast, comlist, last)
    bs_insu = ExcelData(path, 'Balance Sheet.xls', 'Insurance').\
        pipe(RecordLast, comlist, last)
    bs_secu = ExcelData(path, 'Balance Sheet.xls', 'Securities').\
        pipe(RecordLast, comlist, last)    
    cf_indu = ExcelData(path, 'Cashflow Statement.xls', 'General Business').\
        pipe(RecordLast, comlist, last).pipe(QuarterProcess)
    cf_bank = ExcelData(path, 'Cashflow Statement.xls', 'Bank').\
        pipe(RecordLast, comlist, last).pipe(QuarterProcess)
    cf_insu = ExcelData(path, 'Cashflow Statement.xls', 'Insurance').\
        pipe(RecordLast, comlist, last).pipe(QuarterProcess)
    cf_secu = ExcelData(path, 'Cashflow Statement.xls', 'Securities').\
        pipe(RecordLast, comlist, last).pipe(QuarterProcess)    
    is_indu = ExcelData(path, 'Income Statement.xls', 'General Business').\
        pipe(RecordLast, comlist, last).pipe(QuarterProcess)
    is_bank = ExcelData(path, 'Income Statement.xls', 'Bank').\
        pipe(RecordLast, comlist, last).pipe(QuarterProcess)
    is_insu = ExcelData(path, 'Income Statement.xls', 'Insurance').\
        pipe(RecordLast, comlist, last).pipe(QuarterProcess)
    is_secu = ExcelData(path, 'Income Statement.xls', 'Securities').\
        pipe(RecordLast, comlist, last).pipe(QuarterProcess)
    Vs = ['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD']    
    indu = pd.merge(pd.merge(bs_indu,cf_indu,on=Vs,how='inner'),is_indu,on=Vs,how='inner')                            
    bank = pd.merge(pd.merge(bs_bank,cf_bank,on=Vs,how='inner'),is_bank,on=Vs,how='inner')                            
    insu = pd.merge(pd.merge(bs_insu,cf_insu,on=Vs,how='inner'),is_insu,on=Vs,how='inner')                            
    secu = pd.merge(pd.merge(bs_secu,cf_secu,on=Vs,how='inner'),is_secu,on=Vs,how='inner')                            
    df = pd.concat([indu, insu, secu, bank], sort=True).fillna(0).\
         sort_values(Vs).reset_index(drop=True).pipe(ColReorder, normal=False).\
         groupby(Vs).last().reset_index(drop=False)
    return df[~(df.END_DATE.isin([2006,2007]))]