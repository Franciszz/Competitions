#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 00:37:43 2018

@author: Franc
"""

import pandas as pd
import numpy as np
import datetime 
import random

from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV

path = '/home/fddc1_data/%s'
path1 = '/home/fddc1_data/financial_data/%s'

def RecordLast(data, comlist, last = True):
    """ 重复报表的选择 """
    df = data.copy()
    df = df[df.TICKER_SYMBOL.isin(comlist)].fillna(0).\
         drop(['PARTY_ID','END_DATE_REP','REPORT_TYPE','MERGED_FLAG','EXCHANGE_CD'], 
              axis=1).drop_duplicates()
    dfs = df.assign(END_DATE = df.END_DATE.map(lambda x: int(x.year))).\
          sort_values(['TICKER_SYMBOL','END_DATE','PUBLISH_DATE']).\
          reset_index(drop=True).\
          groupby(['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD'], as_index=False)
    if last:
        df = dfs.last().reset_index(drop=True)
    else:
        df = dfs.first().reset_index(drop=True)
    df = df.drop('PUBLISH_DATE',axis=1).assign(FISCAL_PERIOD = df.FISCAL_PERIOD/3)
    return df

def Quarterize(data):
    """ 数据季度化 """
    df = data.copy()
    period = list(df.FISCAL_PERIOD)
    lens = len(period)
    df_np = np.r_[np.zeros([1,df.shape[1]-4]),df.iloc[:,4:].values]
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
    df.iloc[:,4:] = df_np
    return df 

def QuarterProcess(data):
    """ 数据分组季度化 """
    df = data.copy()
    df = df.groupby(['TICKER_SYMBOL','END_DATE'], as_index=False).\
          apply(Quarterize).reset_index(drop=True, inplace=False)
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

def DataInput(path, comlist):
    """ 数据读取及合并 """
    bs_indu = pd.read_excel(path % 'Balance Sheet.xls', sheet_name='General Business', 
                            head=0, encoding='utf8', dtype = {'TICKER_SYMBOL': 'str'}, 
                            parse_dates=[3,4,5]).pipe(RecordLast,comlist)
    bs_bank = pd.read_excel(path % 'Balance Sheet.xls', sheet_name='Bank', 
                            head=0, encoding='utf8', dtype = {'TICKER_SYMBOL': 'str'}, 
                            parse_dates=[3,4,5]).pipe(RecordLast,comlist)
    bs_insu = pd.read_excel(path % 'Balance Sheet.xls', sheet_name='Insurance' ,
                            head=0, encoding='utf8', dtype = {'TICKER_SYMBOL': 'str'}, 
                            parse_dates=[3,4,5]).pipe(RecordLast,comlist)
    bs_secu = pd.read_excel(path % 'Balance Sheet.xls', sheet_name='Securities' ,
                            head=0, encoding='utf8', dtype = {'TICKER_SYMBOL': 'str'}, 
                            parse_dates=[3,4,5]).pipe(RecordLast,comlist)
    cf_indu = pd.read_excel(path % 'Cashflow Statement.xls', sheet_name='General Business', 
                            head=0, encoding='utf8', dtype = {'TICKER_SYMBOL': 'str'}, 
                            parse_dates=[3,4,5]).pipe(RecordLast,comlist).pipe(QuarterProcess)
    cf_bank = pd.read_excel(path % 'Cashflow Statement.xls', sheet_name='Bank' , 
                            head=0, encoding='utf8', dtype = {'TICKER_SYMBOL': 'str'}, 
                            parse_dates=[3,4,5]).pipe(RecordLast,comlist).pipe(QuarterProcess)
    cf_insu = pd.read_excel(path % 'Cashflow Statement.xls', sheet_name='Insurance' , 
                            head=0, encoding='utf8', dtype = {'TICKER_SYMBOL': 'str'}, 
                            parse_dates=[3,4,5]).pipe(RecordLast,comlist).pipe(QuarterProcess)
    cf_secu = pd.read_excel(path % 'Cashflow Statement.xls', sheet_name='Securities', 
                            head=0, encoding='utf8', dtype = {'TICKER_SYMBOL': 'str'}, 
                            parse_dates=[3,4,5]).pipe(RecordLast,comlist).pipe(QuarterProcess)
    is_indu = pd.read_excel(path % 'Income Statement.xls', sheet_name='General Business', 
                            head=0, encoding='utf8', dtype = {'TICKER_SYMBOL': 'str'}, 
                            parse_dates=[3,4,5]).pipe(RecordLast,comlist).pipe(QuarterProcess)
    is_bank = pd.read_excel(path % 'Income Statement.xls', sheet_name='Bank', 
                            head=0,encoding='utf8', dtype = {'TICKER_SYMBOL': 'str'}, 
                            parse_dates=[3,4,5]).pipe(RecordLast,comlist).pipe(QuarterProcess)
    is_insu = pd.read_excel(path % 'Income Statement.xls', sheet_name='Insurance', 
                            head=0, encoding='utf8', dtype = {'TICKER_SYMBOL': 'str'}, 
                            parse_dates=[3,4,5]).pipe(RecordLast,comlist).pipe(QuarterProcess)
    is_secu = pd.read_excel(path % 'Income Statement.xls', sheet_name='Securities', 
                            head=0, encoding='utf8', dtype = {'TICKER_SYMBOL': 'str'}, 
                            parse_dates=[3,4,5]).pipe(RecordLast,comlist).pipe(QuarterProcess)
    indu = pd.merge(pd.merge(bs_indu, cf_indu, 
                    on = ['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD'], how='inner'),
                    is_indu, 
                    on = ['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD'], how='inner')                            
    bank = pd.merge(pd.merge(bs_bank, cf_bank, 
                    on = ['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD'], how='inner'),
                    is_bank, 
                    on = ['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD'], how='inner')
    insu = pd.merge(pd.merge(bs_insu, cf_insu, 
                    on = ['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD'], how='inner'),
                    is_insu, 
                    on = ['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD'], how='inner')
    secu = pd.merge(pd.merge(bs_secu, cf_secu, 
                    on = ['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD'], how='inner'),
                    is_secu, 
                    on = ['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD'], how='inner')
    df = pd.concat([indu, insu, secu, bank], sort=True).fillna(0).\
         sort_values(['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD']).\
         reset_index(drop=True).pipe(ColReorder, normal=False).\
         groupby(['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD']).\
         last().reset_index(drop=False)
    return df[~(df.END_DATE.isin([2006,2007]))]

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
         axis=1,sort=True).dropna()
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
        axis = 1,sort=True)
    return df.dropna() if dropnan else df

def InduMv(data,dropnan=False):
    """ 分组季度化 """
    df = data.copy()
    df_normal = df.groupby(['TICKER_SYMBOL'],as_index=False, sort=True).\
                apply(DataDiffPos,dropnan=dropnan).\
                pipe(ColReorder, normal = False).reset_index(drop=True)
    return df_normal

def revenue_predict_ind(data, reg):
    """ XGBoost模型构建 """
    df = data.copy().dropna(axis='columns')
    cond = (df.END_DATE==2018)&(df.FISCAL_PERIOD==2)
    df_tr, df_te = df[~cond].values, df[cond].values
    tr_x, tr_y = df_tr[:,4:], df_tr[:,3:4]
    te_x = df_te[:,4:] 
    if len(te_x)>0:
        reg.fit(tr_x,tr_y)
        return reg.predict(te_x)[0]
    else:
        return np.nan

def ModelData(data):
    """ 模型训练集和测试集合生成合并 """
    df = data.copy()
    df_normal = InduMv(df,dropnan=False)
    df_test = df_normal[(df_normal.END_DATE==2018)&(df_normal.FISCAL_PERIOD==2)].copy()
    df_train = InduMv(df,dropnan=True)
    df_final = pd.concat([df_train, df_test],axis=0).pipe(ColReorder,normal=True).\
        sort_values(['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD']).\
        drop(['MARKET_VALUE'],axis=1)
    df_final.REVENUE = df_final.REVENUE.fillna(0)
    return df_final


###############################################################
def AddAdvData(data):
    """ 宏观经济数据生成 """
    market_info = pd.read_excel(path % 'Macro&Industry.xlsx',header=0,\
              sheet_name='数据信息',encoding='utf8',dtype = {'indic_id': 'str'},\
              parse_dates=[3])[['indic_id','PERIOD_DATE','DATA_VALUE']].\
                                set_index('PERIOD_DATE')['20101231':]
    market_info1=market_info.groupby('indic_id').resample('Q',how=np.mean).reset_index(level=[0,1])
    market_info1.insert(1,'END_DATE',market_info1.PERIOD_DATE.apply(lambda x: x.year))
    market_info1.insert(2,'FISCAL_PERIOD',market_info1.PERIOD_DATE.apply(lambda x: x.quarter))
    market_info2 = market_info1.pivot_table(index=['END_DATE','FISCAL_PERIOD'],
                                            columns=['indic_id'],values='DATA_VALUE').\
                                            reset_index()[:-1].dropna(axis=1)
    df=pd.merge(df_final,market_info2,how='left')
    return df


###############################################################

def RevenuePre(data):
    """ 营业收入预测 """
    df = data.copy()
    xgb_reg = XGBRegressor(objective = 'reg:linear', max_depth = 10, n_estimators = 50, 
                           learning_rate = 0.02, gamma = 0.01, reg_lambda = 1, 
                           silent = 0, random_state = random.randint(0,909))
    parms = dict(max_depth=[2,4,6],
                 n_estimators = [20,50],
                 learning_rate = [0.01,0.05,0.5])
    xgb_gs = GridSearchCV(xgb_reg, parms, cv=4)
    rev_model = df_final.groupby('TICKER_SYMBOL',as_index=True).\
             apply(revenue_predict_ind, xgb_gs)
    rev_past = df[(df.END_DATE==2018)&(df.FISCAL_PERIOD==2)]\
             [['TICKER_SYMBOL','REVENUE_S1','REVENUE_Y1']].set_index('TICKER_SYMBOL')
    df = pd.DataFrame(rev_model,columns=['predicted']).join(rev_past)
    df['REVENUE_PRE'] = df.predicted*0.5+df.REVENUE_Y1*1.15*0.5+df.REVENUE_S1
    df.reset_index(drop=False,inplace=True)
    return df

def RevenueCom(data):
    """ 营业收入预测缺失值补充 """
    df = data.copy()
    df = pd.merge(df,com_data,how='outer',on='TICKER_SYMBOL')
    df.loc[df.TICKER_SYMBOL=='000563','REVENUE_PRE'] = 499.00*10e5
    df.loc[df.TICKER_SYMBOL=='600816','REVENUE_PRE'] = 2323.00*10e5
    df.loc[df.TICKER_SYMBOL=='000627','REVENUE_PRE'] = 25252.00*10e5
    df['TICKER_SYMBOL'] = [x+'.'+y for x,y in zip(df.TICKER_SYMBOL,df.EXCHANGE)]
    df = df[['TICKER_SYMBOL','REVENUE_PRE']]
    df.REVENUE_PRE = np.round(df.REVENUE_PRE/10e5,2)
    return df


com_data = pd.read_csv(path % 'predict_list.csv', header=None)[0].\
    str.split('.',expand = True).rename(columns={0:'TICKER_SYMBOL',1:'EXCHANGE'})
    
df_lst = DataInput(path = path1, comlist = list(com_data.TICKER_SYMBOL))

df_final = ModelData(df_lst)

df_final1=AddAdvData(df_final)

df_predict = RevenuePre(df_final1)
df_submit = RevenueCom(df_predict)
df_submit.to_csv('/home/47_152/submit/submit_%s_%s.csv'%(
        datetime.datetime.now().strftime('%Y%m%d'),
        datetime.datetime.now().strftime('%H%M%S')),
    encoding='utf-8',header=False,index=False)

