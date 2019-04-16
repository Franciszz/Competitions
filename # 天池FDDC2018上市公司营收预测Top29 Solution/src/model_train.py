#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 09:17:08 2018

@author: Franc, Jimmy
"""
import random
import pandas as pd
import numpy as np

from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV

# ====== XGBOOST =======
xgb_reg = XGBRegressor(objective = 'reg:linear', max_depth = 10, n_estimators = 50, 
                       learning_rate = 0.02, gamma = 0.01, reg_lambda = 1, 
                       silent = 0, random_state = 909)
parms = dict(max_depth=[2,4,6],
             n_estimators = [20,50],
             learning_rate = [0.01,0.05,0.5])
xgb_gs = GridSearchCV(xgb_reg, parms, cv=4)

parms_indu = dict(max_depth=[3,6,9],
             n_estimators = [20,50],
             learning_rate = [0.01,0.05,0.5])

xgb_indu = GridSearchCV(xgb_reg, parms, cv=4)


def revenue_predict_ind(data, reg):
    """ XGBoost模型构建 """
    df = data.copy().dropna(axis='columns',how='all')
    cond = (df.END_DATE==2018)&(df.FISCAL_PERIOD==2)
    df_tr, df_te = df[~cond].values, df[cond].values
    tr_x, tr_y = df_tr[:,4:], df_tr[:,3:4]
    te_x = df_te[:,4:] 
    if len(te_x)>0:
        reg.fit(tr_x,tr_y)
        pre = reg.predict(te_x)[0]
        return pre
    else:
        return np.nan
    
def RevenuePre(data, reg = xgb_gs):
    """ 营业收入预测 """
    df = data.copy()
    rev_model = df.groupby('TICKER_SYMBOL',as_index=True).apply(revenue_predict_ind, reg)
    df = pd.DataFrame(rev_model,columns=['predicted'])
    df.reset_index(drop=False,inplace=True)
    return df

def revenue_predict_indu(data, reg):
    """ XGBoost模型构建 """
    df = data.copy().dropna(axis='columns',how='all')
    cond = (df.END_DATE==2018)&(df.FISCAL_PERIOD==2)
    df_tr, df_te = df[~cond].values, df[cond].values
    tr_x, tr_y = df_tr[:,5:], df_tr[:,4:5]
    te_x = df_te[:,5:]
    reg.fit(tr_x,tr_y)
    pre = reg.predict(te_x)
    return pre

def RevenuePreIndu(data, reg = xgb_indu):
    """ 营业收入预测 """
    df = data.copy()
    df_indu = df.groupby('TICKER_SYMBOL').apply(revenue_predict_indu,xgb_indu)
    df_indu = pd.DataFrame(dict(TICKER_SYMBOL=df_indu.index,
                                predicted = df_indu.values))
    df_indu.predicted = df_indu.predicted.apply(lambda x: x[0] if len(x)>0 else np.nan)
    df_indu.reset_index(drop=True,inplace=True)
    return df_indu

def InduWeight(data, betaIndu, betaBank, betaSecu, betaInsu):
    df = data.copy()
    df['beta'] =  betaIndu
    df.loc[df.TYPE_NAME_EN=='Bank','beta'] = betaBank
    df.loc[df.TYPE_NAME_EN=='Non-bank Finance','beta'] = betaSecu
    df.loc[df.TICKER_SYMBOL.isin(['600291','601318','601336','601601','601628']),
           'beta'] = betaInsu
    return df[['TICKER_SYMBOL','beta']].set_index('TICKER_SYMBOL')

def RevenueTran(data, df_pre, df_indu, beta=[1.15,1.08,0.95,1.12], 
                weight = 0.5, alpha1 = 0.5, alpha2 = 0.15):
    df = data.copy()
    df_past = df[(df.END_DATE==2018)&(df.FISCAL_PERIOD==2)]\
             [['TICKER_SYMBOL','REVENUE_S1','REVENUE_Y1']].set_index('TICKER_SYMBOL') 
    df_indu = InduWeight(df_indu, beta[0], beta[1], beta[2], beta[3])            
    df = df_pre.join(df_past,on='TICKER_SYMBOL').join(df_indu,on='TICKER_SYMBOL')
    df.predicted = np.where(df.predicted<(1-alpha1)*df.REVENUE_Y1,
                            df.REVENUE_Y1*(1-alpha2),df.predicted)
    df.predicted = np.where(df.predicted>(1+alpha1)*df.REVENUE_Y1,
                            df.REVENUE_Y1*(1+alpha2),df.predicted)
    df['PREDICT'] = np.round((df.predicted*(1-weight)+df.REVENUE_Y1*df.beta*weight+
        df.REVENUE_S1)/10e5,2)                    
    return df

def RevenueCom(data,data_pre):
    """ 营业收入预测缺失值补充 """
    df = data.copy()
    df = pd.merge(df,data_pre,how='outer',on='TICKER_SYMBOL')
    df.loc[df.TICKER_SYMBOL=='000563','PREDICT'] = 499.00
    df.loc[df.TICKER_SYMBOL=='600816','PREDICT'] = 2323.00
    df.loc[df.TICKER_SYMBOL=='000627','PREDICT'] = 25252.00
    df['TICKER_SYMBOL'] = [x+'.'+y for x,y in zip(df.TICKER_SYMBOL,df.EXCHANGE_CODE)]
    df = df[['TICKER_SYMBOL','PREDICT']]
    return df
