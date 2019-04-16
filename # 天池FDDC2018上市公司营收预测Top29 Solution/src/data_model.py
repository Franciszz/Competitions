# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:35:31 2018

@author: Franc, Jimmy
"""
import pandas as pd
import numpy as np

def AddMacroData(data, macro_data):
    """ 宏观经济数据添加 """
    macro_info = macro_data.copy()
    macro_info = macro_info.groupby('indic_id').resample('Q').\
        apply(np.mean).reset_index(level=[0,1])
    macro_info.insert(1,'END_DATE',macro_info.PERIOD_DATE.apply(lambda x: x.year))
    macro_info.insert(2,'FISCAL_PERIOD',macro_info.PERIOD_DATE.apply(lambda x: x.quarter))
    macro_info = macro_info.pivot_table(index=['END_DATE','FISCAL_PERIOD'],\
        columns=['indic_id'],values='DATA_VALUE').reset_index()[:-1].dropna(axis=1)
    df = pd.merge(data, macro_info, how='left')
    return df
def FillRev(data):
    df = data.copy()
    df.index = df.END_DATE*10+df.FISCAL_PERIOD
    Year = [2008]+sorted([x for x in range(2009,2018)]*4)+[2018,2018]
    Quar = [4]+[x for x in range(1,5,1)]*9+[1,2]
    df = df.reindex(index = [20084]+[x*10+y for x in range(2009,2018) \
                    for y in range(1,5,1)]+[20181,20182])
    df = df.assign(END_DATE = Year, FISCAL_PERIOD = Quar,
                   REVENUE = df.REVENUE.interpolate(method = 'quadratic',limit_area='inside'),
                   TICKER_SYMBOL = df.TICKER_SYMBOL.fillna(method='bfill'),
                   TYPE_NAME_EN = df.TYPE_NAME_EN.fillna(method='bfill'))
    df.loc[(df.END_DATE==2018)&(df.FISCAL_PERIOD==2),'REVENUE'] = 0
    return df[df.REVENUE.notna()].reset_index(drop=True)
def RevDiff(data):
    df = data.copy()
    a,b = df.shape
    np_full = np.zeros([a,2*(b-5)])+np.nan
    col_names = [col+arg for arg in ['_S1','_Y1'] for col in list(df.columns[5:])]
    np_full[1:,:(b-5)] = df.iloc[:-1,5:].values
    np_full[4:,(b-5):] = df.iloc[:-4,5:].values
    dat = pd.DataFrame(np.c_[df.iloc[:,:5].values,np_full],
                       columns=['TICKER_SYMBOL','TYPE_NAME_EN','END_DATE',
                                'FISCAL_PERIOD','REVENUE']+col_names)
    return dat.reset_index(drop=True).iloc[1:,:]
def AddInduData(data, indu_data):
    indu_info = indu_data.copy()
    df_rev = data[['TICKER_SYMBOL','END_DATE','FISCAL_PERIOD','REVENUE']].\
        join(indu_info.set_index('TICKER_SYMBOL'),on='TICKER_SYMBOL').\
        reindex(columns = ['TICKER_SYMBOL','TYPE_NAME_EN','END_DATE','FISCAL_PERIOD','REVENUE'])
    df_rev = df_rev.groupby('TICKER_SYMBOL',as_index=False).apply(FillRev)
    df_rev_table = df_rev.groupby('TYPE_NAME_EN',as_index=True).\
        apply(pd.pivot_table, index = ['END_DATE','FISCAL_PERIOD'], 
              columns = 'TICKER_SYMBOL', values='REVENUE').reset_index(drop=False)
    df = pd.merge(df_rev,df_rev_table, how = 'outer',
                  on=['TYPE_NAME_EN','END_DATE','FISCAL_PERIOD']).\
                  groupby('TICKER_SYMBOL').apply(RevDiff).reset_index(drop=True)
    return df

