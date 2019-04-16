# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 07:47:52 2018

@author: Franc
"""
import random

import pandas as pd
import numpy as np

import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def from_pickle(filename):
    with open('cache/'+filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def to_pickle(obj, filename):
    with open('cache/'+filename, 'wb') as f:
        pickle.dump(obj, f, -1)
        
def VarRelation(data,feature):
    data = pd.crosstab(data['validRevenue'],
                       data[feature],margins=True).T.reset_index()
    data['ratio_valid'] = data[1]/11515
    data['ratio'] = data[1]/data[0]*77.47
    data = data.sort_values(['All'], ascending=False).reset_index(drop=True)
    return data

def VarPlot(data, feature):
    plt.subplots_adjust(wspace=1,hspace=0.2)
    ax1 = plt.subplot(211)
    sns.distplot(data.query('validRevenue==1')[feature].values);
    plt.subplot(212,sharex=ax1)
    sns.distplot(data.query('validRevenue==0')[feature].values);

def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
def VarDesc(data):
    df = data.agg(['nunique','dtype'],axis=0).T
    df['na'] = data.isnull().mean(axis=0)
    return df.reset_index()
    
    
    