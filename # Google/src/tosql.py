# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:00:29 2018

@author: Franc
"""
import pandas as pd
import pymysql
from sqlalchemy import create_engine
# In[]
host, port, db = 'localhost','3306','data_competitions'
user, pwd = 'root','root'
# In[]
train = pd.read_csv('data/extracted_fields_train.gz', dtype={
        'date': str, 'fullVisitorId': str, 'sessionId':str, 'visitId': int})
test = pd.read_csv('data/extracted_fields_test.gz', dtype={
        'date': str, 'fullVisitorId': str, 'sessionId':str, 'visitId': int})
# In[]
engine = create_engine(
        f'mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}',
        echo=True)
# In[]
train.to_sql('train', engine, if_exists = 'replace', index=False)
# In[]
db = pymysql.connect(
        '127.0.0.1',
        
        'root',
        'root',
        'data_competitions')
# In[]
train.to_sql(name = 'train', con = db, if_exists='replace', schema='mysql')
# In[]
cursor = db.cursor()
# In[]
