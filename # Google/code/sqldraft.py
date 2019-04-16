# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:28:05 2018

@author: Franc
"""
import pymysql
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
# In[连接数据库]
host, port, db = 'localhost','3306','data_competitions'
user, pwd, encoding = 'root','root', 'utf8'
engine = create_engine(
        f'mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}?charset={encoding}',
        echo=False)
# In[]
db = engine.connect()
sql = 'select count(*) from train_sample;'
db.execute(sql).fetchall()
# In[]
Session = sessionmaker(bind = engine)
session = Session()
# In[操作数据库]
sql = 'show databases;'
session.execute(sql).fetchall()
# In[]
sql = 'use data_competitions;'
dir(session.execute(sql))
# In[]
sql = 'select * from train_sample;'
a = session.execute(sql).fetchall()
# In[]
import pandas as pd
train = pd.read_sql_table('train_sample', engine)
sql = 'select * from train_sample where device_browser=%d;'
train = pd.read_sql_query(str(sql)%7,engine)
# In[pymysql]
conn = pymysql.connect(host=host, user=user, password = pwd, 
                       database=db, charset=encoding)
cur = conn.cursor()
cur.execute(str(sql)%6)
cols = pd.DataFrame(list(cur.description))[0].tolist()
train = pd.DataFrame(list(cur.fetchall()))
train.columns = cols
cur.close()
conn.close()
# In[创建映射——创建表]
#from sqlalchemy.ext.declarative import declarative_base
#from sqlalchemy import Column, Integer, String
#Base = declarative_base()
#class Person(Base):
#    __tablename__ = 'userinfo'
#    id = Column(Integer,primary_key = True)
#    name = Column(String(32))
#    
#    def __repr__(self):
#        return "Person(name='%s')>"%self.name
## In[添加数据]
#person = Person(name='xxx')
#session.add(person)
#session.commit()