# -*- coding: utf-8 -*-
"""
@author: ATCG, Jan 7th, 2019
"""

import os
import pandas as pd
import numpy as np
import warnings
import gc
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
warnings.simplefilter('ignore')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
gc.enable()
os.chdir('C:/Users/Franc/Desktop/Dir/Competitions/# DigitalManufacture')


mse_diff = lambda df1, df2: np.mean(np.square(df1.iloc[:, 1]-df2.iloc[:, 1]))


# ====== Data Input ======================================================
df_trn = pd.read_csv('data/jinnan_round1_train_20181227.csv', encoding='GB2312')
trn = pd.read_csv('data/jinnan_round1_train_20181227.csv', encoding='GB2312')
df_tst = pd.read_csv('data/jinnan_round1_testA_20181227.csv', encoding='GB2312')
tst = pd.read_csv('data/jinnan_round1_testA_20181227.csv', encoding='GB2312')


# ====== Abnormal Revise ======
df_trn['A5'] = df_trn['A5'].replace('1900/1/21 0:00', '21:00:00')
df_trn['A5'] = df_trn['A5'].replace('1900/1/29 0:00', '14:00:00')
df_trn['A9'] = df_trn['A9'].replace('1900/1/9 7:00', '23:00:00')
df_trn['A9'] = df_trn['A9'].replace('700', '7:00:00')
df_trn['A11'] = df_trn['A11'].replace(':30:00', '00:30:00')
df_trn['A11'] = df_trn['A11'].replace('1900/1/1 2:30', '21:30:00')
df_trn['A16'] = df_trn['A16'].replace('1900/1/12 0:00', '12:00:00')
df_trn['A20'] = df_trn['A20'].replace('6:00-6:30分', '6:00-6:30')
df_trn['A20'] = df_trn['A20'].replace('18:30-15:00', '18:30-19:00')
df_trn['A22'] = df_trn['A22'].replace(3.5, np.nan)
df_trn['A25'] = df_trn['A25'].replace('1900/3/10 0:00', 70).astype(int)
df_trn['A26'] = df_trn['A26'].replace('1900/3/13 0:00', '13:00:00')
df_trn['B1'] = df_trn['B1'].replace(3.5, np.nan)
df_trn['B4'] = df_trn['B4'].replace('15:00-1600', '15:00-16:00')
df_trn['B4'] = df_trn['B4'].replace('18:00-17:00', '16:00-17:00')
df_trn['B4'] = df_trn['B4'].replace('19:-20:05', '19:05-20:05')
df_trn['B9'] = df_trn['B9'].replace('23:00-7:30', '23:00-00:30')
df_trn['B14'] = df_trn['B14'].replace(40, 400)
df_tst['A5'] = df_tst['A5'].replace('1900/1/22 0:00', '22:00:00')
df_tst['A7'] = df_tst['A7'].replace('0:50:00', '21:50:00')
df_tst['B14'] = df_tst['B14'].replace(785, 385)

# ====== Manual Judgement ======
df_trn.loc[df_trn['样本id'] == 'sample_1894', 'A5'] = '14:00:00'
df_trn.loc[df_trn['样本id'] == 'sample_1234', 'A9'] = '0:00:00'
df_trn.loc[df_trn['样本id'] == 'sample_1020', 'A9'] = '18:30:00'

df_trn.loc[df_trn['样本id'] == 'sample_1380', 'A11'] = '15:30:00'
df_trn.loc[df_trn['样本id'] == 'sample_844', 'A11'] = '10:00:00'
df_trn.loc[df_trn['样本id'] == 'sample_1348', 'A11'] = '17:00:00'
df_trn.loc[df_trn['样本id'] == 'sample_25', 'A11'] = '00:30:00'
df_trn.loc[df_trn['样本id'] == 'sample_1105', 'A11'] = '4:00:00'
df_trn.loc[df_trn['样本id'] == 'sample_313', 'A11'] = '15:30:00'

df_trn.loc[df_trn['样本id'] == 'sample_291', 'A14'] = '19:30:00'

df_trn.loc[df_trn['样本id'] == 'sample_1398', 'A16'] = '11:00:00'
df_trn.loc[df_trn['样本id'] == 'sample_1177', 'A20'] = '19:00-20:00'

df_trn.loc[df_trn['样本id'] == 'sample_71', 'A20'] = '16:20-16:50'
df_trn.loc[df_trn['样本id'] == 'sample_14', 'A20'] = '18:00-18:30'
df_trn.loc[df_trn['样本id'] == 'sample_69', 'A20'] = '6:10-6:50'
df_trn.loc[df_trn['样本id'] == 'sample_1500', 'A20'] = '23:00-23:30'

df_trn.loc[df_trn['样本id'] == 'sample_1524', 'A24'] = '15:00:00'
df_trn.loc[df_trn['样本id'] == 'sample_1524', 'A26'] = '15:30:00'

df_trn.loc[df_trn['样本id'] == 'sample_1046', 'A28'] = '18:00-18:30'

df_trn.loc[df_trn['样本id'] == 'sample_1230', 'B5'] = '17:00:00'
df_trn.loc[df_trn['样本id'] == 'sample_97', 'B7'] = '1:00:00'
df_trn.loc[df_trn['样本id'] == 'sample_752', 'B9'] = '11:00-14:00'

df_trn.loc[df_trn['样本id'] == 'sample_609', 'B11'] = '11:00-12:00'
df_trn.loc[df_trn['样本id'] == 'sample_643', 'B11'] = '12:00-13:00'
df_trn.loc[df_trn['样本id'] == 'sample_1164', 'B11'] = '5:00-6:00'

df_tst.loc[df_tst['样本id'] == 'sample_919', 'A9'] = '19:50:00'


def desc(data, *func):
    df = data.agg(['dtype', 'nunique', *func]).T
    df['absent'] = data.isnull().mean(axis=0)*100
    df = df.reset_index().rename(columns={'index': 'variable'})
    return df


# ====== Time Transform Functions ==========================================
def time_to_min(x):
    if x is np.nan:
        return np.nan
    else:
        x = x.replace(';', ':').replace('；', ':')
        x = x.replace('::', ':').replace('"', ':')
        h, m = x.split(':')[:2]
        h = 0 if not h else h
        m = 0 if not m else m
        return int(h)*60 + int(m)


# ====== Extract Target Value ================================================
df_target = df_trn[['样本id', '收率']]
del df_trn['收率']
df_trn_tst = df_trn.append(df_tst, ignore_index=False).reset_index(drop=True)
# df_trn_tst['A3'].fillna(570, inplace=True)
# df_trn_tst['A7'] = df_trn_tst['A7'].notnull().astype('int8')
# del df_trn_tst['A7'], df_trn_tst['A8']
del df_trn, df_tst
gc.collect()


# ====== TimeIndex to Hour ====================================================
cols_timer = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
for _df in [df_trn_tst]:
    _df.rename(columns={_col: _col + '_t' for _col in cols_timer}, inplace=True)
    for _col in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
        _idx_col = _df.columns.tolist().index(_col)
        _df.insert(_idx_col + 1, _col + '_at', _df[_col].str.split('-').str[0])
        _df.insert(_idx_col + 2, _col + '_bt', _df[_col].str.split('-').str[1])
        del _df[_col]
        cols_timer = cols_timer + [_col + '_at', _col + '_bt']
cols_timer = list(filter(lambda x: x.endswith('t'), df_trn_tst.columns))
del _df, _col, _idx_col
gc.collect()


# ====== Transform TimeIndex to Minutes ========================================
for _df in [df_trn_tst]:
    for _col in cols_timer:
        _df[_col] = _df[_col].map(time_to_min)
del _df, _col, cols_timer
gc.collect()


# ====== TimeIndex Difference ==================================================
def duration_outer(series1, series2):
    duration = series1 - series2
    duration = np.where(duration < 0, duration + 24*60, duration)
    duration = np.where(duration > 12*60, 24*60 - duration, duration)
    duration = np.where(duration > 6*60, 12*60 - duration, duration)
    return duration


# ====== Feature Engineer ======================================================
def feature_engineer(df_raw):
    data = df_raw.copy()
    df = pd.DataFrame(index=data.index)
    # ====== 加热流程 ======
    df['样本id'] = data['样本id']
    # 4-氰基吡啶投入量
    df['A1_m'] = data['A1']
    # 氢氧化钠溶液投入量
    df['A3_m'] = data['A3'].fillna(570)
    # 开始加热时点
    df['A5_t'] = data['A5_t']
    # 初始容器温度
    df['A6_c'] = data['A6']
    # 是否多一次测量温度（有点怪）
    df['A7_b'] = data['A7_t'].notnull().astype('int8')
    # 开始加热到初次沸腾时长
    df['A9_td'] = duration_outer(data['A9_t'], data['A5_t'])
    # 初次沸腾温度
    df['A10_c'] = data['A10']
    # ====== 水解流程 ======
    # 开始水解时点
    df['A11_t'] = data['A11_t']
    # 开始加热到开始水解时间
    df['A11_0_td'] = duration_outer(data['A11_t'], data['A5_t'])
    # 沸腾至开始水解恒温时间
    df['A11_1_td'] = duration_outer(data['A11_t'], data['A9_t'])
    # 开始水解时刻温度
    df['A12_c'] = data['A12']
    # 容器初始温度与水解温度温差
    df['A12_cd'] = data['A12'] - data['A6']
    # 水解反应中间测温时差
    df['A14_td'] = duration_outer(data['A14_t'], data['A11_t'])
    # 水解反应中间测温温度
    df['A15_c'] = data['A15']
    # 水解反应中间测温温差
    df['A15_cd'] = data['A15'] - data['A12']
    # 水解反应完成时点
    df['A16_t'] = data['A16_t']
    # 水解反应时长
    df['A16_0_td'] = duration_outer(data['A16_t'], data['A11_t'])
    # 初始记时到水解反应完成时长
    df['A16_1_td'] = duration_outer(data['A16_t'], data['A5_t'])
    # 水解完成温度
    df['A17_c'] = data['A17']
    # 水解完成温差
    df['A17_0_cd'] = data['A17'] - data['A12']
    # 水解完成与容器初始温差
    df['A17_1_cd'] = data['A17'] - data['A12']
    # 水解过程添加水量（保持体积不变）
    df['A19_m'] = data['A19']
    # ====== 脱色流程 ======
    # 水解反应完成到脱色过程时长
    df['A20_0_td'] = duration_outer(data['A20_at'], data['A16_t'])
    # 脱色过程(或降温过程)开始时点
    df['A20_0_t'] = data['A20_at']
    # 脱色过程(或降温过程)时长
    df['A20_1_td'] = duration_outer(data['A20_at'], data['A20_bt'])
    # 脱色过程添加材料
    df['A21_m'] = data['A21']
    df['A22_m'] = data['A22']
    df['A23_m'] = data['A23']
    df['A23_md'] = data['A23'] - data['A22']
    # 脱色过程到脱色保温过程时长
    df['A24_td'] = duration_outer(data['A24_t'], data['A20_bt'])
    # 降温温差
    df['A25_0_cd'] = data['A25'] - data['A17']
    # 脱色保温初始温度
    df['A25_c'] = data['A25']
    # 脱色保温时长
    df['A26_td'] = duration_outer(data['A26_t'], data['A24_t'])
    # 脱色保温结束温度
    df['A27_c'] = data['A27']
    # 脱色保温温差
    df['A27_cd'] = data['A27'] - data['A25']
    # 脱色保温到抽滤去除活性炭时长
    df['A28_0_td'] = duration_outer(data['A28_at'], data['A26_t'])
    # 抽滤时长
    df['A28_td'] = duration_outer(data['A28_bt'], data['A28_at'])
    # 脱色流程时长
    df['A28_1_td'] = duration_outer(data['A28_bt'], data['A16_t'])
    # ====== 酸化流程 ======
    # 酸化结晶过程添加盐酸
    df['B1_m'] = data['B1']
    # 酸化初始PH
    df['B2_m'] = data['B2']
    # 酸化初始PH与之前PH差
    df['B2_md'] = data['B2'] - data['A23']
    # 酸化PH差
    df['B3_md'] = data['B3'] - data['B2']
    # 抽滤结束至酸化时长
    df['B4_0_td'] = duration_outer(data['B4_at'], data['A28_bt'])
    # 酸化开始时点
    df['B4_0_t'] = data['B4_at']
    # 酸化时长
    df['B4_1_td'] = duration_outer(data['B4_bt'], data['B4_at'])
    # 自然结晶时长
    df['B5_td'] = duration_outer(data['B5_t'], data['B4_bt'])
    # 自然结晶结束温度
    df['B6_c'] = data['B6']
    # 自然结晶结束前后温差
    df['B6_cd'] = data['B6'] - data['A27']
    # 调温时长
    df['B7_td'] = duration_outer(data['B7_t'], data['B5_t'])
    # 甩滤前温度
    df['B8_c'] = data['B8']
    # 调温温度
    df['B8_cd'] = data['B8'] - data['B6']
    # 调温结束至甩滤时长
    df['B9_0_td'] = duration_outer(data['B9_at'], data['B7_t'])
    # 甩滤开始时点
    df['B9_0_t'] = data['B9_at']
    # 甩滤基本流程时长
    df['B9_1_td'] = duration_outer(data['B9_bt'], data['B9_at'])
    # 甩滤补充流程1时长
    df['B10_td'] = duration_outer(data['B10_bt'], data['B10_at'])
    # 甩滤补充流程2时长
    df['B11_td'] = duration_outer(data['B11_bt'], data['B11_at'])
    # 甩滤流程数
    df['B11_m'] = data[['B9_bt', 'B10_bt', 'B11_bt']].notnull().sum(axis=1)
    # 甩率结束时点
    df['B12_t'] = np.where(
        data['B11_bt'].isnull(),
        np.where(data['B10_bt'].isnull(), data['B9_bt'], data['B10_bt']),
        data['B11_bt'])
    # 甩滤总时长
    df['B12_td'] = duration_outer(df['B12_t'], data['B9_at'])
    # 甩滤过程添加水量
    df['B12_m'] = data['B12']
    # 甩率理论产出
    df['B14_m'] = data['B14']

    # ====== 归纳特征 ======
    _col_c = [_f for _f in df.columns if _f.endswith('c')]
    df['B14_c_avg'] = df[_col_c].mean(axis=1)
    df['B14_c_std'] = df[_col_c].std(axis=1)
    _col_cd = [_f for _f in df.columns if _f.endswith('cd')]
    df['B14_cd_avg'] = df[_col_cd].mean(axis=1)
    df['B14_cd_std'] = df[_col_cd].std(axis=1)
    _col_t = [_f for _f in df.columns if _f.endswith('t')]
    df['B14_t_avg'] = df[_col_t].mean(axis=1)
    df['B14_t_std'] = df[_col_t].std(axis=1)
    _col_td = [_f for _f in df.columns if _f.endswith('td')]
    df['B14_td_avg'] = df[_col_td].mean(axis=1)
    df['B14_td_std'] = df[_col_td].std(axis=1)
    for _f in _col_cd:
        if (df[_f] < 0).mean() > 0.20:
            _idx_col = df.columns.tolist().index(_f)
            df.insert(_idx_col + 1, _f+'_abs', df[_f].abs())
    _col_abs = [_f for _f in df.columns if _f.endswith('abs')]
    df['B14_abs_avg'] = df[_col_abs].mean(axis=1)
    df['B14_abs_std'] = df[_col_abs].std(axis=1)
    # ====== 关于原料投入
    # 总用水量
    df['B15_1m'] = data['A4'] + data['A19']
    # 水解用水
    df['B15_2m'] = data['A4'] + data['A19'] + data['B12']
    # 4-氰基吡啶投入量 / 用水
    df['B15_1mr'] = data['A1'] / data['A4']
    # 4-氰基吡啶投入量 / 水解用水
    df['B15_2mr'] = data['A1'] / df['B15_2m']
    # 4-氰基吡啶投入量 / 氢氧化钠
    df['B15_3mr'] = data['A1'] / df['A3_m']
    # 4-氰基吡啶投入量 / 水解盐酸量
    df['B15_4mr'] = data['A1'] / data['A21']
    # 氢氧化钠 / 水解用水
    df['B15_5mr'] = df['A3_m'] / df['B15_2m']
    # 总盐酸量
    df['B15_3m'] = data['A21'] + data['B1']
    # 脱色用料
    df['B15_4m'] = data['A21'] + data['A22'] + data['A23']
    # 脱色物质3/2
    df['B15_6mr'] = data['A23'] / data['A22']
    # 脱色物质2/1
    df['B15_7mr'] = data['A22'] / data['A21']
    _col_m = [_f for _f in df.columns if _f.endswith('m')]
    for _f in _col_m:
        if _f != 'B14_m':
            df[f'{_f}_B14_m_r'] = df['B14_m'] / df[_f]
    # ====== 关于测温次数
    df['B16_1m'] = data[
        ['A5_t', 'A7_t', 'A9_t', 'A11_t', 'A14_t', 'A16_t', 'A20_at',
         'A24_t', 'A26_t', 'A28_at', 'B4_at', 'B5_t', 'B7_t', 'B9_at',
         'B10_at', 'B11_at']].notnull().sum(axis=1)

    # ====== 关于水解过程温度变化
    df['B17_1m'] = data[['A10', 'A12', 'A15', 'A17']].mean(axis=1)
    df['B17_2m'] = data[['A10', 'A12', 'A15', 'A17']].std(axis=1)
    df['B17_3m'] = data[['A10', 'A12', 'A15', 'A17']].min(axis=1)
    df['B17_4m'] = data[['A10', 'A12', 'A15', 'A17']].max(axis=1)

    # ====== 净流程时间(不包括加热，降温)
    df['B18_1m'] = df[['A16_0_td', 'A20_1_td', 'A26_td', 'A28_td',
                       'B4_1_td', 'B12_td']].sum(axis=1)
    # ====== 净流程 + 加热
    df['B18_2m'] = df['B18_1m'] + df['A11_0_td']
    # ====== 净流程 + 加热 + 降温
    df['B18_3m'] = df[
        ['B18_2m', 'A11_0_td', 'A20_0_td', 'B7_td', 'B5_td']].sum(axis=1)
    # ====== 闲杂时间
    df['B18_4m'] = df[
        ['B9_0_td', 'B4_0_td', 'A28_0_td', 'A24_td', 'A20_0_td']].sum(axis=1)
    return df


df_feature = feature_engineer(df_trn_tst)


def train_duplicated_median(data):
    df, len_df = data.head(1), len(data)
    df['收率'] = data['收率'].median()
    return df


df_trn = df_feature.iloc[:len(trn)].reset_index(drop=True)
df_trn['收率'] = df_target['收率']
df_tst = df_feature.iloc[len(trn):].reset_index(drop=True)
df_trn = df_trn.fillna(999).\
    groupby(df_trn.columns.tolist()[1:-1], as_index=False).\
    apply(train_duplicated_median).\
    replace(999, np.nan).reset_index(drop=True)

# df_trn = df_trn.query('收率 >= 0.80').reset_index(drop=True)
# df_trn = df_trn.query('收率 <= 1.00').reset_index(drop=True)
#
# for i in range(10):
#     df_trn[f'recall_{i}'] = \
#         ((0.85 + i * 0.015 <= df_trn['收率']) & \
#          (df_trn['收率'] < 0.85 + (i+1) * 0.015)).astype('int8')

df_trn = df_trn.query('收率 > 0.8671').reset_index(drop=True)# 8671
df_trn = df_trn.query('收率 < 0.9861').reset_index(drop=True)# 9861

df_trn['recall_1'] = (df_trn['收率'] < 0.8908).astype('int8')
df_trn['recall_2'] = ((0.8908 <= df_trn['收率']) & \
                      (df_trn['收率'] < 0.9299)).astype('int8')
df_trn['recall_3'] = ((0.9010 <= df_trn['收率']) & \
                      (df_trn['收率'] < 0.9299)).astype('int8')
df_trn['recall_4'] = ((0.9299 <= df_trn['收率']) & \
                      (df_trn['收率'] < 0.9460)).astype('int8')
df_trn['recall_5'] = (df_trn['收率'] >= 0.9460).astype('int8')

# df_trn['recall'] = pd.cut(df_trn['收率'], 5, labels=False)
# df_trn = pd.get_dummies(df_trn, columns=['recall'])
# li = ['recall_0', 'recall_1', 'recall_2', 'recall_3', 'recall_4']
# mean_columns = []

for _f in df_tst.columns.tolist()[1:]:
    vc = df_trn[_f].value_counts(normalize=True)
    if len(vc) <= 12 and vc.iloc[0] <= 0.6:
        recall_mean = df_trn.groupby([_f])['收率'].mean()
        for _df in [df_trn, df_tst]:
            _idx_col = _df.columns.tolist().index(_f)
            _df.insert(_idx_col + 1, f'{_f}_mean', _df[_f].map(recall_mean))
    corr = np.abs(np.corrcoef(df_trn[_f], df_trn['收率']))[0, 1]
    if len(vc) <= 20 and corr >= 0.32:
        for _col in [f'recall_{i}' for i in range(1, 6)]:
            recall_count = df_trn.groupby([_f])[_col].mean()
            for _df in [df_trn, df_tst]:
                _idx_col = _df.columns.tolist().index(_f)
                _df.insert(_idx_col + 1, f'{_f}_{_col}_count',
                           _df[_f].map(recall_count))
for _f in [f'recall_{i}' for i in range(1, 6)]:
    del df_trn[_f]
del _f, vc, corr, recall_mean, recall_count, _idx_col, _df, _col
# for _df in [df_trn, df_tst]:
#     _df.insert(1, 'id', _df['样本id'].str.split('_').str[1].astype(int))
# del _df
gc.collect()


def lgb_cv(train, test, params, fit_params,
           cat_features, feature_names, nfold, seed):
    train.Pred = pd.DataFrame({
        'id': train['样本id'],
        'true': train['收率'],
        'pred': np.zeros(len(train))})
    test.Pred = pd.DataFrame({'id': test['样本id'], 'pred': np.zeros(len(test))})
    kfolder = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    for fold_id, (trn_idx, val_idx) in enumerate(kfolder.split(train['收率'])):
        print(f'\nFold_{fold_id} Training ================================\n')
        lgb_trn = lgb.Dataset(
            data=train.iloc[trn_idx][feature_names],
            label=train.iloc[trn_idx]['收率'],
            categorical_feature=cat_features,
            feature_name=feature_names)
        lgb_val = lgb.Dataset(
            data=train.iloc[val_idx][feature_names],
            label=train.iloc[val_idx]['收率'],
            categorical_feature=cat_features,
            feature_name=feature_names)
        lgb_reg = lgb.train(params=params, train_set=lgb_trn, **fit_params,
                  valid_sets=[lgb_trn, lgb_val])
        val_pred = lgb_reg.predict(
            train.iloc[val_idx][feature_names],
            num_iteration=lgb_reg.best_iteration)
        train.Pred.loc[val_idx, 'pred'] = val_pred
        print(f'Fold_{fold_id}', mse(train.iloc[val_idx]['收率'], val_pred))
        test.Pred['pred'] += lgb_reg.predict(
            test[feature_names], num_iteration=lgb_reg.best_iteration) / nfold
    print('\n\nCV LOSS:', mse(train.Pred['true'], train.Pred['pred']))
    return test.Pred, train.Pred


def xgb_cv(train, test, params, fit_params, feature_names, nfold, seed):
    train.Pred = pd.DataFrame({
        'id': train['样本id'],
        'true': train['收率'],
        'pred': np.zeros(len(train))})
    test.Pred = pd.DataFrame({'id': test['样本id'], 'pred': np.zeros(len(test))})
    kfolder = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    xgb_tst = xgb.DMatrix(data=test[feature_names])
    for fold_id, (trn_idx, val_idx) in enumerate(kfolder.split(train['收率'])):
        print(f'\nFold_{fold_id} Training ================================\n')
        xgb_trn = xgb.DMatrix(
            train.iloc[trn_idx][feature_names],
            train.iloc[trn_idx]['收率'])
        xgb_val = xgb.DMatrix(
            train.iloc[val_idx][feature_names],
            train.iloc[val_idx]['收率'])
        xgb_reg = xgb.train(params=params, dtrain=xgb_trn, **fit_params,
                  evals=[(xgb_trn, 'train'), (xgb_val, 'valid')])
        val_pred = xgb_reg.predict(
            xgb.DMatrix(train.iloc[val_idx][feature_names]),
            ntree_limit=xgb_reg.best_ntree_limit)
        train.Pred.loc[val_idx, 'pred'] = val_pred
        print(f'Fold_{fold_id}', mse(train.iloc[val_idx]['收率'], val_pred))
        test.Pred['pred'] += xgb_reg.predict(
            xgb_tst, ntree_limit=xgb_reg.best_ntree_limit) / nfold
    print('\n\nCV LOSS:', mse(train.Pred['true'], train.Pred['pred']))
    return test.Pred, train.Pred


cat_features = [df_trn.columns.tolist()[1:-1].index(_f) for
                _f in df_trn.columns if _f.endswith('_all')]

# ====== lgb ============================================================
fit_params = {'num_boost_round': 10800, 'verbose_eval': 360,
              'early_stopping_rounds': 360}
params_lgb = {'num_leaves': 120, 'max_depth': 7, 'learning_rate': 0.005,
              'min_data_in_leaf': 20, # 'min_child_samples': 45,
              'objective': 'regression', 'boosting': 'gbdt',
              'feature_fraction': 0.723, 'bagging_freq': 5,
              'bagging_fraction': 0.723, 'bagging_seed': 19960101,
              'metric': 'mse', 'lambda_l1': 0.01, 'verbosity': -1}
pred_lgb, pred_trn_lgb = lgb_cv(
    df_trn, df_tst, params_lgb, fit_params,
    cat_features, df_trn.columns.tolist()[1:-1], 4, 19960101)

# ====== xgb ==================================================================
fit_params = {'num_boost_round': 10800,
              'verbose_eval': 360,
              'early_stopping_rounds': 360}
params_xgb = {'eta': 0.005, 'max_depth': 7, 'subsample': 0.723,
              'booster': 'gbtree', 'colsample_bytree': 0.723,
              # 'reg_lambda': 0.01,
              #  'reg_alpha': 0.01, 'gamma':0.01,
              'objective': 'reg:linear', 'silent': True, 'nthread': 4}
pred_xgb, pred_trn_xgb = xgb_cv(
    df_trn, df_tst, params_xgb, fit_params,
    df_trn.columns.tolist()[1:-1], 4, 19960101)

pred_tst = pred_lgb.copy()
stack_trn = np.vstack([pred_trn_lgb['pred'], pred_trn_xgb['pred']]).transpose()
stack_tst = np.vstack([pred_lgb['pred'], pred_xgb['pred']]).transpose()

stack_folds = KFold(n_splits=4, random_state=19960101)
stack_oof = np.zeros(stack_trn.shape[0])
pred_tst['pred'] = np.zeros(stack_tst.shape[0])

for _fold, (trn_idx, val_idx) in enumerate(
        stack_folds.split(stack_trn, df_trn['收率'])):
    trn_x, trn_y = stack_trn[trn_idx], df_trn['收率'].iloc[trn_idx].values
    val_x, val_y = stack_trn[val_idx], df_trn['收率'].iloc[val_idx].values

    clf_3 = BayesianRidge()
    clf_3.fit(trn_x, trn_y)

    stack_oof[val_idx] = clf_3.predict(val_x)
    pred_tst['pred'] += clf_3.predict(stack_tst) / 4
print('\nThe Bagging Loss', mse(df_trn['收率'].values, stack_oof))
del val_x, val_y, trn_x, trn_y, trn_idx, val_idx, cat_features
del params_lgb, params_xgb, fit_params
del pred_trn_xgb, pred_trn_lgb
del _fold, clf_3, stack_oof, stack_folds, stack_trn, stack_tst
del KFold, RepeatedKFold, BayesianRidge, trn, tst
gc.collect()

# pred_tst1 = pred_tst.copy()
# pred_tst['pred'] = pred_tst1['pred']*0.5 + pred_tst2['pred']*0.5
pred_tst.to_csv(f'submit/submit_{datetime.now().strftime("%m%d%H%M")}.csv',
                index=False, header=None)
