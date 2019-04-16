# -*- coding: utf-8 -*-
"""
@author: ATCG, Jan 7th, 2019
"""

import os
import pandas as pd
import numpy as np
import warnings
import gc
import xgboost as xgb
from datetime import datetime
import lightgbm as lgb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
warnings.simplefilter('ignore')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
gc.enable()
os.chdir('C:/Users/Franc/Desktop/Dir/Competitions/# DigitalManufacture')

cal_diff = lambda df1, df2: np.mean(np.square(df1.iloc[:, 1]-df2.iloc[:, 1]))
cal_corr = lambda df, _f: df.iloc[:, 1:].corrwith(df[_f]).abs().sort_values()

df_trn = pd.read_csv('data/jinnan_round1_train_20181227.csv', encoding='GB2312')
trn = pd.read_csv('data/jinnan_round1_train_20181227.csv', encoding='GB2312')
df_tst = pd.read_csv('data/jinnan_round1_testA_20181227.csv', encoding='GB2312')
tst = pd.read_csv('data/jinnan_round1_testA_20181227.csv', encoding='GB2312')

# ====== Abnormal Revise ======

df_trn.loc[(df_trn['A1'] == 200) & (df_trn['A3'] == 405), 'A1'] = 300
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
df_target = df_trn['收率']
del df_trn['收率']
df_trn_tst = df_trn.append(df_tst, ignore_index=False).reset_index(drop=True)
for _df in [df_trn, df_tst, df_trn_tst]:
    _df['A3'] = _df['A3'].fillna(405)
gc.collect()

# ====== TimeIndex to Hour ====================================================
cols_timer = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
for _df in [df_trn_tst, df_trn, df_tst]:
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
for _df in [df_trn_tst, df_trn, df_tst]:
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


# ====== Feature Engineering ===================================================
def feature_engineer_temperature(data):
    raw = data.copy()
    df = pd.DataFrame(raw['样本id'])
    # 加热过程
    df['P1_S1_A6_0C'] = raw['A6']  # 容器初始温度
    df['P1_S2_A8_1C'] = raw['A8']  # 首次测温温度
    df['P1_S3_A10_2C'] = raw['A10']  # 准备水解温度
    df['P1_C1_C0_D'] = raw['A8'] - raw['A6']  # 测温温差
    df['P1_C2_C0_D'] = raw['A10'] - raw['A6']  # 初次沸腾温差

    # 水解过程
    df['P2_S1_A12_3C'] = raw['A12']  # 水解开始温度
    df['P2_S2_A15_4C'] = raw['A15']  # 水解过程测温温度
    df['P2_S3_A17_5C'] = raw['A17']  # 水解结束温度
    df['P2_C3_C0_D'] = raw['A12'] - raw['A6']  # 水解开始与初始温度温差
    df['P2_C3_C2_D'] = raw['A12'] - raw['A10']  # 水解开始前恒温温差
    df['P2_C4_C3_D'] = raw['A15'] - raw['A12']  # 水解过程中途温差
    df['P2_C5_C4_D'] = raw['A17'] - raw['A15']  # 水解结束中途温差
    df['P2_C5_C3_KD'] = raw['A17'] - raw['A12']  # 水解起止温差

    # 脱色过程
    df['P3_S2_A25_7C'] = raw['A25']  # 脱色保温开始温度
    df['P3_S3_A27_8C'] = raw['A27']  # 脱色保温结束温度
    df['P3_C7_C5_D'] = raw['A25'] - raw['A17']  # 降温温差
    df['P3_C8_C7_KD'] = raw['A27'] - raw['A25']  # 保温温差

    # 结晶过程
    df['P4_S2_B6_11C'] = raw['B6']  # 结晶开始温度
    df['P4_S3_B8_12C'] = raw['B8']  # 结晶结束温度
    df['P4_C11_C8_D'] = raw['B6'] - raw['A27']  # 脱色结束到结晶温差
    df['P4_C12_C11_KD'] = raw['B8'] - raw['B6']  # 结晶温差

    # 统计特征
    _funcs = ['mean', 'std', 'max', 'min', 'sum']
    for _func in _funcs:
        # 沸腾过程温度
        df[f'P2_C2-C5_{_func}'] = raw[['A10', 'A12', 'A15', 'A17']].\
            agg(_func, axis=1)
        # 水解过程温度
        df[f'P2_C3-C5_{_func}'] = raw[['A12', 'A15', 'A17']].\
            agg(_func, axis=1)
        # # 沸腾过程温差
        # df[f'P2_D3-D5_{_func}'] = \
        #     df[[f'P2_C{i}_C{i-1}_D' for i in range(3, 6)]].agg(_func, axis=1)
        # 沸腾过程绝对温差
        df[f'P2_D3-D5_{_func}'] = \
            df[[f'P2_C{i}_C{i-1}_D' for i in range(3, 6)]].\
                abs().agg(_func, axis=1)
        # 关键过程温差
        # df[f'P2_C1-C12_KD_{_func}'] = \
        #     df[[_f for _f in df.columns if _f.endswith('KD')]].agg([_func])
        # 关键过程绝对温差
        df[f'P2_C1-C12_KD_ABS_{_func}'] = \
            df[[_f for _f in df.columns if _f.endswith('KD')]].\
                abs().agg(_func, axis=1)
        # 所有过程温差
        # df[f'P2_C1-C12_D_{_func}'] = \
        #     df[[_f for _f in df.columns if _f.endswith('D')]].agg([_func])
        # 所有过程绝对温差
        df[f'P2_C1-C12_D_{_func}'] = \
            df[[_f for _f in df.columns if _f.endswith('D')]].\
                abs().agg(_func, axis=1)
        # 大温差绝对温差
        df[f'P2_LARGE_KD_{_func}'] = \
            df[['P2_C3_C0_D', 'P3_C7_C5_D', 'P4_C12_C11_KD']].\
                abs().agg(_func, axis=1)
    return df.set_index('样本id')


df_temperature = feature_engineer_temperature(df_trn_tst)


def feature_engineer_duration(data):
    raw = data.copy()
    df = pd.DataFrame(raw['样本id'])
    # 加热过程
    df['P1_S1_A5_0T'] = raw['A5_t']  # 初始时刻
    df['P1_S2_A9_2T'] = raw['A9_t']  # 初始时刻
    df['P1_T1_T0_D'] = duration_outer(raw['A7_t'], raw['A5_t'])
    # 初次测温时间差
    df['P1_T2_T1_D'] = duration_outer(raw['A9_t'], raw['A7_t'])
    # 二次测温时间差
    df['P1_T2_T0_K_D'] = duration_outer(raw['A9_t'], raw['A5_t'])
    # 开始加热至沸腾时间差

    # 水解过程
    df['P2_S1_A11_3T'] = raw['A11_t']  # 水解开始时刻
    df['P2_S1_A16_5T'] = raw['A16_t']  # 水解结束时刻

    df['P2_T3_T0_K_D'] = duration_outer(raw['A11_t'], raw['A5_t'])
    # 开始加热至投料时间差
    df['P2_T3_T2_K_D'] = duration_outer(raw['A11_t'], raw['A9_t'])
    # 恒温至投料投料时间差
    # df['P2_T4_T3_D'] = raw['A14_t'] - raw['A11_t']  # 水解初次测温时间差
    # df['P2_T5_T4_D'] = raw['A16_t'] - raw['A14_t']  # 水解结束时间差
    df['P2_T5_T3_K_D'] = duration_outer(raw['A16_t'], raw['A11_t'])
    # 水解时间差

    # 脱色过程
    df['P3_S1_A20_6T'] = raw['A20_at']  # 中和开始时刻
    df['P3_S2_A25_7T'] = raw['A24_t']  # 保温时刻

    df['P3_T6_T5_K_D'] = duration_outer(raw['A20_at'], raw['A16_t'])
    # 水解结束至中和间歇时间
    df['P3_T6_T6_K_D'] = duration_outer(raw['A20_bt'], raw['A20_at'])
    # 酸碱度中和时间
    df['P3_T7_T6_D'] = duration_outer(raw['A24_t'], raw['A20_bt'])
    # 中和结束至脱色间歇时间
    df['P3_T8_T7_K_D'] = duration_outer(raw['A26_t'], raw['A24_t'])
    # 脱色保温时间
    df['P3_T9_T8_D'] = duration_outer(raw['A28_at'], raw['A26_t'])
    # 脱色至抽滤间歇时间
    df['P3_T9_T9_K_D'] = duration_outer(raw['A28_bt'], raw['A28_at'])
    # 抽滤时间
    df['P3_T9_T5_1D'] = duration_outer(raw['A28_bt'], raw['A16_t'])
    df['P3_T9_T6_2D'] = duration_outer(raw['A28_bt'], raw['A20_at'])
    # 脱色总时间

    # 结晶过程
    df['P4_S1_B4_10T'] = raw['B4_at']  # 酸化开始时刻
    df['P4_S2_B5_11T'] = raw['B5_t']  # 结晶开始时刻
    df['P4_S3_B7_12T'] = raw['B7_t']  # 结晶结束时刻

    df['P4_T10_T9_D'] = duration_outer(raw['B4_at'], raw['A28_bt'])
    # 抽滤结束至酸化间歇时间
    df['P4_T10_T10_K_D'] = duration_outer(raw['B4_bt'], raw['B4_at'])
    # 酸化时间
    df['P4_T11_T10_K_D'] = duration_outer(raw['B5_t'], raw['B4_bt'])
    # 酸化至结晶间歇时间
    df['P4_T12_T11_K_D'] = duration_outer(raw['B7_t'], raw['B5_t'])
    # 自然结晶时间
    df['P4_T12_T9_1D'] = duration_outer(raw['B7_t'], raw['A28_bt'])
    df['P4_T12_T10_2D'] = duration_outer(raw['B7_t'], raw['B4_at'])
    # 结晶总时间

    # 甩滤过程
    df['P5_S1_B9_13T'] = raw['B9_at']  # 甩滤开始时刻
    df['P5_S3_B12_15T'] = np.where(
        raw['B11_bt'].isnull(),
        np.where(raw['B10_bt'].isnull(), raw['B9_bt'], raw['B10_bt']),
        raw['B11_bt'])  # 甩滤结束时刻
    df['P5_T13_T12_D'] = duration_outer(raw['B9_at'], raw['B7_t'])
    # 酸化结束至甩滤间歇时间
    df['P5_T13_T13_K_D'] = duration_outer(raw['B9_bt'], raw['B9_at'])
    # 基本甩滤时间
    df['P5_T14_T13_D'] = duration_outer(raw['B10_at'], raw['B9_bt'])
    # 基本甩滤至补充甩滤1间歇时间
    df['P5_T14_T14_K_D'] = duration_outer(raw['B10_bt'], raw['B10_at'])
    # 补充甩滤1时间
    df['P5_T15_T14_D'] = duration_outer(raw['B11_at'], raw['B10_bt'])
    # 补充甩滤1至补充甩滤2间歇时间
    df['P5_T15_T13_K_D'] = duration_outer(raw['B11_bt'], raw['B11_at'])
    # 补充甩滤2时间
    df['P5_T15_T13_1D'] = \
        df[['P5_T13_T13_K_D', 'P5_T14_T14_K_D', 'P5_T13_T13_K_D']].sum(axis=1)
    df['P5_T15_T12_2D'] = duration_outer(
        df['P5_S3_B12_15T'], df['P4_S3_B7_12T'])
    df['P5_T15_T12_3D'] = duration_outer(
        df['P5_S3_B12_15T'], df['P5_S1_B9_13T'])
    # 总甩滤时间

    df['P5_T15_T1_4D'] = \
        df[['P5_T15_T12_2D', 'P4_T12_T9_1D', 'P3_T9_T5_1D',
            'P2_T3_T0_K_D', 'P2_T5_T3_K_D']].sum(axis=1)
    _funcs = ['mean', 'std', 'max', 'min', 'sum']
    for _func in _funcs:
        df[f'P5__D_{_func}'] = \
            df[[_f for _f in df.columns if _f.endswith('_D')]].\
                abs().agg(_func, axis=1)
        df[f'P5_K_D_{_func}'] = \
            df[[_f for _f in df.columns if _f.endswith('_K_D')]]. \
                abs().agg(_func, axis=1)
        df[f'P5__D_{_func}'] = \
            df[[_f for _f in df.columns if _f.endswith('D')]]. \
                abs().agg(_func, axis=1)
    # 总流程时长
    return df.set_index('样本id')


df_duration = feature_engineer_duration(df_trn_tst)


def feature_engineer_materials(data, na_value=405):
    raw = data.copy()
    df = pd.DataFrame(raw['样本id'])
    # 耗水
    df['P2_W_1M'] = raw['A4']
    df['P2_W_2M'] = raw['A19']
    df['P5_W_3M'] = raw['B12']
    # 耗盐酸
    df['P3_H_1M'] = raw['A21']
    df['P4_H_2M'] = raw['B1']
    # 氢氧化钠
    df['P2_N_1M'] = raw['A3'].fillna(na_value)
    # 4-氰基吡啶
    df['P2_C_1M'] = raw['A1']

    df['P5_W_1M'] = df['P2_W_1M'] + df['P2_W_2M']
    df['P5_W_2M'] = df['P2_W_2M'] + df['P5_W_3M']
    # df['P5_W_4M'] = df['P2_W_1M'] + df['P5_W_3M']
    df['P5_W_3M'] = df['P2_W_1M'] + df['P2_W_2M'] + df['P5_W_3M']
    df['P5_H_1M'] = df['P3_H_1M'] + df['P4_H_2M']
    # 理论产出
    df['P5_O_1M'] = raw['B14']
    df['P5_O_3M'] = np.where(raw['B14'] <= 360, raw['B14'] + 100, raw['B14'])
    df['P5_O_4M'] = np.where(raw['A3'] <= 360, raw['B14'] + 100, raw['B14'])
    df['P5_O_5M'] = raw['B14'].replace(418, 420).replace(405, 400).\
        replace(395, 390).replace(392, 390).replace(387, 385).\
        replace(380, 385).replace(370, 380).replace(360, 380).\
        replace(350, 385).replace(340, 285).replace(290, 280).\
        replace(260, 280).replace(256, 280)
    _fs = [_f for _f in df.columns if _f.endswith('M')][::-1]
    for i in range(len(_fs)):
        _f, _sub_fs = _fs[i], _fs[(i+1):]
        for _f_div in _sub_fs:
            df[f'{_f}_{_f_div}_R'] = df[_f] / df[_f_div]
    return df.set_index('样本id')


df_materials = feature_engineer_materials(df_trn_tst)


def feature_engineer_interact(data):
    raw = data.copy()
    df = pd.DataFrame(raw['样本id'])
    df['P5_NOT_NUM_N'] = raw.iloc[:, 1:-1].notnull().sum(axis=1)
    df['P5_PH_1N'] = raw['A22']
    df['P5_PH_2N'] = raw['A23']
    df['P5_PH_2N'] = raw['B2']
    df['P5_A7_1N'] = raw['A7_t'].isnull().astype(int)
    df['P5_O_2M'] = (raw['B14'] <= 360).astype(int)
    return df.set_index('样本id')


df_interact = feature_engineer_interact(df_trn_tst)


df_feature = pd.concat(
    [df_materials, df_duration, df_temperature, df_interact], axis=1).\
    reset_index()
del df_materials, df_duration, df_temperature, df_interact
gc.collect()

def train_duplicated_median(data):
    df, len_df = data.head(1), len(data)
    df['收率'] = data['收率'].median()
    return df


df_trn = df_feature.iloc[:len(trn)].reset_index(drop=True)
df_trn['收率'] = df_target
df_tst = df_feature.iloc[len(trn):].reset_index(drop=True)
df_tst['收率'] = np.nan
# df_trn = df_trn.fillna(999).\
#     groupby(df_trn.columns.tolist()[1:-1], as_index=False).\
#     apply(train_duplicated_median).\
#     replace(999, np.nan).reset_index(drop=True)

df_trn = df_trn.query('收率 > 0.8701').reset_index(drop=True)# 8671
df_trn = df_trn.query('收率 < 0.9861').reset_index(drop=True)# 9861

# df_trn = df_trn.query('收率 > 0.8671').reset_index(drop=True)# 8671
# df_trn = df_trn.query('收率 < 0.9861').reset_index(drop=True)# 9861

df_trn['recall_1'] = (df_trn['收率'] < 0.8930).astype('int8')
df_trn['recall_2'] = ((0.8930 <= df_trn['收率']) & \
                      (df_trn['收率'] < 0.9310)).astype('int8')
df_trn['recall_3'] = ((0.9310 <= df_trn['收率']) & \
                      (df_trn['收率'] < 0.9510)).astype('int8')
df_trn['recall_4'] = (df_trn['收率'] >= 0.9510).astype('int8')


# df_trn['recall_1'] = (df_trn['收率'] < 0.8908).astype('int8')
# df_trn['recall_2'] = ((0.8908 <= df_trn['收率']) & \
#                       (df_trn['收率'] < 0.9299)).astype('int8')
# df_trn['recall_3'] = ((0.9010 <= df_trn['收率']) & \
#                       (df_trn['收率'] < 0.9299)).astype('int8')
# df_trn['recall_4'] = ((0.9299 <= df_trn['收率']) & \
#                       (df_trn['收率'] < 0.9460)).astype('int8')
# df_trn['recall_5'] = (df_trn['收率'] >= 0.9460).astype('int8')


for _f in df_tst.columns.tolist()[1:-1]:
    vc = df_trn[_f].value_counts(normalize=True)
    if len(vc) <= 15 and vc.iloc[0] <= 0.5:
        recall_mean = df_trn.groupby([_f])['收率'].mean()
        for _df in [df_trn, df_tst]:
            _idx_col = _df.columns.tolist().index(_f)
            _df.insert(_idx_col + 1, f'{_f}_mean', _df[_f].map(recall_mean))
    corr = np.abs(np.corrcoef(df_trn[_f], df_trn['收率']))[0, 1]
    if len(vc) <= 25 and corr >= 0.32:
        for _col in [f'recall_{i}' for i in range(1, 5)]:
            recall_count = df_trn.groupby([_f])[_col].mean()
            for _df in [df_trn, df_tst]:
                _idx_col = _df.columns.tolist().index(_f)
                _df.insert(_idx_col + 1, f'{_f}_{_col}_count',
                           _df[_f].map(recall_count))
for _f in [f'recall_{i}' for i in range(1, 5)]:
    del df_trn[_f]
for _df in [df_trn, df_tst]:
    for _f1 in ['P5_O_1M', 'P5_O_3M', 'P5_O_5M',
                'P5_O_4M', 'P5_W_3M', 'P4_S2_B6_11C']:
        for _f2 in ['P2_C_1M', 'P2_W_1M', 'P5_W_1M', 'P1_S3_A10_2C']:
            vm = df_trn.groupby([_f2, _f1])['收率'].mean().reset_index()
            _idx_col = _df.shape[1] - 1
            vo = _df[[_f2, _f1]].merge(vm, on=[_f2, _f1])
            _df.insert(_idx_col, f'P5_{_f2}_{_f1}_M', vo['收率'])
del _f, vc, corr, recall_count, _idx_col, _df, _col, vm, vo  #, recall_mean,
for _df in [df_trn, df_tst]:
    _df.insert(1, 'id', _df['样本id'].str.split('_').str[1].astype(int))
del _df, _f2, _f1
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
        # print(f'\nFold_{fold_id} Training ================================\n')
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
    print('\nCV LOSS:', mse(train.Pred['true'], train.Pred['pred']), '\n')
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
        # print(f'\nFold_{fold_id} Training ================================\n')
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
    print('\nCV LOSS:', mse(train.Pred['true'], train.Pred['pred']))
    return test.Pred, train.Pred


cat_features = [df_trn.columns.tolist()[1:-1].index(_f) for
                _f in df_trn.columns if _f.endswith('_all')]

# ====== lgb ============================================================
fit_params = {'num_boost_round': 10800, 'verbose_eval': False,
              'early_stopping_rounds': 360}
params_lgb = {'num_leaves': 120, 'max_depth': 7, 'learning_rate': 0.005,
              'min_data_in_leaf': 25,  # 'min_child_samples': 45,
              'objective': 'regression', 'boosting': 'gbdt',
              'feature_fraction': 0.723, 'bagging_freq': 5,
              'bagging_fraction': 0.723, 'bagging_seed': 19960101,
              'metric': 'mse', 'lambda_l1': 0.025, 'verbosity': -1}
pred_lgb, pred_trn_lgb = lgb_cv(
    df_trn, df_tst, params_lgb, fit_params,
    cat_features, df_trn.columns.tolist()[1:-1], 6, 19960101)

# ====== xgb ==================================================================
fit_params = {'num_boost_round': 10800,
              'verbose_eval': False,
              'early_stopping_rounds': 360}
params_xgb = {'eta': 0.0025, 'max_depth': 7, 'subsample': 0.723,
              'booster': 'gbtree', 'colsample_bytree': 0.723,
              # 'reg_lambda': 0.025,
              #  'reg_alpha': 0.01, 'gamma':0.01,
              'objective': 'reg:linear', 'silent': True, 'nthread': 4}
pred_xgb, pred_trn_xgb = xgb_cv(
    df_trn, df_tst, params_xgb, fit_params,
    df_trn.columns.tolist()[1:-1], 6, 19960101)

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
del _fold, clf_3, stack_folds, stack_trn, stack_tst
del KFold, RepeatedKFold, BayesianRidge, trn, tst
gc.collect()
# df_trn_pred_2 = stack_oof.copy()
# pred_tst1 = pred_tst.copy()
# pred_tst['pred'] = pred_tst1['pred']*0.5 + pred_tst2['pred']*0.5
pred_tst.iloc[:, 1] = np.round(pred_tst.iloc[:, 1], 3)
pred_tst.to_csv(f'submit/submit_{datetime.now().strftime("%m%d%H%M")}.csv',
                index=False, header=None)
