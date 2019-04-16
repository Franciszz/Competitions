# -*- coding: utf-8 -*-
"""
@author: _ATCG__, Feb 19th, 2019
"""

# ==============================================================================
# ====== Import Basic Module ===================================================
# ==============================================================================
import pandas as pd
import numpy as np

# import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse

import time
import warnings
# import logging
import tqdm
# from contextlib import contextmanager

warnings.simplefilter('ignore')
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     format='[%(levelname)s] %(asctime)s %(filename)s: %(lineno)d: %(message)s',
#     datefmt='%Y-%m-%d: %H:%M:%S',
#     level=logging.DEBUG)
#
#
# @contextmanager
# def timer(title):
#     t0 = time.time()
#     yield
#     logger.info(f'{title} - done in {time.time()-t0}')


def train_abnormal_revise(data):
    df_trn = data.copy()
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
    df_trn['A25'] = df_trn['A25'].replace('1900/3/10 0:00', 70).\
        fillna(70).astype(int)
    df_trn['A26'] = df_trn['A26'].replace('1900/3/13 0:00', '13:00:00')
    df_trn['B1'] = df_trn['B1'].replace(3.5, np.nan)
    df_trn['B4'] = df_trn['B4'].replace('15:00-1600', '15:00-16:00')
    df_trn['B4'] = df_trn['B4'].replace('18:00-17:00', '16:00-17:00')
    df_trn['B4'] = df_trn['B4'].replace('19:-20:05', '19:05-20:05')
    df_trn['B9'] = df_trn['B9'].replace('23:00-7:30', '23:00-00:30')
    df_trn['B14'] = df_trn['B14'].replace(40, 400)
    df_trn['A5'] = df_trn['A5'].replace('1900/1/22 0:00', '22:00:00')
    df_trn['A7'] = df_trn['A7'].replace('0:50:00', '21:50:00')
    df_trn['B14'] = df_trn['B14'].replace(785, 385)
    return df_trn


def train_abnormal_adjust(data):
    df_trn = data.copy()
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
    df_trn.loc[df_trn['样本id'] == 'sample_919', 'A9'] = '19:50:00'
    df_trn.loc[df_trn['样本id'] == 'sample_566', 'A5'] = '18:00:00'
    df_trn.loc[df_trn['样本id'] == 'sample_40', 'A20'] = '5:00-5:30'
    df_trn.loc[df_trn['样本id'] == 'sample_531', 'B5'] = '1:00'
    df_trn.loc[df_trn['样本id'].isin(
        ['sample_35', 'sample_1285', 'sample_1057', 'sample_1695',
         'sample_1824', 'sample_1249', 'sample_1181',
         'sample_1832', 'sample_1773', 'sample_1827',
         'sample_1818', 'sample_695', 'sample_1742',
         'sample_1767', 'sample_698', 'sample_152',
         'sample_1779', 'sample_1208']), 'B14'] = 460
    df_trn.loc[df_trn['样本id'].isin(
        ['sample_1916', 'sample_1631', 'sample_1637', 'sample_1664',
         'sample_1628', 'sample_1681', 'sample_1690', 'sample_545',
         'sample_1757', 'sample_117', 'sample_1912', 'sample_1903'
         ]), 'B14'] = 440
    df_trn.loc[df_trn['样本id'].isin(
        ['sample_1202', 'sample_1764', 'sample_667', 'sample_1754',
         'sample_1835', 'sample_1237', 'sample_143', 'sample_1596',
         'sample_1744', 'sample_1708', 'sample_1787', 'sample_1197',
         'sample_1611', 'sample_1792', 'sample_93', 'sample_1241',
         'sample_1699', 'sample_650']), 'B14'] = 420
    df_trn.loc[df_trn['样本id'].isin(
        ['sample_793', 'sample_1972', 'sample_1072',
         'sample_1687', 'sample_1695', 'sample84',
         'sample_78', 'sample_531', 'sample_1072']), 'B14'] = 400
    df_trn.loc[df_trn['样本id'].isin(
        ['sample_79', 'sample_109']), 'B14'] = 385
    return df_trn


# ==============================================================================
# ====== Time Transformation Function ==========================================
# ==============================================================================
def time_to_min(x):
    if x is np.nan or type(x) is float:
        return np.nan
    else:
        x = x.replace(';', ':').replace('；', ':')
        x = x.replace('::', ':').replace('"', ':')
        h, m = x.split(':')[:2]
        h = 0 if not h else h
        m = 0 if not m else m
        return int(h)*60 + int(m)


def duration_outer(series1, series2):
    duration = series1 - series2
    duration = np.where(duration < 0, duration + 24*60, duration)
    duration = np.where(duration > 12*60, 24*60 - duration, duration)
    duration = np.where(duration > 6*60, 12*60 - duration, duration)
    return duration


# ==============================================================================
# ====== 特征工程构建：时差，温度，材料以及交叉 ======================================
# ==============================================================================
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
    _funcs = ['mean', 'std', 'sum']
    for _func in _funcs:
        df[f'P2_C2-C5_{_func}'] = raw[['A10', 'A12', 'A15', 'A17']].\
            agg(_func, axis=1)  # 沸腾过程温度
        df[f'P2_D3-D5_{_func}'] = \
            df[[f'P2_C{i}_C{i-1}_D' for i in range(3, 6)]].\
                abs().agg(_func, axis=1)  # 沸腾过程绝对温差
        df[f'P2_C1-C12_KD_ABS_{_func}'] = \
            df[[_f for _f in df.columns if _f.endswith('KD')]].\
                abs().agg(_func, axis=1)  # 关键过程绝对温差
        df[f'P2_C1-C12_D_{_func}'] = \
            df[[_f for _f in df.columns if _f.endswith('D')]].\
                abs().agg(_func, axis=1)  # 所有过程绝对温差
        df[f'P2_LARGE_KD_{_func}'] = \
            df[['P2_C3_C0_D', 'P3_C7_C5_D', 'P4_C12_C11_KD']].\
                abs().agg(_func, axis=1)  # 大温差绝对温差
    print('number of temperate relative features:', df.shape)
    return df.set_index('样本id')


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

    # 总流程时长
    df['P5_T15_T1_4D'] = \
        df[['P5_T15_T12_2D', 'P4_T12_T9_1D', 'P3_T9_T5_1D',
            'P2_T3_T0_K_D', 'P2_T5_T3_K_D']].sum(axis=1)
    _funcs = ['mean', 'std', 'sum']
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
    print('number of time relative features:', df.shape)
    return df.set_index('样本id')


def feature_engineer_materials(data, na_value=405):
    raw = data.copy()
    df = pd.DataFrame(raw['样本id'])
    # 耗水
    df['P2_W_1M'] = raw['A4']
    df['P2_W_2M'] = raw['A19']
    # 耗盐酸
    df['P3_H_1M'] = raw['A21'].fillna(50)
    df['P4_H_2M'] = raw['B1'].fillna(320)
    # 氢氧化钠
    df['P2_N_1M'] = raw['A3'].fillna(na_value)
    # 4-氰基吡啶
    df['P2_C_1M'] = raw['A1']

    df['P5_W_3M'] = raw['B12'].fillna(1200)
    df['P5_W_1M'] = df['P2_W_1M'] + df['P2_W_2M']
    df['P5_W_3M'] = df['P2_W_1M'] + df['P2_W_2M'] + df['P5_W_3M']
    df['P5_H_1M'] = df['P3_H_1M'] + df['P4_H_2M']
    df['P5_M_0M'] = raw['A1'] + df['P2_N_1M'] + df['P5_W_1M'] + df['P4_H_2M']
    df['P5_M_1M'] = df['P5_M_0M'] + df['P5_W_3M']
    df['P5_M_2M'] = df['P5_M_1M'] + df['P3_H_1M']
    # 理论产出
    df['P5_O_1M'] = raw['B14']
    df['P5_O_5M'] = raw['B14'].replace(418, 420).replace(405, 400).\
        replace(395, 390).replace(392, 390).replace(387, 380).\
        replace(385, 380).replace(370, 360).replace(360, 460).\
        replace(350, 450)
    _fs = [_f for _f in df.columns if _f.endswith('M')]
    for _f in _fs[:-2]:
        df[f'{_f}_P5_O_1M_R'] = df['P5_O_1M'] / df[_f]
        df[f'{_f}_P5_O_5M_R'] = df['P5_O_5M'] / df[_f]
    for i in range(len(_fs[:6])):
        _f, _sub_fs = _fs[i], _fs[(i+1):6]
        for _f_div in _sub_fs:
            df[f'{_f}_{_f_div}_R'] = df[_f] / df[_f_div]
    print('number of materials relative features:', df.shape)
    return df.set_index('样本id')


def feature_engineer_interact(data):
    raw = data.copy()
    df = pd.DataFrame(raw['样本id'])
    df['P5_NOT_NUM_N'] = raw.iloc[:, 1:-1].notnull().sum(axis=1)
    df['P5_PH_1N'] = raw['A22']
    df['P5_PH_2N'] = raw['A23']
    df['P5_PH_2N'] = raw['B2']
    df['P5_A7_1N'] = raw['A7_t'].isnull().astype(int)
    df['P5_O_2M'] = (raw['B14'] <= 360).astype(int)
    df['P5_1_3M'] = raw['B13']
    return df.set_index('样本id')


# ==============================================================================
# ====== 数据删重 ===============================================================
# ==============================================================================
def train_duplicated_choice(data):
    df, len_df = data.head(1), len(data)
    if len(data) == 2:
        df['收率'] = data['收率'].median()
    else:
        df['收率'] = data['收率'].mode().tolist()[0]
    return df


# ==============================================================================
# ====== Train and Valid =======================================================
# ==============================================================================
def make_train(train, test, *train_add):
    df_trn, df_tst = train.copy(), test.copy()
    if train_add is not None:
        for _df in train_add:
            df_trn = pd.concat([df_trn, _df], axis=0).reset_index(drop=True)
    df_target = df_trn['收率']
    del df_trn['收率']
    df_trn_tst = df_trn.append(df_tst, ignore_index=False).reset_index(
        drop=True)
    for _df in [df_trn, df_tst, df_trn_tst]:
        _df['A3'] = _df['A3'].fillna(405)
    cols_timer = [
        'A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
    for _df in [df_trn_tst, df_trn, df_tst]:
        _df.rename(columns={_col: _col + '_t' for _col in cols_timer},
                   inplace=True)
        for _col in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
            _idx_col = _df.columns.tolist().index(_col)
            _df.insert(_idx_col + 1, _col + '_at',
                       _df[_col].str.split('-').str[0])
            _df.insert(_idx_col + 2, _col + '_bt',
                       _df[_col].str.split('-').str[1])
            del _df[_col]
            cols_timer = cols_timer + [_col + '_at', _col + '_bt']
    cols_timer = list(filter(lambda x: x.endswith('t'), df_trn_tst.columns))
    for _df in [df_trn_tst, df_trn, df_tst]:
        for _col in cols_timer:
            _df[_col] = _df[_col].map(time_to_min)
    df_temperature = feature_engineer_temperature(df_trn_tst)
    df_duration = feature_engineer_duration(df_trn_tst)
    df_materials = feature_engineer_materials(df_trn_tst)
    df_interact = feature_engineer_interact(df_trn_tst)
    df_feature = pd.concat(
        [df_materials, df_duration, df_temperature, df_interact], axis=1). \
        reset_index()
    df_trn = df_feature.iloc[:len(df_trn)].reset_index(drop=True)
    df_trn['收率'] = df_target
    df_tst = df_feature.iloc[len(df_trn):].reset_index(drop=True)
    df_tst['收率'] = np.nan
    df_seed = df_trn.fillna(999). \
        groupby(df_trn.columns.tolist()[2:-1], as_index=False). \
        apply(train_duplicated_choice). \
        replace(999, np.nan).reset_index(drop=True)
    df_seed = df_seed.sample(frac=0.25)
    return df_trn, df_tst, df_seed


# ==============================================================================
# ====== Model Train ===========================================================
# ==============================================================================
def xgb_cv(train, test, params, fit_params, feature_names, nfold, seed):
    train_pred = pd.DataFrame({
        'id': train['样本id'],
        'true': train['收率'],
        'pred': np.zeros(len(train))})
    test_pred = pd.DataFrame({'id': test['样本id'], 'pred': np.zeros(len(test))})
    kfolder = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    xgb_tst = xgb.DMatrix(data=test[feature_names])
    for fold_id, (trn_idx, val_idx) in enumerate(kfolder.split(train['收率'])):
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
        train_pred.loc[val_idx, 'pred'] = val_pred
        test_pred['pred'] += xgb_reg.predict(
            xgb_tst, ntree_limit=xgb_reg.best_ntree_limit) / nfold
    print('\nCV LOSS:', mse(train_pred['true'], train_pred['pred']))
    return test_pred


# ==============================================================================
# ====== Model Predictio========================================================
# ==============================================================================
def make_prediction(train, test, sed):
    df_trn, df_tst, df_seed = train.copy(), test.copy(), sed.copy()
    trn, tst = df_trn.copy(), df_tst.copy()
    feature_name = trn.columns.tolist()[1:-1]
    pred_control = pd.DataFrame(index=df_tst['样本id'])
    for i in tqdm.tqdm(range(len(df_seed))):
        # print('\n',f'# ====== record {i}/{len(df_seed)} training ......', '\n')
        trn, tst = df_trn.copy(), df_tst.copy()
        trn.iloc[:, 2:-1] = \
            trn.iloc[:, 2:-1].values - df_seed.iloc[i, 2:-1].values
        tst.iloc[:, 2:-1] = \
            tst.iloc[:, 2:-1].values - df_seed.iloc[i, 2:-1].values
        trn, tst = trn.copy(), tst.copy()
        trn['recall_1'] = (trn['收率'] < 0.8930).astype('int8')
        trn['recall_2'] = ((0.8930 <= trn['收率']) &
                           (trn['收率'] < 0.9310)).astype('int8')
        trn['recall_3'] = ((0.9310 <= trn['收率']) &
                           (trn['收率'] < 0.9510)).astype('int8')
        trn['recall_4'] = (trn['收率'] >= 0.9510).astype('int8')
        for _f in tst.columns.tolist()[1:-1]:
            vc = trn[_f].value_counts(normalize=True)
            corr = np.abs(np.corrcoef(trn[_f], trn['收率']))[0, 1]
            if len(vc) <= 18 and corr >= 0.36 and vc.iloc[0] <= 0.75:
                for _col in [f'recall_{i}' for i in range(1, 5)]:
                    recall_count = trn.groupby([_f])[_col].mean()
                    for _df in [trn, tst]:
                        _idx_col = _df.columns.tolist().index(_f)
                        _df.insert(_idx_col + 1, f'{_f}_{_col}_count',
                                   _df[_f].map(recall_count))
        for _f in [f'recall_{i}' for i in range(1, 5)]:
            del trn[_f]
        pred_xgb = xgb_cv(
            trn, tst, params_xgb, fit_params,
            feature_name, 5, i)
        pred_control[f'ctr_{i}'] = pred_xgb['pred'].values
        pred_control['pred'] = pred_control.mean(axis=1)
    return pred_control['pred'].reset_index()


def main():
    print(f'''{    "-"*50}
    Train Data Input......
    {"-"*50}''')
    df_trn = pd.read_csv(
        'data/jinnan_round1_train_20181227.csv', encoding='GB2312'). \
        sort_values(['样本id']).reset_index(drop=True)

    df_trn_a = pd.read_csv(
        'data/jinnan_round1_testA_20181227.csv', encoding='GB2312'). \
        sort_values(['样本id']).reset_index(drop=True)
    df_ans_a = pd.read_csv(
        'data/jinnan_round1_ansA_20190125.csv', encoding='GB2312', header=None). \
        rename(columns={0: '样本id', 1: '收率'}). \
        sort_values(['样本id']).reset_index(drop=True)
    df_trn_a['收率'] = df_ans_a['收率'].values

    df_trn_b = pd.read_csv(
        'data/jinnan_round1_testB_20190121.csv', encoding='GB2312'). \
        sort_values(['样本id']).reset_index(drop=True)
    df_ans_b = pd.read_csv(
        'data/jinnan_round1_ansB_20190125.csv', encoding='GB2312', header=None). \
        rename(columns={0: '样本id', 1: '收率'}). \
        sort_values(['样本id']).reset_index(drop=True)
    df_trn_b['收率'] = df_ans_b['收率'].values

    df_trn_c = pd.read_csv(
        'data/jinnan_round1_test_20190201.csv', encoding='GB2312'). \
        sort_values(['样本id']).reset_index(drop=True)
    df_ans_c = pd.read_csv(
        'data/jinnan_round1_ans_20190201.csv', encoding='GB2312', header=None). \
        rename(columns={0: '样本id', 1: '收率'}). \
        sort_values(['样本id']).reset_index(drop=True)
    df_trn_c['收率'] = df_ans_c['收率'].values

    df_trn = pd.concat(
        [df_trn, df_trn_a, df_trn_b, df_trn_c], ignore_index=True, sort=True). \
        reindex(columns=df_trn_a.columns)
    print('the shape of train data is:', df_trn.shape)
    del df_trn_a, df_trn_b, df_trn_c, df_ans_a, df_ans_b, df_ans_c
    df_trn = train_abnormal_revise(df_trn).pipe(train_abnormal_adjust)
    df_trn = df_trn[(df_trn['收率'] >= 0.85) & (df_trn['收率'] <= 1)]
    df_trn = df_trn[(df_trn['B14'] <= 460) & (df_trn['B14'] >= 350)]. \
        reset_index(drop=True)
    print('the shape of train data changes into:', df_trn.shape)
    print('FuSai data input......')
    df_FuSai = pd.read_csv('data/FuSai.csv', encoding='GB2312')
    print('the shape of test data is:', df_FuSai.shape)
    print('Optimal data input......')
    df_opt = pd.read_csv('data/optimize.csv', encoding='GB2312')
    print('the shape of optimize data changes into:', df_opt.shape)
    df_tst = pd.concat([df_FuSai, df_opt], axis=0).reset_index(drop=True)
    print('make the train, test and control data......')
    df_trn, df_tst, df_seed = make_train(df_trn, df_tst)
    print('make prediction for FuSai and Optimize Data......')
    pred_xgb = make_prediction(df_trn, df_tst, df_seed)
    pred_FuSai = pred_xgb[pred_xgb['样本id'].isin(df_FuSai['样本id'])]
    pred_opt = pred_xgb[pred_xgb['样本id'].isin(df_opt['样本id'])]
    print('最优收率:', pred_opt['pred'])
    pred_FuSai.to_csv('submit_FuSai.csv', index=False, header=None)
    pred_opt.to_csv('submit_optimize.csv', index=False, header=None)


# ==============================================================================
# ====== 模型训练参数 ============================================================
# ==============================================================================
fit_params = {'num_boost_round': 10800,
              'verbose_eval': False,
              'early_stopping_rounds': 360}

params_xgb = {'eta': 0.01,
              'max_depth': 7,
              'subsample': 0.8,
              'booster': 'gbtree',
              'colsample_bytree': 0.8,
              'objective': 'reg:linear',
              'silent': True,
              'nthread': 4}

if __name__ == '__main__':
    main()
