# -*- coding: utf-8 -*-
"""
@author: ATCG, Jan 6th, 2019
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
import copy
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


# ====== Extract Target Value ================================================
df_target = df_trn['收率']
del df_trn['收率']
df_trn_tst = df_trn.append(df_tst, ignore_index=False)
df_trn_tst['A3'].fillna(570, inplace=True)
df_trn_tst['A7'] = df_trn_tst['A7'].notnull().astype('int8')
del df_trn_tst['A7'], df_trn_tst['A8']


# ====== Time Transform Functions ==========================================
def time_to_min(x):
    if x is np.nan:
        return np.nan
    else:
        x = x.replace(';', ':').replace('；', ':'). \
            replace('::', ':').replace('"', ':')
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


# ====== TimeIndex to Hour ====================================================
cols_timer = ['A5', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
for _df in [df_trn_tst]:
    _df.rename(columns={_col: _col + '_t' for _col in cols_timer}, inplace=True)
    for _col in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
        _idx_col = _df.columns.tolist().index(_col)
        _df.insert(_idx_col, _col + '_at', _df[_col].str.split('-').str[0])
        _df.insert(_idx_col + 2, _col + '_bt', _df[_col].str.split('-').str[1])
        del _df[_col]
        cols_timer = cols_timer + [_col + '_at', _col + '_bt']
cols_timer = list(filter(lambda x: x.endswith('t'), df_trn_tst.columns))
del _df, _col, _idx_col
gc.collect()


for _df in [df_trn_tst]:
    for _col in cols_timer:
        _df[_col] = _df[_col].map(time_to_min)
del _df, _col
gc.collect()


# ====== Manual Features ======================================================
# ====== About Materials


# ====== Time Difference ======================================================
def duration_cal_tran(data, cols_timer=cols_timer):
    df = data.copy()
    df_temp = data.copy()
    for _col_a, _col_b in zip(cols_timer[1:], cols_timer[:-1]):
        df_temp[_col_a] = np.where(
            df_temp[_col_a].isnull(), df_temp[_col_b], df_temp[_col_a])
    _idx_col = df.columns.tolist().index('B11_bt')
    process_end_time = np.where(
        df['B11_bt'].isnull(), df_temp['B11_at'], df['B11_at'])
    df.insert(_idx_col + 1, 'B11', process_end_time)
    for _col_a, _col_b in zip(cols_timer[::-1][:-1], cols_timer[::-1][1:]):
        df[_col_a] = duration_outer(df[_col_a], df_temp[_col_b])
    del _col_a, _col_b
    return df.rename(columns={'A5_t': 'A5'})


df_trn_tst = duration_cal_tran(df_trn_tst)


# ====== Drop Useless Columns ================================================
for _df in [df_trn_tst]:
    for _col in desc(df_tst).query('nunique==1')['variable']:
        del _df[_col]
del _col, _df, df_trn, df_tst
gc.collect()


def train_duplicated_median(data):
    df, len_df = data.head(1), len(data)
    df['收率'] = data['收率'].median()
    return df


df_trn = df_trn_tst.iloc[:len(trn)].reset_index(drop=True)
df_trn['收率'] = df_target
df_tst = df_trn_tst.iloc[len(trn):].reset_index(drop=True)
# df_trn = df_trn.fillna(999).\
#     groupby(df_trn.columns.tolist()[1:-1], as_index=False).\
#     apply(train_duplicated_median).\
#     replace(999, np.nan).reset_index(drop=True)
df_trn = df_trn.query('收率 > 0.86').reset_index(drop=True)
df_trn = df_trn.query('收率 < 0.996').reset_index(drop=True)

df_trn['recall_1'] = (df_trn['收率'] < 0.8908).astype('int8')
df_trn['recall_2'] = ((0.8908 <= df_trn['收率']) & \
                      (df_trn['收率'] < 0.9299)).astype('int8')
df_trn['recall_3'] = ((0.9010 <= df_trn['收率']) & \
                      (df_trn['收率'] < 0.9299)).astype('int8')
df_trn['recall_4'] = ((0.9299 <= df_trn['收率']) & \
                      (df_trn['收率'] < 0.9460)).astype('int8')
df_trn['recall_5'] = (df_trn['收率'] >= 0.9460).astype('int8')


for _f in df_tst.columns.tolist()[1:]:
    vc = df_trn[_f].value_counts(normalize=True)
    if ((len(vc) < 12) and (vc.iloc[0] < 0.6)):
        #or (_f in ['A1', 'A3', 'A6', 'A21', 'A22', 'A23']):
        recall_mean = df_trn.groupby([_f])['收率'].mean()
        for _df in [df_trn, df_tst]:
            _idx_col = _df.columns.tolist().index(_f)
            _df.insert(_idx_col + 1, f'{_f}_mean', _df[_f].map(recall_mean))
    corr = np.abs(np.corrcoef(df_trn[_f], df_trn['收率']))[0, 1]
    if corr >= 0.32:#or (_f in ['A1', 'A3', 'A6', 'A21', 'A22', 'A23']):
        for _col in [f'recall_{i}' for i in range(1, 6)]:
            recall_count = df_trn.groupby([_f])[_col].mean()
            for _df in [df_trn, df_tst]:
                _idx_col = _df.columns.tolist().index(_f)
                _df.insert(_idx_col + 1, f'{_f}_{_col}_count',
                           _df[_f].map(recall_count))
for _f in [f'recall_{i}' for i in range(1, 6)]:
    del df_trn[_f]
del _f, vc, corr, recall_mean, recall_count, _idx_col, _df, _col
for _df in [df_trn, df_tst]:
    _df.insert(1, 'id', _df['样本id'].str.split('_').str[1].astype(int))
del _df
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
fit_params = {'num_boost_round': 10000, 'verbose_eval': 500,
              'early_stopping_rounds': 500}
params_lgb = {'num_leaves': 120, 'max_depth': 7, 'learning_rate': 0.01,
              'min_data_in_leaf': 12, # 'min_child_samples': 45,
              'objective': 'regression', 'boosting': 'gbdt',
              'feature_fraction': 0.723, 'bagging_freq': 5,
              'bagging_fraction': 0.723, 'bagging_seed': 19950520,
              'metric': 'mse', 'lambda_l1': 0.1, 'verbosity': -1}
pred_lgb, pred_trn_lgb = lgb_cv(
    df_trn, df_tst, params_lgb, fit_params,
    cat_features, df_trn.columns.tolist()[1:-1], 6, 19950520)

# ====== xgb ==================================================================
fit_params = {'num_boost_round': 10000,
              'verbose_eval': 500,
              'early_stopping_rounds': 500}
params_xgb = {'eta': 0.005, 'max_depth': 7, 'subsample': 0.723,
              'booster': 'gbtree', 'colsample_bytree': 0.723,
              'reg_lambda': 0.1,
              #  'reg_alpha': 0.01, 'gamma':0.01,
              'objective': 'reg:linear', 'silent': True, 'nthread': 4}
pred_xgb, pred_trn_xgb = xgb_cv(
    df_trn, df_tst, params_xgb, fit_params,
    df_trn.columns.tolist()[1:-1], 6, 19950520)


# ====== Average Bagging ======
pred_tst = pred_lgb.copy()
stack_trn = np.vstack([pred_trn_lgb['pred'], pred_trn_xgb['pred']]).transpose()
stack_tst = np.vstack([pred_lgb['pred'], pred_xgb['pred']]).transpose()

stack_folds = KFold(n_splits=3, random_state=19950520)
stack_oof = np.zeros(stack_trn.shape[0])
pred_tst['pred'] = np.zeros(stack_tst.shape[0])

for _fold, (trn_idx, val_idx) in enumerate(
        stack_folds.split(stack_trn, df_trn['收率'])):
    trn_x, trn_y = stack_trn[trn_idx], df_trn['收率'].iloc[trn_idx].values
    val_x, val_y = stack_trn[val_idx], df_trn['收率'].iloc[val_idx].values

    clf_3 = BayesianRidge()
    clf_3.fit(trn_x, trn_y)

    stack_oof[val_idx] = clf_3.predict(val_x)
    pred_tst['pred'] += clf_3.predict(stack_tst) / 3
print('\nThe Bagging Loss', mse(df_trn['收率'].values, stack_oof))
del val_x, val_y, trn_x, trn_y, trn_idx, val_idx, cat_features
del params_lgb, params_xgb, fit_params
del pred_lgb, pred_xgb, pred_trn_lgb, pred_trn_xgb
del _fold, clf_3, stack_oof, stack_folds, stack_trn, stack_tst
del KFold, RepeatedKFold, BayesianRidge, cols_timer
gc.collect()
pred_tst2 = pred_tst.copy()
# pred_tst.to_csv(f'submit/submit_{datetime.now().strftime("%m%d%H%M")}.csv',
#                 index=False, header=None)
