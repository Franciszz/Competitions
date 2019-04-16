# -*- coding: utf-8 -*-
"""
Created on Monday Dec 31 23:59:59 2018

@author: Franc
"""
# In[]
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn.metrics import mean_squared_error as mse
import copy
warnings.simplefilter('ignore')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100
gc.enable()


# ====== Data Input ======
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
df_trn['B14'] = df_trn['B14'].replace(40, 400)
df_tst['A5'] = df_tst['A5'].replace('1900/1/22 0:00', '22:00:00')
df_tst['A7'] = df_tst['A7'].replace('0:50:00', '21:50:00')

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

df_trn.loc[df_trn['样本id'] == 'sample_1046', 'A28'] = '1:00-18:30'

df_trn.loc[df_trn['样本id'] == 'sample_1230', 'B5'] = '17:00:00'
df_trn.loc[df_trn['样本id'] == 'sample_97', 'B7'] = '1:00:00'
df_trn.loc[df_trn['样本id'] == 'sample_752', 'B9'] = '11:00-14:00'

df_tst.loc[df_tst['样本id'] == 'sample_919', 'A9'] = '19:50:00'


# ====== Description Function =======
def desc(data, *func):
    df = data.agg(['dtype', 'nunique', *func]).T
    df['absent'] = np.round(data.isnull().mean(axis=0)*100, 4)
    df = df.reset_index().rename(columns={'index': 'variable'})
    return df


# ====== Description =======
trn_desc = desc(df_trn, 'min', 'max')
tst_desc = desc(df_tst, 'min', 'max')
df_trn_tst = df_trn.append(df_tst).reindex(columns=df_trn.columns)


# ====== Time Relative Variables ======
def time_to_min(x):
    if x is np.nan:
        return np.nan
    else:
        x = x.replace(';', ':').replace('；', ':').\
            replace('::', ':').replace('"', ':')
        h, m = x.split(':')[:2]
        h = 0 if not h else h
        m = 0 if not m else m
        return int(h)*60 + int(m)


def duration_outer(series1, series2):
    duration = series1 - series2
    duration = np.where(duration < 0, duration + 1440, duration)
    duration = np.where(duration > 720, 720 - duration, duration)
    duration = np.where(duration > 360, 360 - duration, duration)
    return duration


def duration_inner(series, result=1):
    series = series.str.replace(';', ':').str.split('-')
    series1 = series.str[0].map(time_to_min)
    series2 = series.str[1].map(time_to_min)
    if result == 1:
        return duration_outer(series2, series1)
    elif result == 0:
        return series1
    else:
        return series2


# ====== timeclock to duration ======
cols_timer = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
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


# ====== train & test description ======
df_desc = desc(df_trn, 'min', 'max').assign(train=1).\
    append(desc(df_tst, 'min', 'max').assign(train=0)).\
    sort_values(['variable', 'train']).reset_index(drop=True)
del trn_desc, tst_desc
gc.collect()


def duration_cal_tran(data):
    df = data.copy()
    df_temp = data.copy()
    for _col_a, _col_b in zip(cols_timer[1:], cols_timer[:-1]):
        df_temp[_col_a] = np.where(
            df_temp[_col_a].isnull(), df_temp[_col_b], df_temp[_col_a])
    for _col_a, _col_b in zip(cols_timer[::-1][:-1], cols_timer[::-1][1:]):
        df[_col_a] = duration_outer(df[_col_a], df_temp[_col_b])
    del _col_a, _col_b
    return df.reindex(columns=df_trn_tst.columns)


# # ====== drop useless columns
df_trn_tst = duration_cal_tran(df_trn_tst).drop(
    ['A2', 'A3', 'A5_t', 'A13', 'A18', 'A23', 'B3', 'B13'], axis=1)
# df_trn_tst = df_trn_tst.drop(
#     ['A2', 'A3', 'A13', 'A18', 'A23', 'B3', 'B13'], axis=1)


# ====== train cv set up ======
def cv(train, test, model, fit_params, nfold, seed, knn=False):
    # model.FI = pd.DataFrame(index= train.columns.tolist()[1:-1])
    train.Pred = pd.DataFrame({
        'id': train['样本id'],
        'true': train['收率'],
        'pred': np.zeros(len(train))})
    test.Pred = pd.DataFrame({'id': test['样本id'], 'pred': np.zeros(len(test))})
    kfolder = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    for fold_id, (trn_idx, val_idx) in enumerate(kfolder.split(train['收率'])):
        trn_df, val_df = train.iloc[trn_idx], train.iloc[val_idx]
        trn_x, trn_y = trn_df.iloc[:, 1:-1], trn_df['收率']
        val_x, val_y = val_df.iloc[:, 1:-1], val_df['收率']
        print(f'\nFold_{fold_id}  ==========================================\n')
        if not knn:
            model.fit(trn_x, trn_y, **fit_params,
                      eval_set=[(trn_x, trn_y), (val_x, val_y)])
        else:
            model.fit(trn_x, trn_y, **fit_params)
        # model.FI[f'fold_{fold_id}'] = \
        #     model.feature_importances_ / model.feature_importances_.sum()
        val_pred = model.predict(val_x)
        train.Pred.loc[val_idx, 'pred'] = val_pred
        print(f'Loss of Fold_{fold_id}', mse(val_y, val_pred))
        test.Pred['pred'] += model.predict(test.iloc[:, 1:-1])/nfold
    print('\n\nCV LOSS:', mse(train.Pred['true'], train.Pred['pred']))
    return test.Pred


def train_duplicated_median(data):
    df, len_df = data.head(1), len(data)
    df['收率'] = data['收率'].median()
    return df


df_trn = df_trn_tst[:1396].reset_index(drop=True)
df_tst = df_trn_tst[1396:].reset_index(drop=True)
df_trn = df_trn.fillna(9999).\
    groupby(df_trn.columns.tolist()[1:-1], as_index=False).\
    apply(train_duplicated_median).\
    replace(9999, np.nan).reset_index(drop=True)
# outliers = train[train['收率'] < 0.7]
df_trn = df_trn.query('收率 > 0.85').reset_index(drop=True)
# df_trn = df_trn.query('收率 > 0.8670').reset_index(drop=True)
df_trn = df_trn.query('收率 < 1.00').reset_index(drop=True)


# # ====== xgb regressor ======
# fit_params = {
#     'early_stopping_rounds': 100, 'verbose': 100, 'eval_metric': 'rmse'}
# reg_xgb = xgb.XGBRegressor(
#     learning_rate=0.01, n_estimators=1000, max_depth=5,
#     objective='reg:linear', seed=2019, silent=False,
#     # colsample_bytree=0.054, colsample_bylevel=0.50, gamma=1.45,
#     subsample=0.95)
# pred_xgb = cv(df_trn, df_tst, reg_xgb, fit_params, 5, 19960101)


# ====== lgb regressor ======
# fit_params = {
#     'early_stopping_rounds': 100, 'verbose': 100, 'eval_metric': 'mse'}
# reg_lgb = lgb.LGBMRegressor(
#     n_estimators=1000, objective='regression', metric='mse',
#     learning_rate=0.025, num_leaves=26,
#     # num_leaves=120, learning_rate=0.1, reg_alpha=0.5,
#     # min_data_in_leaf=5, boosting='gbdt', silent=True,
#     # reg_alpha=1, reg_lambda=1,
#     # max_depth = 4, colsample_bytree=.9,
#     bagging_seed=19960101, subsample=1.0)
# pred_lgb = cv(df_trn, df_tst, reg_lgb, fit_params, 5, 19960101)
#

# # ====== knn regressor ======
# fit_params = {}
# reg_knn = knn(n_neighbors=7)
# pred_knn = cv(
#     df_trn.fillna(0), df_tst.fillna(0), reg_knn, fit_params, 5, 19960101, 1)


# ====== continuous float to label encoder ======
df_trn_tst = pd.concat([df_trn, df_tst], axis=0, ignore_index=False)
df_trn_tst_desc = desc(df_trn_tst)


def col_shrinkage(data, col, perct, suffix):
    df = data.copy()
    df[col].fillna(999, inplace=True)
    vc = df[col].value_counts(normalize=True).\
        sort_values(ascending=True).cumsum()
    rare = vc <= perct
    rare = set(rare.index[rare].values)
    _idx_col = df.columns.tolist().index(col)
    df.insert(_idx_col + 1, f'{col}_{suffix}',
              np.where(df[col].isin(rare), 998, df[col]))
    vc = df[f'{col}_{suffix}'].value_counts()
    df[f'{col}_{suffix}'] = df[f'{col}_{suffix}'].map(
        dict(zip(vc.index, range(1, df[f'{col}_{suffix}'].nunique()+1))))
    return df

for _col in df_trn_tst.columns.tolist().copy()[1:-1]:
    df_trn_tst = col_shrinkage(df_trn_tst, _col, 0.01, 'all')
#     df_trn_tst = col_shrinkage(df_trn_tst, _col, 0.05, 'rare')
# del _col

cate_columns = [f for f in df_trn_tst.columns if f not in ['样本id', '收率']]
for _df in [df_trn_tst]:
    for _f in cate_columns:
        _df[_f] = _df[_f].map(
            dict(zip(_df[_f].unique(), range(0, _df[_f].nunique()))))

df_trn = df_trn_tst.iloc[:df_trn.shape[0]]
df_tst = df_trn_tst.iloc[df_trn.shape[0]:]


df_trn['low_recall_1'] = (df_trn['收率'] < 0.8911).astype('int8')
df_trn['low_recall_2'] = ((0.8911 <= df_trn['收率']) &
                          (df_trn['收率'] < 0.9030)).astype('int8')
df_trn['low_recall_3'] = ((0.9030 <= df_trn['收率']) &
                         (df_trn['收率'] < 0.931)).astype('int8')
df_trn['low_recall_4'] = ((0.931 <= df_trn['收率']) &
                         (df_trn['收率'] < 0.970)).astype('int8')
df_trn['low_recall_5'] = (df_trn['收率'] >= 0.970).astype('int8')

for _f in cate_columns:
    for _col in [f'low_recall_{i}' for i in range(1, 6)]:
        order_label = df_trn.groupby([_f])[_col].mean()
        for df in [df_trn, df_tst]:
            df[f'{_f}_{_col}_mean'] = df['B14'].map(order_label)

# train.drop(li, axis=1, inplace=True)


# for _col in filter(lambda x: x.endswith('_rare'), df_trn.columns.tolist()):
#     for _f in [f'low_recall_{i}' for i in range(1, 6)]:
#         low_recall_mean = df_trn.groupby([_col])[_f].mean()
#         for _df in [df_trn, df_tst]:
#             _idx_col = _df.columns.tolist().index(_col)
#             _df.insert(_idx_col + 1, f'{_col}_{_f}_low_recall_count',
#                        _df['B14_all'].map(low_recall_mean))
#     # del _df[_col]

# for _col in ['A7_t', 'A10', 'A25', 'A26_t', 'B6', 'B12', 'B14']:
#     for _f in [f'low_recall_{i}' for i in range(1, 6)]:
#         low_recall_mean = df_trn.groupby([_col])[_f].mean()
#         for _df in [df_trn, df_tst]:
#             _idx_col = _df.columns.tolist().index(_col)
#             _df.insert(_idx_col + 1, f'{_col}_{_f}_count',
#                        _df[_col].map(low_recall_mean))
    # del df_trn[_col], _idx_col, low_recall_mean
for _f in [f'low_recall_{i}' for i in range(1, 6)]:
    del df_trn[_f]
del _df, _f, _col, df
gc.collect()
# ====== encode all


def cat_cv(train, test, params, fit_params,
           cat_features, feature_names, nfold, seed):
    train.Pred = pd.DataFrame({
        'id': train['样本id'],
        'true': train['收率'],
        'pred': np.zeros(len(train))})
    test.Pred = pd.DataFrame({'id': test['样本id'], 'pred': np.zeros(len(test))})
    kfolder = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    cat_tst = cat.Pool(
        data=test[feature_names],
        cat_features=cat_features,
        feature_names=feature_names)
    for fold_id, (trn_idx, val_idx) in enumerate(kfolder.split(train['收率'])):
        print(f'\nFold_{fold_id} Training ================================\n')
        cat_trn = cat.Pool(
            data=train.iloc[trn_idx][feature_names],
            label=train.iloc[trn_idx]['收率'],
            cat_features=cat_features,
            feature_names=feature_names)
        cat_val = cat.Pool(
            data=train.iloc[val_idx][feature_names],
            label=train.iloc[val_idx]['收率'],
            cat_features=cat_features,
            feature_names=feature_names)
        cat.train(params=params, pool=cat_trn, **fit_params, eval_set=cat_val)
        val_pred = cat.predict(train.iloc[val_idx][feature_names])
        train.Pred.loc[val_idx, 'pred'] = val_pred
        print(f'Fold_{fold_id}', mse(train.iloc[val_idx]['收率'], val_pred))
        test.Pred['pred'] += cat.predict(cat_tst) / nfold
    print('\n\nCV LOSS:', mse(train.Pred['true'], train.Pred['pred']))
    return test.Pred


def lgb_cv(train, test, params, fit_params,
           cat_features, feature_names, nfold, seed):
    train.Pred = pd.DataFrame({
        'id': train['样本id'],
        'true': train['收率'],
        'pred': np.zeros(len(train))})
    test.Pred = pd.DataFrame({'id': test['样本id'], 'pred': np.zeros(len(test))})
    kfolder = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    lgb_tst = lgb.Dataset(
        data=test[feature_names],
        categorical_feature=cat_features,
        feature_name=feature_names)
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


def xgb_cv(train, test, params, fit_params,
           cat_features, feature_names, nfold, seed):
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
# ====== lgb ======
fit_params = {'num_boost_round': 5000, 'verbose_eval': 200,
              'early_stopping_rounds': 200}
params_lgb = {'num_leaves': 120, 'max_depth': 7, 'learning_rate': 0.00723,
              'min_data_in_leaf': 12, # 'min_child_samples': 45,
              'objective': 'regression', 'boosting': 'gbdt',
              'feature_fraction': 0.7, 'bagging_freq': 5,
              'bagging_fraction': 0.7, 'bagging_seed': 19960101,
              'metric': 'mse', 'lambda_l1': 0.0015, 'verbosity': -1}
pred_lgb, pred_trn_lgb = lgb_cv(
    df_trn, df_tst, params_lgb, fit_params,
    cat_features, df_trn.columns.tolist()[1:-1], 5, 19960101)

# ====== xgb ======
fit_params = {'num_boost_round': 5000,
              'verbose_eval': 500,
              'early_stopping_rounds': 200}
params_xgb = {'eta': 0.00723, 'max_depth': 7, 'subsample': 0.7,
              'booster': 'gbtree', 'colsample_bytree': 0.7,
              # 'reg_lambda': 0.01, # 'reg_alpha': 0.01, 'gamma':0.01,
              'objective': 'reg:linear', 'silent': True, 'nthread': 4}
pred_xgb, pred_trn_xgb = xgb_cv(
    df_trn, df_tst, params_xgb, fit_params,
    cat_features, df_trn.columns.tolist()[1:-1], 5, 19960101)

# ====== cat ======
fit_params = {'num_boost_round': 5000, 'verbose': 100}
params_cat = {'learning_rate': 0.1, 'max_depth': 7, 'objective': 'RMSE',
              # 'l2_leaf_reg': 1, 'rsm': 0.9,
              # 'one_hot_max_size': 20,
              # 'subsample': 0.9,
              'custom_metric': 'RMSE', 'eval_metric': 'RMSE',
              'random_seed': 19960101, 'use_best_model': True}
#
# pred_cat = cat_cv(df_trn, df_tst, params_cat, fit_params,
#                   cat_features, df_trn.columns.tolist()[1:-1], 5, 19960101)


# # ====== average bagging ======
pred_tst = pred_lgb.copy()
pred_tst['pred'] = 0.5*pred_tst['pred'] + 0.5*pred_xgb['pred']
# # xgb_pred.to_csv('submit/xgb_lr0.01_depth5.csv',index=False, header=None)
# pred_tst.to_csv(
#     'submit/submit12_(0.87,0.99)_lgb_cut5_lr0.005_dep7_xgb_lr0.002_dep7_dart.csv',
#     index=False, header=None)
# # del pred_lgb, pred_xgb
# gc.collect()

# # ====== BayesianRidge Bagging ======
stack_trn = np.vstack([pred_trn_lgb['pred'], pred_trn_xgb['pred']]).transpose()
stack_tst = np.vstack([pred_lgb['pred'], pred_xgb['pred']]).transpose()

stack_folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
stack_oof = np.zeros(stack_trn.shape[0])
pred_tst['pred'] = np.zeros(stack_tst.shape[0])

for _fold, (trn_idx, val_idx) in enumerate(
        stack_folds.split(stack_trn, df_trn['收率'])):
    trn_x, trn_y = stack_trn[trn_idx], df_trn['收率'].iloc[trn_idx].values
    val_x, val_y = stack_trn[val_idx], df_trn['收率'].iloc[val_idx].values

    clf_3 = BayesianRidge()
    clf_3.fit(trn_x, trn_y)

    stack_oof[val_idx] = clf_3.predict(val_x)
    pred_tst['pred'] += clf_3.predict(stack_tst) / 10
print('\nThe Bagging Loss', mse(df_trn['收率'].values, stack_oof))
del val_x, val_y, trn_x, trn_y, trn_idx, val_idx, cat_features
del pred_lgb, pred_xgb, params_lgb, params_xgb, params_cat, fit_params
del _fold, clf_3, stack_oof, stack_folds, stack_trn, stack_tst
del knn, KFold, RepeatedKFold, BayesianRidge, cols_timer, df_desc
gc.collect()
