import tensorflow as tf
import pandas as pd
import numpy as np


def cal_acc(pred, true, mask):
    pred = tf.cast(pred, tf.int64)
    true = tf.cast(true, tf.int64)
    equal = tf.cast(tf.equal(pred, true), tf.float32)
    acc = tf.reduce_sum(equal * mask) / tf.reduce_sum(mask)
    return acc


def get_rnn_output(input_seq, cell='tf.nn.rnn_cell.GRUCell',
                   activation='tf.tanh', n_hidden_units=10,
                   name='rnn1'):
    batch_size = tf.shape(input_seq)[0]
    # seq_length = tf.shape(input_seq)[1]

    # rnn
    cell = eval(cell)(
        n_hidden_units, activation=eval(activation), name=f'{name}_cell')
    # cell = tf.contrib.rnn.LayerNormBasicLSTMCell(PARAS.n_hu, reuse=True)
    # initial_state
    # init_state = cell.zero_state(batch_size, dtype=tf.float32)
    init_state = tf.get_variable(f'{name}_init_state', [1, n_hidden_units])
    init_state = tf.tile(init_state, tf.stack([batch_size, 1]))
    outputs, state = tf.nn.dynamic_rnn(
        cell, input_seq, initial_state=init_state)
    return outputs, state


def get_fc_output(fc_input, activation='tf.nn.leaky_rule',
                  n_hidden_units_in=10, n_hidden_units_out=10,
                  name='fc1'):
    W = tf.get_variable(f'{name}_W', [n_hidden_units_in, n_hidden_units_out])
    b = tf.get_variable(f'{name}_b', [n_hidden_units_out])

    output = eval(activation)(tf.matmul(fc_input, W) + b)
    return output


def get_fc_with_bn_output(
        fc_input, activation='tf.nn.leaky_rule', target='train',
        n_hidden_units_in=10, n_hidden_units_out=10, name='fc1'):
    W = tf.get_variable(f'{name}_W', [n_hidden_units_in, n_hidden_units_out])
    b = tf.get_variable(f'{name}_b', [n_hidden_units_out])
    fc_output = tf.matmul(fc_input, W) + b
    train_mode = True if target == 'train' else False
    fc_output = tf.layers.batch_normalization(
        fc_output, name=f'{name}_bn', training=train_mode)
    fc_output = eval(activation)(fc_output)
    return fc_output


def get_fc_output_from_rnn_output(
        rnn_output, activation='tf.nn.leaky_relu',
        n_hidden_units=20, n_target=3, name='task'):
    batch_size = tf.shape(rnn_output)[0]
    seq_length = tf.shape(rnn_output)[1]

    W_1 = tf.get_variable(f'{name}_W1', [n_hidden_units, n_hidden_units])
    b_1 = tf.get_variable(f'{name}_b', [n_hidden_units])

    W_out = tf.get_variable(f'{name}_W_out', [n_hidden_units, n_target])
    b_out = tf.get_variable(f'{name}_b_out', [1, n_target])

    rnn_output = tf.reshape(rnn_output, [-1, n_hidden_units])
    rnn_output = eval(activation)(tf.matmul(rnn_output, W_1) + b_1)
    rnn_output = tf.matmul(rnn_output, W_out) + b_out

    fc_output = tf.reshape(
        rnn_output, tf.stack([batch_size, seq_length, n_target]))
    return fc_output


# ====== 提取周期因子 ==========================================================
def get_seasonal_ratio(data, feature):
    df = data.copy()
    df['weekday'] = df['date_dt'].dt.dayofweek
    df['week_id'] = 24 - df.groupby(['district_code', 'weekday'])['date_dt']. \
        rank(ascending=False).values
    df['week_avg'] = df.groupby(['district_code', 'week_id'])[feature]. \
        transform('mean').values
    df['ratio'] = df[feature] / df['week_avg'] - 1
    return df


def get_seasonal_coef(data, feature, mean_mode='mean',
                      rule_mode=False, ascend=False):
    df = data.copy()
    df['weekday'] = df['date_dt'].dt.dayofweek
    df['week_id'] = df.groupby(['district_code', 'weekday'])['date_dt']. \
        rank(ascending=ascend).values
    df['week_avg'] = df.groupby(['district_code', 'week_id'])[feature]. \
        transform('mean').values
    df['ratio'] = df[feature] / df['week_avg']
    if mean_mode == 'mean':
        cycle_rule = df.groupby(
            ['district_code', 'weekday'], as_index=False)\
            ['ratio'].mean().values
    else:
        cycle_rule = df.groupby(
            ['district_code', 'weekday'], as_index=False)\
            ['ratio'].median().values
    return cycle_rule if rule_mode else df


def get_rnn_data(data):
    rnn_data = data[['district_code', 'weekday', 'week_id', 'ratio']].copy()
    rnn_data['dist_weekday'] = rnn_data['district_code'] + '_' + \
        rnn_data['weekday'].astype(str)
    rnn_data = rnn_data.pivot(
        index='dist_weekday', columns='week_id', values='ratio')
    rnn_data = rnn_data.iloc[:, :-2].copy()
    rnn_data_feed = rnn_data.values.reshape(list(rnn_data.values.shape) + [1])
    return rnn_data, rnn_data_feed


# ============== rnn model set up =============================================
with tf.device('/cpu:0'):
    tf.reset_default_graph()
    tf.set_random_seed(19960101)

    # variable input
    lr = tf.placeholder(tf.float32, [], name='learning_rate')
    x = tf.placeholder(tf.float32, [None, None, 1])

    # batch_size & seqence length
    batch_size = tf.shape(x)[0]
    seq_length = tf.shape(x)[1]

    # rnn layer
    rnn_output, state = get_rnn_output(
        x, cell='tf.nn.rnn_cell.GRUCell', activation='tf.tanh',
        n_hidden_units=10)

    # full_connect layer
    prediction = get_fc_output_from_rnn_output(
        rnn_output, activation='tf.nn.leaky_relu',
        n_hidden_units=10, n_target=1)

    prediction_without_last = prediction[:, :-1, :]
    prediction_last = tf.reshape(prediction[:, -1, :], [-1])

    # 损失层
    label = x[:, 1:, :]
    obj = tf.losses.mean_squared_error(label, prediction_without_last)

    # 优化层
    optimizer = tf.train.AdamOptimizer(lr)
    train_step = optimizer.minimize(obj)


def train(n_batch, data_feed_train, data_feed_valid, lr_feed=0.01):
    # 训练损失
    for i in range(n_batch):
        _, _loss = sess.run(
            [train_step, obj], feed_dict={x: data_feed_train, lr: lr_feed})
    _, _loss = sess.run(
        [train_step, obj], feed_dict={x: data_feed_valid, lr: lr_feed})
    print(f'valid loss:', _loss)
    return _loss


def smooth_outlier_indi(
        data,
        features=['dwell', 'flow_in', 'flow_out'],
        days=7, threshold=2.5, adjust=0.5):
    df = data.copy().sort_values(by=['date_dt'])
    for _f in features:
        df[f'{_f}_avg'] = (df[_f].rolling(
            window=days, center=True).sum()-df[_f])/(days-1)
        df[f'{_f}_std'] = df[_f].rolling(
            window=days, center=True).std()
        df[f'{_f}_ratio'] = (df[_f]-df[f'{_f}_avg']) / df[f'{_f}_std']
        df[f'{_f}_sign'] = np.where(df[f'{_f}_ratio'] > 0, 1, -1)
        df[f'{_f}'] = np.where(
                df[f'{_f}_ratio'].abs() >= threshold,
                df[f'{_f}_avg']+df[f'{_f}_sign']*df[f'{_f}_std']*adjust,
                df[_f])
        df.drop([f'{_f}_avg', f'{_f}_std', f'{_f}_ratio',
                 f'{_f}_sign'], axis=1, inplace=True)
    return df


def smooth_outliers(
        data,
        bylist=['district_code'],
        features=['dwell', 'flow_in', 'flow_out'],
        days_list=[7, 15, 7, 15],
        threshold_list=[3, 2.5, 2.25, 2.5],
        rate_list=[1, 1.25, 1, 1.25]):
    df = data.copy().sort_values(['date_dt'])
    for (days, threshold, rate) in zip(days_list, threshold_list, rate_list):
        df = df.groupby(by=bylist).\
            apply(smooth_outlier_indi,
                  features=features,
                  days=days,
                  threshold=threshold,
                  adjust=rate)
    return df.reset_index(drop=True)


def get_city_dist_map(data, transform_date='2017-11-04'):
    df_lst = data[data['date_dt'] == transform_date]
    df_by_city = df_lst.groupby(
        by=['city_code'])['dwell'].sum().sort_values(ascending=False)
    dict_map = dict(zip(
        df_by_city.index, [f'C{i+1}' for i in range(len(df_by_city))]))
    df_lst['city_code'] = df_lst['city_code'].map(dict_map)
    fun_rank = lambda df: df.sort_values(
        by=['dwell']).assign(rank=[f'D{i+1}' for i in range(len(df))])
    df_by_dist = df_lst.groupby(by=['city_code']).apply(fun_rank)
    df_by_dist['rank'] = df_by_dist['city_code'] + '_' + df_by_dist['rank']
    dict_map = {**dict_map,
                **dict(df_by_dist[['district_code', 'rank']].values)}
    return dict_map

# ============ offline train ===================================================
flow = pd.read_csv('datalast/flow_train.csv', parse_dates=[0])
tran = pd.read_csv('datalast/flow_rate_train.csv', parse_dates=[0])
tran = tran[
    (tran['date_dt'] < '2017-08-19') | (tran['date_dt'] > '2017-08-23')]

code_dict = get_city_dist_map(flow, '2017-11-04')
for _df, _i in zip([flow, tran], [3, 3]):
    for _col in _df.columns[1:_i]:
        _df[_col] = _df[_col].replace(code_dict)
del _df, _col, _i

flow = flow.sort_values(['city_code', 'district_code', 'date_dt']). \
    reset_index(drop=True)
tran = tran.sort_values(['city_code', 'district_code', 'date_dt']). \
    reset_index(drop=True)

flow.iloc[:, 3:] = flow.iloc[:, 3:].values / tran.iloc[:, 3:].values
del tran

flow = smooth_outliers(flow)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for i in range(10):
#     train(200, RNN_data_feed_1, RNN_data_feed_2, 0.01)
# for i in range(10):
#     train(200, RNN_data_feed_1, RNN_data_feed_2, 0.001)
# train(100, RNN_data_feed_1, RNN_data_feed_2, 0.01)
# for i in range(20):
#     train(200, RNN_data_feed_1, RNN_data_feed_2, 0.001)
#


# ============ online train ====================================================
def get_coef_for_upper(flow_for_upper):
    RNN_data = get_seasonal_coef(flow_for_upper, 'dwell', rule_mode=False)
    RNN_data, RNN_data_feed = get_rnn_data(RNN_data)
    df_coef_upper = pd.DataFrame(index=RNN_data.index)
    for _f in ['dwell', 'flow_in', 'flow_out']:
        RNN_data = get_seasonal_coef(
            flow_for_upper, _f, rule_mode=False, ascend=True)
        RNN_data, RNN_data_feed = get_rnn_data(RNN_data)
        RNN_data_feed_1 = RNN_data_feed[:-50, :, :]
        RNN_data_feed_2 = RNN_data_feed[-50:, :, :]
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            train(200, RNN_data_feed, RNN_data_feed, 0.01)
        for i in range(10):
            train(200, RNN_data_feed, RNN_data_feed, 0.001)
        train(100, RNN_data_feed, RNN_data_feed, 0.01)
        for i in range(20):
            train(200, RNN_data_feed, RNN_data_feed, 0.001)
        # ========== make prediction
        pred = sess.run(prediction_last, feed_dict={x: RNN_data_feed})
        df_coef_upper.loc[:, f'{_f}_coef'] = pred
    df_coef_upper.reset_index(inplace=True)
    df_coef_upper['district_code'] = df_coef_upper['dist_weekday'].str[:-2]
    df_coef_upper['weekday'] = df_coef_upper['dist_weekday'].str[-1:]
    return df_coef_upper


def get_coef_for_lower(flow_for_lower):
    RNN_data = get_seasonal_coef(flow_for_lower, 'dwell', rule_mode=False)
    RNN_data, RNN_data_feed = get_rnn_data(RNN_data)
    df_coef_lower = pd.DataFrame(index=RNN_data.index)
    for _f in ['dwell', 'flow_in', 'flow_out']:
        RNN_data = get_seasonal_coef(flow_for_lower, _f, rule_mode=False)
        RNN_data, RNN_data_feed = get_rnn_data(RNN_data)
        RNN_data_feed_1 = RNN_data_feed[:-50, :, :]
        RNN_data_feed_2 = RNN_data_feed[-50:, :, :]
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            train(200, RNN_data_feed, RNN_data_feed, 0.01)
        for i in range(10):
            train(200, RNN_data_feed, RNN_data_feed, 0.001)
        train(100, RNN_data_feed, RNN_data_feed, 0.01)
        for i in range(20):
            train(200, RNN_data_feed, RNN_data_feed, 0.001)
        # ========== make prediction
        pred = sess.run(prediction_last, feed_dict={x: RNN_data_feed})
        df_coef_lower.loc[:, f'{_f}_coef'] = pred
    df_coef_lower = coef_for_lower.copy()
    df_coef_lower.reset_index(inplace=True)
    df_coef_lower['district_code'] = df_coef_lower['dist_weekday'].str[:-2]
    df_coef_lower['weekday'] = df_coef_lower['dist_weekday'].str[-1:]
    return df_coef_lower


def get_coef_for_future(flow_for_future):
    RNN_data = get_seasonal_coef(flow_for_future, 'dwell', rule_mode=False)
    RNN_data, RNN_data_feed = get_rnn_data(RNN_data)
    df_coef_future = pd.DataFrame(index=RNN_data.index)
    for _f in ['dwell', 'flow_in', 'flow_out']:
        RNN_data = get_seasonal_coef(flow_for_future, _f, rule_mode=False)
        RNN_data, RNN_data_feed = get_rnn_data(RNN_data)
        RNN_data_feed_1 = RNN_data_feed[:-100, :, :]
        RNN_data_feed_2 = RNN_data_feed[-100:, :, :]
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            train(200, RNN_data_feed, RNN_data_feed, 0.01)
        for i in range(10):
            train(200, RNN_data_feed, RNN_data_feed, 0.001)
        train(100, RNN_data_feed, RNN_data_feed, 0.01)
        for i in range(20):
            train(200, RNN_data_feed, RNN_data_feed, 0.001)
        # ========== make prediction
        pred = sess.run(prediction_last, feed_dict={x: RNN_data_feed})
        df_coef_future.loc[:, f'{_f}_coef'] = pred
    df_coef_future.reset_index(inplace=True)
    df_coef_future['district_code'] = df_coef_future['dist_weekday'].str[:-2]
    df_coef_future['weekday'] = df_coef_future['dist_weekday'].str[-1:]
    return df_coef_future


def make_rnn_pred_for_future():
    flow_for_future = flow[(flow['date_dt'] < '2017-08-18') |
                           (flow['date_dt'] > '2017-08-24')].\
        sort_values(['district_code'])
    df_preds = get_coef_for_future(flow_for_future)
    for _f in ['dwell', 'flow_in', 'flow_out']:
        _f_avg = flow_for_future.iloc[-7:, :].groupby(
            ['district_code'], as_index=False)[_f].mean()
        df_preds = df_preds.merge(_f_avg, on=['district_code'])
        df_preds[f'{_f}_pred'] = (
                df_preds[f'{_f}_coef']*df_preds[_f]).astype(float)
    df_preds_fst = df_preds[['district_code', 'weekday', 'dwell_pred',
                             'flow_in_pred', 'flow_out_pred']].copy()
    df_preds_fst['weekday'] = df_preds_fst['weekday']. \
        replace(6, 20171105). \
        replace(dict(zip(range(6), range(20171106, 20171112))))
    df_preds_fst.columns = [
        'district_code', 'date_dt', 'dwell', 'flow_in', 'flow_out']
    for _f in ['dwell', 'flow_in', 'flow_out']:
        df_preds[_f] = df_preds. \
            groupby(['district_code'])[f'{_f}_pred'].transform('mean').values
        df_preds[f'{_f}_pred_snd'] = df_preds[f'{_f}_{mode}'] * df_preds[_f]
    df_preds_snd = df_preds[['district_code', 'weekday', 'dwell_pred_snd',
                             'flow_in_pred_snd', 'flow_out_pred_snd']]
    df_preds_snd = df_preds_snd[df_preds_snd['weekday'].isin([6, 0, 1])]
    df_preds_snd['weekday'] = df_preds_snd['weekday']. \
        replace({6: 20171112, 0: 20171113, 1: 20171114})
    df_preds_snd.columns = [
        'district_code', 'date_dt', 'dwell', 'flow_in', 'flow_out']
    df_preds_all = pd.concat([df_preds_fst, df_preds_snd], axis=0)
    df_preds_all.sort_values(['district_code', 'date_dt'], inplace=True)
    return df_preds_all


def make_rnn_pred_for_upper():
    flow_for_upper = flow[flow['date_dt'] >= '2017-08-24'].\
        sort_values(['district_code'])
    df_preds = get_coef_for_upper(flow_for_upper)
    for _f in ['dwell', 'flow_in', 'flow_out']:
        _f_avg = df_preds.iloc[:7, :].groupby(
            ['district_code'], as_index=False)[_f].mean()
        df_preds = df_preds.merge(_f_avg, on=['district_code'])
        df_preds[f'{_f}_pred'] = (
                df_preds[f'{_f}_coef'] * df_preds[_f]).astype(float)
        df_preds_fst = df_preds[['district_code', 'weekday', 'dwell_pred',
                                 'flow_in_pred', 'flow_out_pred']].copy()
        df_preds_fst = df_preds_fst[
            df_preds_fst['weekday'].isin([5, 6, 0, 1, 2])]
        df_preds_fst['weekday'] = df_preds_fst['weekday'].replace(
            dict(zip([5, 6, 0, 1, 2], range(20170819, 20170824))))
        df_preds_fst.columns = [
            'district_code', 'date_dt', 'dwell', 'flow_in', 'flow_out']
        df_preds_fst.sort_values(['district_code', 'date_dt'], inplace=True)
        return df_preds_fst


def make_rnn_pred_for_lower():
    flow_for_lower = flow[flow['date_dt'] <= '2017-08-18']. \
        sort_values(['district_code'])
    df_preds = get_coef_for_lower(flow_for_lower)
    for _f in ['dwell', 'flow_in', 'flow_out']:
        _f_avg = flow_for_lower.iloc[-7:, :].groupby(
            ['district_code'], as_index=False)[_f].mean()
        df_preds = df_preds.merge(_f_avg, on=['district_code'])
        df_preds[f'{_f}_pred'] = (
                df_preds[f'{_f}_coef'] * df_preds[_f]).astype(float)
        df_preds_fst = df_preds[['district_code', 'weekday', 'dwell_pred',
                                 'flow_in_pred', 'flow_out_pred']].copy()
        df_preds_fst = df_preds_fst[
            df_preds_fst['weekday'].isin([5, 6, 0, 1, 2])]
        df_preds_fst['weekday'] = df_preds_fst['weekday'].replace(
            dict(zip([5, 6, 0, 1, 2], range(20170819, 20170824))))
        df_preds_fst.columns = [
            'district_code', 'date_dt', 'dwell', 'flow_in', 'flow_out']
        df_preds_fst.sort_values(['district_code', 'date_dt'], inplace=True)
        return df_preds_fst


def make_prediction_with_rnn_swa():
    df_preds_future = make_rnn_pred_for_future()
    df_preds_lower = make_rnn_pred_for_lower()
    df_preds_upper = make_rnn_pred_for_upper()
    df_preds_lower.iloc[:, 2:] = (df_preds_lower.iloc[:, 2:] + \
                                  df_preds_upper.iloc[:, 2:]) / 2
    df_preds = pd.concat([df_preds_future, df_preds_lower], axis=0)
    df_preds.insert(0, 'city_code',
                    df_preds['district_code'].str.split('_').str[0])
    df_preds = df_preds.reindex(columns=flow.columns)
    df_preds.replace(dict(zip(code_dict.values(), code_dict.keys())),
                     inplace=True)
    df_preds.sort_values(['city_code', 'district_code', 'date_dt'],
                         inplace=True)
    return df_preds
