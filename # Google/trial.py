# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 23:06:19 2018

@author: Franc
"""

# In[Package Input]
import pandas as pd
pd.options.display.max_columns = 999
import numpy as np
import lightgbm as lgb
import warnings
warnings.simplefilter("ignore")
import gc
import holidays
from datetime import timedelta
gc.enable()
# In[Data Input]
train = pd.read_csv('data/extracted_fields_train.gz', dtype={
        'date': str, 'fullVisitorId': str, 'sessionId':str, 'visitId': int})
test = pd.read_csv('data/extracted_fields_test.gz', dtype={
        'date': str, 'fullVisitorId': str, 'sessionId':str, 'visitId': int})
train_store_1 = pd.read_csv(
        'data/Train_external_data.csv', low_memory=False, 
        skiprows=6, dtype={'Client Id':str})
train_store_2 = pd.read_csv(
        'data/Train_external_data_2.csv', low_memory=False, 
        skiprows=6, dtype={"Client Id":'str'})
test_store_1 = pd.read_csv(
        'data/Test_external_data.csv', low_memory=False, 
        skiprows=6, dtype={"Client Id":'str'})
test_store_2 = pd.read_csv(
        'data/Test_external_data_2.csv', low_memory=False, 
        skiprows=6, dtype={"Client Id":'str'})
# In[Data Merge]
for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
    df['visitId'] = df['Client Id'].apply(
            lambda x: x.split('.', 1)[1]).astype(np.int64)
# Merge with train/test data
train = train.merge(pd.concat([train_store_1, train_store_2], 
                              sort=False), how="left", on="visitId")
test = test.merge(pd.concat([test_store_1, test_store_2], 
                            sort=False), how="left", on="visitId")
# Drop Client Id
for df in [train, test]:
    df.drop("Client Id", 1, inplace=True)
    df.columns = df.columns.str.replace('.','_')
    
for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
    del df
    
gc.collect()
# In[Variable Clean]
def browser_transform(x):
    browsers = ['chrome','safari','firefox','internet explorer',
                'edge','opera','coc coc','maxthon','iron']
    if x in browsers:
        return x.lower()
    elif  ('android' in x) or ('samsung' in x) or ('mini' in x) or (
            'iphone' in x) or ('in-app' in x) or ('playstation' in x):
        return 'mobile browser'
    elif  ('mozilla' in x) or ('chrome' in x) or ('blackberry' in x) or (
            'nokia' in x) or ('browser' in x) or ('amazon' in x):
        return 'mobile browser'
    elif  ('lunascape' in x) or ('netscape' in x) or (
            'blackberry' in x) or ('konqueror' in x) or (
                    'puffin' in x) or ('amazon' in x):
        return 'mobile browser'
    elif '(not set)' in x:
        return x
    else:
        return 'others'
def adContent_transform(x):
    if  ('google' in x):
        return 'google'
    elif  ('placement' in x) | ('placememnt' in x):
        return 'placement'
    elif ('(not set)' in x) or ('nan' in x):
        return x
    elif 'ad' in x:
        return 'ad'
    else:
        return 'others'
def source_transform(x):
    if  'google' in x:
        return 'google'
    elif  'youtube' in x:
        return 'youtube'
    elif 'nan' in x:
        return x
    elif 'yahoo' in x:
        return 'yahoo'
    elif 'facebook' in x:
        return 'facebook'
    elif 'reddit' in x:
        return 'reddit'
    elif 'bing' in x:
        return 'bing'
    elif 'quora' in x:
        return 'quora'
    elif 'outlook' in x:
        return 'outlook'
    elif 'linkedin' in x:
        return 'linkedin'
    elif 'pinterest' in x:
        return 'pinterest'
    elif 'ask' in x:
        return 'ask'
    elif 'siliconvalley' in x:
        return 'siliconvalley'
    elif 'lunametrics' in x:
        return 'lunametrics'
    elif 'amazon' in x:
        return 'amazon'
    elif 'mysearch' in x:
        return 'mysearch'
    elif 'qiita' in x:
        return 'qiita'
    elif 'messenger' in x:
        return 'messenger'
    elif 'twitter' in x:
        return 'twitter'
    elif 't.co' in x:
        return 't.co'
    elif 'vk.com' in x:
        return 'vk.com'
    elif 'search' in x:
        return 'search'
    elif 'edu' in x:
        return 'edu'
    elif 'mail' in x:
        return 'mail'
    elif 'ad' in x:
        return 'ad'
    elif 'golang' in x:
        return 'golang'
    elif 'direct' in x:
        return 'direct'
    elif 'dealspotr' in x:
        return 'dealspotr'
    elif 'sashihara' in x:
        return 'sashihara'
    elif 'phandroid' in x:
        return 'phandroid'
    elif 'baidu' in x:
        return 'baidu'
    elif 'mdn' in x:
        return 'mdn'
    elif 'duckduckgo' in x:
        return 'duckduckgo'
    elif 'seroundtable' in x:
        return 'seroundtable'
    elif 'metrics' in x:
        return 'metrics'
    elif 'sogou' in x:
        return 'sogou'
    elif 'businessinsider' in x:
        return 'businessinsider'
    elif 'github' in x:
        return 'github'
    elif 'gophergala' in x:
        return 'gophergala'
    elif 'yandex' in x:
        return 'yandex'
    elif 'msn' in x:
        return 'msn'
    elif 'dfa' in x:
        return 'dfa'
    elif '(not set)' in x:
        return '(not set)'
    elif 'feedly' in x:
        return 'feedly'
    elif 'arstechnica' in x:
        return 'arstechnica'
    elif 'squishable' in x:
        return 'squishable'
    elif 'flipboard' in x:
        return 'flipboard'
    elif 't-online.de' in x:
        return 't-online.de'
    elif 'sm.cn' in x:
        return 'sm.cn'
    elif 'wow' in x:
        return 'wow'
    elif 'baidu' in x:
        return 'baidu'
    elif 'partners' in x:
        return 'partners'
    else:
        return 'others'
for df in [train, test]:
    df['visitStartTime'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['date'] = df['visitStartTime']
    df["Revenue"].fillna('$', inplace=True)
    df["Revenue"] = df["Revenue"].apply(
            lambda x: x.replace('$', '').replace(',', ''))
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["Revenue"].fillna(0.0, inplace=True)
    df['Avg_ Session Duration'] = pd.to_timedelta(
        df['Avg_ Session Duration']).map(lambda x:x.seconds).\
        fillna(0).astype(int)
    df['Bounce Rate'] = df['Bounce Rate'].str.replace('%','').\
        fillna(0).astype(float)
    df['Goal Conversion Rate'] = df['Goal Conversion Rate'].str.\
        replace('%','').fillna(0).astype(float)  
    df['device_browser_revise'] = df['device_browser'].map(
            lambda x: browser_transform(str(x).lower())).astype('str')
    df['trafficSource_source_revise'] = df['trafficSource_source'].map(
            lambda x: browser_transform(str(x).lower())).astype('str')
    df['trafficSource_adContent_revise'] = df['trafficSource_adContent'].map(
            lambda x: browser_transform(str(x).lower())).astype('str')
    df['trafficSource_adContent_dominant'] = np.where(
                df['trafficSource_adContent'].notnull(),
                np.where(df['trafficSource_adContent'].isin([
                        'Google Merchandise Collection', 'Google Online Store', 
                        'Full auto ad IMAGE ONLY','Swag with Google Logos',
                        '20% discount','{KeyWord:Google Branded Gear}',
                        '{KeyWord:Want Google Stickers?}']),
                        df['trafficSource_adContent'],'others'), np.nan)
    df['trafficSource_keyword_dominant'] = np.where(
                df['trafficSource_keyword'].notnull(),
                np.where(df['trafficSource_keyword'].isin([
                        '(not provided)', '6qEhsCssdK0z36ri', 
                        '1hZbAqLCbjwfgOH7', '(Remarketing/Content targeting)',
                        '1X4Me6ZKNV0zg-jV']), df['trafficSource_keyword'],
                    np.where(df['trafficSource_keyword'].isin([
                            'google merchandise store','google store',
                            'Google Merchandise','+Google +Merchandise']),
                            'Google','others')),np.nan)
    df.set_index(['visitStartTime'], inplace=True)
    df.sort_index(inplace=True)

# In[Clear Rare Features]
def clearRare(columnname, limit = 1000):
    vc = test[columnname].value_counts()
    common = vc > limit
    common = set(common.index[common].values)
    print("\nSet", sum(vc <= limit), columnname, 
          "categories to 'other';", end=" ")
    train.loc[train[columnname].map(
            lambda x: x not in common), columnname] = 'other'
    test.loc[test[columnname].map(
            lambda x: x not in common), columnname] = 'other'
    print("\nnow there are", train[columnname].nunique(), "categories in train")
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
for feature in [
        'device_browser','device_operatingSystem',
        'geoNetwork_networkDomain','geoNetwork_country',
        'geoNetwork_city','geoNetwork_metro',
        'geoNetwork_networkDomain','geoNetwork_region',
        'geoNetwork_subContinent','trafficSource_adContent',
        'trafficSource_referralPath','trafficSource_source',
        'trafficSource_campaign','trafficSource_keyword',
        'trafficSource_medium','trafficSource_referralPath']:
    clearRare(feature)
# In[Horizonal Features]
for df in [train, test]:
    df['totals_hits'] = df['totals_hits'].astype(float)
    df['totals_pageviews'] = df['totals_pageviews'].astype(float)
    df['totals_hits_interval'] = np.where(
        df['totals_hits']==1, 0,
        np.where(df['totals_hits']<4,1,
                 np.where(df['totals_hits']<7,2,
                          np.where(df['totals_hits']<11,3,\
        np.where(df['totals_hits']<22,4,
                 np.where(df['totals_hits']<33,5,
                          np.where(df['totals_hits']<49,6,7)))))))
    df['totals_pageviews_interval'] = np.where(
        df['totals_pageviews']==1, 0,
        np.where(df['totals_pageviews']<4,1,
                 np.where(df['totals_pageviews']<7,2,
                          np.where(df['totals_pageviews']<10,3,\
        np.where(df['totals_pageviews']<16,4,
                 np.where(df['totals_pageviews']<24,5,
                          np.where(df['totals_pageviews']<49,6,7)))))))
    df['totals_hits_pageviews_ratio'] = \
        (df['totals_hits'])/(df['totals_pageviews'] + 5)
    df['totals_high_visit'] = \
        np.logical_or(df['totals_hits']>4, df['totals_pageviews']>4).astype(int)
    df["id_incoherence"] = (pd.to_datetime(
            df['visitId'], unit='s') != df['date']).astype(int)
    df["visitId_dublicates"] = df['visitId'].map(
            df['visitId'].value_counts())
    df['session_dublicates'] = df['sessionId'].map(
            df['sessionId'].value_counts())
    df['visitDate'] = df['date'].dt.date.apply(
            lambda x: x.strftime('%Y%m%d')).astype(int)
    df['visitTime'] = df['date'].dt.time.apply(
            lambda x:int(x.strftime('%H%M%S')))
    df['visitMonth'] = df['date'].dt.month
    df['visitHour'] = df['date'].dt.hour
    df['visitWeekday'] = df['date'].dt.dayofweek
    df['visitHour_interval'] = np.where(
        (0<df['visitHour'])&(df['visitHour']<9),1,
        np.where((13<df['visitHour'])&(df['visitHour']<21),3,2))
    df['visitWeekday_weekend'] = \
        ((0<df['visitWeekday'])&(df['visitWeekday']<7)).astype(int)
# In[Vertical Feature]
df = pd.concat([train, test], sort=False)
df.sort_values(['fullVisitorId', 'date'], ascending=True, inplace=True)
groupby_df = df.groupby(['fullVisitorId'])
df['prev_session'] = (df['date'] - groupby_df['date'].shift(1)).\
    astype(np.int64) // 1e9 // 60 // 60
df['next_session'] = (df['date'] - groupby_df['date'].shift(-1)).\
    astype(np.int64) // 1e9 // 60 // 60
df['prev_hits'] = df['totals_hits']-groupby_df['totals_hits'].shift(1)
df['next_hits'] = df['totals_hits']-groupby_df['totals_hits'].shift(-1)
df['prev_pageviews'] = df['totals_pageviews']-groupby_df[
        'totals_pageviews'].shift(1)
df['next_pageviews'] = df['totals_pageviews']-groupby_df[
        'totals_pageviews'].shift(-1)
component = df.groupby('fullVisitorId')['totals_hits'].mean()
df['totals_hits_usermean'] = df['fullVisitorId'].map(component)
component = df.groupby('fullVisitorId')['totals_pageviews'].mean()
df['totals_pageviews_usermean'] = df['fullVisitorId'].map(component)
component = df.groupby('fullVisitorId')['visitNumber'].max()
df['totals_number_usermax'] = df['fullVisitorId'].map(component)
df['mergedate'] = df['date'].apply(
        lambda x: x.strftime('%Y%m%d')).astype(int)
component = df.groupby(['fullVisitorId','mergedate']).size().\
    reset_index().rename(columns={0:'visit_nums_inday'})
df = df.merge(component, on=['fullVisitorId','mergedate'], how='left')
#features = ['channelGrouping','device_browser',
#            'device_deviceCategory','visitDate','trafficSource_source']
#shift_df_pos = groupby_df[features].shift(1)
#shift_df_neg = groupby_df[features].shift(-1)
#for feature in features:
#    df[f'prev_{feature}_equal'] = (
#            df[feature]!=shift_df_pos[feature]).astype(int)
#    df[f'next_{feature}_equal'] = (
#            df[feature]!=shift_df_neg[feature]).astype(int)
# In[External Feature]
component = pd.read_csv(
        'data/usdxIndex.csv', parse_dates=[0]).fillna(method='ffill')
component['mergedate'] = component['date'].apply(
        lambda x: x.strftime('%Y%m%d')).astype(int)
component.drop(['date'],axis=1,inplace=True)
df = df.merge(component, on=['mergedate'], how='left')
component = pd.read_csv(
        'data/economicsIndex.csv', parse_dates=[0]).fillna(method='ffill')
component['mergedate'] = component['date'].apply(
        lambda x: x.strftime('%Y%m%d')).astype(int)
component.drop(['date'],axis=1,inplace=True)
df = df.merge(component, on=['mergedate'], how='left')
df[['external_usdx_index','external_employment',
      'external_rate','external_unemployment']] = \
    df[['external_usdx_index','external_employment','external_rate',
          'external_unemployment']].fillna(method='ffill')
us_holidays = list(holidays.UnitedStates(
                years=[2016,2017,2018]).keys())
df['external_holiday'] = (df['date'].dt.date.isin(us_holidays)).astype(int)
def judge_us_holiday(df, delta):
    judge_1 = (df['date']+timedelta(days=delta)).dt.date.isin(us_holidays)
    judge_2 = (df['date']+timedelta(days=-delta)).dt.date.isin(us_holidays)
    judge_holiday_delta = (judge_1|judge_2).astype(int)
    return judge_holiday_delta
df['external_holiday_1'] = df['external_holiday'] + judge_us_holiday(df, 1)
df['external_holiday_2'] = df['external_holiday_1'] + judge_us_holiday(df, 2)
df['external_holiday_3'] = df['external_holiday_2'] + judge_us_holiday(df, 3)
df['external_holiday_4'] = df['external_holiday_3'] + judge_us_holiday(df, 4)
df['external_holiday_5'] = df['external_holiday_4'] + judge_us_holiday(df, 5)
# In[Compound Feature]
""" compound variable """
df['extra_source_country'] = \
    df['trafficSource_source'] + '_' + df['geoNetwork_country']
df['extra_campaign_medium'] = \
    df['trafficSource_campaign'] + '_' + df['trafficSource_medium']
df['extra_browser_category'] = \
    df['device_browser'] + '_' + df['device_deviceCategory']
df['extra_browser_os'] = \
    df['device_browser'] + '_' + df['device_operatingSystem']
df['extra_device_deviceCategory_channelGrouping'] = \
    df['device_deviceCategory'] + "_" + df['channelGrouping']
df['extra_channelGrouping_browser'] = \
    df['device_browser'] + "_" + df['channelGrouping']
df['extra_channelGrouping_OS'] = \
    df['device_operatingSystem'] + "_" + df['channelGrouping']
for region in ['geoNetwork_country', 'geoNetwork_metro',
               'geoNetwork_city', 'geoNetwork_networkDomain',
               'geoNetwork_region', 'geoNetwork_subContinent']:
    for device in ['device_browser','device_deviceCategory', 
                   'device_operatingSystem', 'trafficSource_source']:
        df['extra_' + region + '_' + device] = \
            df[region] + '_' + df[device]
df['extra_content_source'] = \
    df['trafficSource_adContent'].astype(
            str) + "_" + df['extra_source_country']
df['extra_medium_source'] = \
    df['trafficSource_medium'] + "_" + df['extra_source_country']
train = df[df['date']<'2017-08-02 07:00:00']
test = df[df['date']>='2017-08-02 07:00:00']

# In[Encoding Feature]:
train = df[df['date']<'2017-08-02 07:00:00']
test = df[df['date']>='2017-08-02 07:00:00']
excluded = ['date', 'fullVisitorId', 'sessionId', 'mergedate',
            'totals_transactionRevenue', 'visitId', 'visitStartTime', 
            'visitDate', 'visitTime']
cat_cols = [f for f in train.columns if (
        train[f].dtype == 'object' and f not in excluded)]
real_cols = [f for f in train.columns if (
        not f in cat_cols and f not in excluded)]
# In[Train]
for feature in ['date', 'sessionId', 'mergedate',
                'visitId','visitDate', 'visitTime']:
    del train[feature]
    del test[feature]
# In[]
from sklearn.preprocessing import LabelEncoder
for col in cat_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(
            test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))

# In[]
# Preparing validation
# Function to tell us the score using the metric we actually care about
    
from sklearn.metrics import mean_squared_error
def score(data, y):
    validation_res = pd.DataFrame(
    {"fullVisitorId": data["fullVisitorId"].values,
     "transactionRevenue": data["totals_transactionRevenue"].values,
     "predictedRevenue": np.expm1(y)})

    validation_res = validation_res.groupby("fullVisitorId")[
            "transactionRevenue", "predictedRevenue"].sum().reset_index()
    return np.sqrt(mean_squared_error(
            np.log1p(validation_res["transactionRevenue"].values), 
            
            np.log1p(validation_res["predictedRevenue"].values)))


# Cute function to validate and prepare stacking

# In[ ]:


from sklearn.model_selection import GroupKFold

class KFoldValidation():
    def __init__(self, data, n_splits=4):
        unique_vis = np.array(sorted(
                data['fullVisitorId'].astype(str).unique()))
        folds = GroupKFold(n_splits)
        ids = np.arange(data.shape[0])
        
        self.fold_ids = []
        for trn_vis, val_vis in folds.split(
                X=unique_vis, y=unique_vis, groups=unique_vis):
            self.fold_ids.append([
                    ids[data['fullVisitorId'].astype(str).isin(
                            unique_vis[trn_vis])],
                    ids[data['fullVisitorId'].astype(str).isin(
                            unique_vis[val_vis])]
                ])
            
    def validate(self, train, test, features, 
                 model, name="", prepare_stacking=False, 
                 fit_params={"early_stopping_rounds": 50, "verbose": 100, 
                             "eval_metric": "rmse"}):
        model.FI = pd.DataFrame(index=features)
        full_score = 0
        
        if prepare_stacking:
            test[name] = 0
            train[name] = np.NaN
        
        for fold_id, (trn, val) in enumerate(self.fold_ids):
            devel = train[features].iloc[trn]
            y_devel = np.log1p(train["totals_transactionRevenue"].iloc[trn])
            valid = train[features].iloc[val]
            y_valid = np.log1p(train["totals_transactionRevenue"].iloc[val])
                       
            print("Fold ", fold_id, ":")
            model.fit(devel, y_devel, eval_set=[(valid, y_valid)], **fit_params)
            
            if len(model.feature_importances_) == len(features):  # some bugs in catboost?
                model.FI['fold' + str(fold_id)] = \
                    model.feature_importances_/model.feature_importances_.sum()

            predictions = model.predict(valid)
            predictions[predictions < 0] = 0
            print("Fold ", fold_id, " error: ", 
                  mean_squared_error(y_valid, predictions)**0.5)
            
            fold_score = score(train.iloc[val], predictions)
            full_score += fold_score / len(self.fold_ids)
            print("Fold ", fold_id, " score: ", fold_score)
            
            if prepare_stacking:
                train[name].iloc[val] = predictions
                
                test_predictions = model.predict(test[features])
                test_predictions[test_predictions < 0] = 0
                test[name] += test_predictions / len(self.fold_ids)
                
        print("Final score: ", full_score)
        return full_score


# In[ ]:

Kfolder = KFoldValidation(train)
lgbmodel = lgb.LGBMRegressor(
        n_estimators=1000, objective="regression", metric="rmse", 
        num_leaves=28   , min_child_samples=180, learning_rate=0.025, 
        bagging_fraction=0.75, feature_fraction=0.75, bagging_frequency=5, 
        bagging_seed=9611, subsample=.75, colsample_bytree=.75, 
        use_best_model=True)

Kfolder.validate(train, test, real_cols + cat_cols, 
                 lgbmodel, "lgbpred", prepare_stacking=True)

# In[ ]:

def create_user_df(df):
    agg_data = df[real_cols + cat_cols + ['fullVisitorId']].groupby(
            'fullVisitorId').mean()
    
    pred_list = df[['fullVisitorId', 'lgbpred']].groupby(
            'fullVisitorId').apply(lambda visitor_df: list(visitor_df.lgbpred)).\
            apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})
    all_predictions = pd.DataFrame(list(pred_list.values), index=agg_data.index)
#    feats = all_predictions.columns

    all_predictions['t_mean'] = all_predictions.mean(axis=1)
    all_predictions['t_median'] = all_predictions.median(axis=1)   # including t_mean as one of the elements? well, ok
    all_predictions['t_max'] = all_predictions.max(axis=1)
    all_predictions['t_min'] = all_predictions.min(axis=1)
    all_predictions['t_var'] = all_predictions.std(axis=1)
    all_predictions['t_sum_log'] = all_predictions.sum(axis=1)
    all_predictions['t_sum_act'] = all_predictions.fillna(0).sum(axis=1)
    all_predictions['t_nb_sess'] = all_predictions.isnull().sum(axis=1)

    full_data = pd.concat([
            agg_data, all_predictions], axis=1).astype(float)
    full_data['fullVisitorId'] = full_data.index
    del agg_data, all_predictions
    gc.collect()
    return full_data


# In[ ]:

user_train = create_user_df(train)
user_test = create_user_df(test)


# In[ ]:

features = list(user_train.columns)[:-1]  # don't include "fullVisitorId"
user_train["totals_transactionRevenue"] = train[[
        'fullVisitorId', 'totals_transactionRevenue']].groupby(
        'fullVisitorId').sum()
for f in features:
    if f not in user_test.columns:
        user_test[f] = np.nan
# In[]
Kfolder = KFoldValidation(user_train)
lgbmodel = lgb.LGBMRegressor(
        n_estimators=1000, objective="regression", metric="rmse", 
        num_leaves=36, min_child_samples=100,learning_rate=0.01, 
        bagging_fraction=0.8, feature_fraction=0.8, bagging_frequency=5, 
        bagging_seed=9611, subsample=.8, colsample_bytree=.8,
        use_best_model=True)

Kfolder.validate(user_train, user_test, features, lgbmodel, 
                 name="lgbfinal", prepare_stacking=True)


# In[ ]:

user_train['PredictedLogRevenue'] = user_train["lgbfinal"]
score(user_train, user_train.PredictedLogRevenue)

user_test['PredictedLogRevenue'] = user_test["lgbfinal"]
user_test[['PredictedLogRevenue']].to_csv(
        'submission/leaky_submission_11.csv', index=True)
#user_test2 = pd.read_csv('submission/leaky submission.csv')
#user_test1 = user_test[['PredictedLogRevenue']].reset_index()
#user_test3 = user_test2.merge(user_test1,on=['fullVisitorId'],
#                              how='left')
#user_test3['PredictedLogRevenue'] = np.where(
#        (user_test3['PredictedLogRevenue_x']<1)|(
#                user_test3['PredictedLogRevenue_x']<1), 0,
#        (user_test3['PredictedLogRevenue_x'] + user_test3[
#                'PredictedLogRevenue_y'])/2)
#user_test[['PredictedLogRevenue']].to_csv(
#        'submission/leaky_submission_5.csv', index=True)







