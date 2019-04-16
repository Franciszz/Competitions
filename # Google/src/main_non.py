# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 09:30:38 2018

@author: Franc
"""
# In[]
import pandas as pd
import numpy as np
# DRAGONS
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
# plots
import seaborn as sns
import matplotlib.pyplot as plt
# pandas / plt options
pd.options.display.max_columns = 999
plt.rcParams['figure.figsize'] = (14, 7)
font = {'family' : 'verdana',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)
# remove warnings
import warnings
warnings.simplefilter("ignore")
# garbage collector
import gc
gc.enable()
# In[]
train = pd.read_csv('data/extracted_fields_train.gz', 
                    dtype={'date': str, 'fullVisitorId': str,
                           'sessionId':str, 'visitId': np.int64})
test = pd.read_csv('data/extracted_fields_test.gz', 
                   dtype={'date': str, 'fullVisitorId': str, 
                          'sessionId':str, 'visitId': np.int64})
train.shape, test.shape

# In[]
# Getting data from leak
train_store_1 = pd.read_csv('data/Train_external_data.csv', 
                            low_memory=False, skiprows=6, 
                            dtype={"Client Id":'str'})
train_store_2 = pd.read_csv('data/Train_external_data_2.csv', 
                            low_memory=False, skiprows=6, 
                            dtype={"Client Id":'str'})
test_store_1 = pd.read_csv('data/Test_external_data.csv', 
                           low_memory=False, skiprows=6, 
                           dtype={"Client Id":'str'})
test_store_2 = pd.read_csv('data/Test_external_data_2.csv', 
                           low_memory=False, skiprows=6, 
                           dtype={"Client Id":'str'})
# In[]
# Getting VisitId from Google Analytics...
for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
    df["visitId"] = df["Client Id"].apply(
            lambda x: x.split('.', 1)[1]).astype(np.int64)
# In[]:

# Merge with train/test data
train = train.merge(pd.concat([train_store_1, train_store_2], 
                              sort=False), how="left", on="visitId")
test = test.merge(pd.concat([test_store_1, test_store_2], 
                            sort=False), how="left", on="visitId")

# Drop Client Id
for df in [train, test]:
    df.drop("Client Id", 1, inplace=True)
# In[]:

# Cleaning Revenue
for df in [train, test]:
    df["Revenue"].fillna('$', inplace=True)
    df["Revenue"] = df["Revenue"].apply(
            lambda x: x.replace('$', '').replace(',', ''))
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["Revenue"].fillna(0.0, inplace=True)

# In[ ]:
    
for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
    del df
gc.collect()
# In[ ]:


target_sums = train.groupby("fullVisitorId")[
        "totals.transactionRevenue"].sum().reset_index()


# In[ ]:


plt.scatter(range(target_sums.shape[0]), np.sort(
        np.log1p(target_sums["totals.transactionRevenue"].values)))
plt.xlabel('index')
plt.ylabel('TransactionRevenue')
plt.show()


# Key problem:

# In[ ]:

train.date = pd.to_datetime(train.date, format="%Y%m%d")
test.date = pd.to_datetime(test.date, format="%Y%m%d")
train.date.value_counts().sort_index().plot(label="train")
test.date.value_counts().sort_index().plot(label="test")
plt.legend()

# In[ ]:


def drawBars(columnname):
    sns.barplot(x="count", y="index", hue="dataset",
        data=pd.melt(pd.concat(
                [train[columnname].value_counts().rename("train"), 
                 test[columnname].value_counts().rename("test")], 
                 axis=1, sort="False").reset_index(),
            id_vars="index", var_name="dataset", value_name="count"))

drawBars("channelGrouping")

# In[ ]:


drawBars("geoNetwork.continent")


# In[ ]:


ids_train = set(train.fullVisitorId.unique())
ids_test = set(test.fullVisitorId.unique())
print("Unique visitor ids in train:", len(ids_train))
print("Unique visitor ids in test:", len(ids_test))
print("Common visitors in train and test:", len(ids_train & ids_test))


# Weird "double" sessions:

# In[ ]:


problem = train[train.sessionId.map(
        train.sessionId.value_counts() == 2)].sort_values(
    ["sessionId", 'visitStartTime'])
problem.head(10)


# VisitStartTime seems to be same thing as visitId... yet not always!

# In[ ]:


(train.visitStartTime == train.visitId).value_counts()


# Suspicious simultaneous visitors with same visitorId and same visitStartTime:

# In[ ]:


train.loc[pd.to_datetime(
        train.visitStartTime, unit='s') == "2017-04-25 18:49:35"].head(8)


# Seems to be a serious problem:

# In[ ]:


print("Train: ", np.bincount(train.visitId.value_counts()))


# In[ ]:


print("test: ", np.bincount(test.visitId.value_counts()))

# # Preprocessing

# Setting time as index and saving time as feature (for FE purposes only)

# In[ ]:


train.visitStartTime = pd.to_datetime(
        train.visitStartTime, unit='s')
test.visitStartTime = pd.to_datetime(
        test.visitStartTime, unit='s')
train["date"] = train.visitStartTime
test["date"] = test.visitStartTime


# In[ ]:


train.set_index("visitStartTime", inplace=True)
test.set_index("visitStartTime", inplace=True)
train.sort_index(inplace=True)
test.sort_index(inplace=True)


# Clearing rare categories and setting 0 to NaNs:

# In[ ]:

def clearRare(columnname, limit = 1000):
    # you may search for rare categories in train, train&test, or just test
    #vc = pd.concat([train[columnname], test[columnname]], sort=False).value_counts()
    vc = test[columnname].value_counts()
    
    common = vc > limit
    common = set(common.index[common].values)
    print("Set", sum(vc <= limit), columnname, 
          "categories to 'other';", end=" ")
    
    train.loc[train[columnname].map(
            lambda x: x not in common), columnname] = 'other'
    test.loc[test[columnname].map(
            lambda x: x not in common), columnname] = 'other'
    print("now there are", train[columnname].nunique(), "categories in train")


# In[ ]:


train.fillna(0, inplace=True)
test.fillna(0, inplace=True)


# In[ ]:

clearRare("device.browser")
clearRare("device.operatingSystem")
clearRare("geoNetwork.country")
clearRare("geoNetwork.city")
clearRare("geoNetwork.metro")
clearRare("geoNetwork.networkDomain")
clearRare("geoNetwork.region")
clearRare("geoNetwork.subContinent")
clearRare("trafficSource.adContent")
clearRare("trafficSource.campaign")
clearRare("trafficSource.keyword")
clearRare("trafficSource.medium")
clearRare("trafficSource.referralPath")
clearRare("trafficSource.source")


# In[ ]:


# Clearing leaked data:
for df in [train, test]:
    df["Avg. Session Duration"][df[
            "Avg. Session Duration"] == 0] = "00:00:00"
    df["Avg. Session Duration"] = df[
            "Avg. Session Duration"].str.split(':').apply(
            lambda x: int(x[0]) * 60 + int(x[1]))
    df["Bounce Rate"] = df["Bounce Rate"].astype(str).apply(
            lambda x: x.replace('%', '')).astype(float)
    df["Goal Conversion Rate"] = df[
            "Goal Conversion Rate"].astype(str).apply(
            lambda x: x.replace('%', '')).astype(float)

# # Features
    
# Based on strange things in dataset:

# In[ ]:


for df in [train, test]:
    # remember these features were equal, but not always? May be it means something...
    df["id_incoherence"] = pd.to_datetime(df.visitId, unit='s') != df.date
    # remember visitId dublicates?
    df["visitId_dublicates"] = df.visitId.map(df.visitId.value_counts())
    # remember session dublicates?
    df["session_dublicates"] = df.sessionId.map(df.sessionId.value_counts())


# Basic time features:

# In[ ]:


for df in [train, test]:
    df['weekday'] = df['date'].dt.dayofweek.astype(object)
    df['time'] = df['date'].dt.second + df['date'].dt.minute*60 + df[
            'date'].dt.hour*3600
    #df['month'] = df['date'].dt.month   # it must not be included in features during learning!
    df['day'] = df['date'].dt.date       # it must not be included in features during learning!


# Looking to future features (from https://www.kaggle.com/ashishpatel26/future-is-here):

# In[ ]:


df = pd.concat([train, test])
df.sort_values(['fullVisitorId', 'date'], ascending=True, inplace=True)
df['prev_session'] = (df['date'] - df[['fullVisitorId', 'date']].groupby(
        'fullVisitorId')['date'].shift(1)).astype(np.int64) // 1e9 // 60 // 60
df['next_session'] = (df['date'] - df[['fullVisitorId', 'date']].groupby(
        'fullVisitorId')['date'].shift(-1)).astype(np.int64) // 1e9 // 60 // 60
df.sort_index(inplace=True)

train = df[:len(train)]
test = df[len(train):]


# Paired categories from "teach-lightgbm-to-sum-predictions" kernel

# In[ ]:


for df in [train, test]:
    df['source.country'] = df['trafficSource.source'] + '_' + df[
            'geoNetwork.country']
    df['campaign.medium'] = df['trafficSource.campaign'] + '_' + df[
            'trafficSource.medium']
    df['browser.category'] = df['device.browser'] + '_' + df[
            'device.deviceCategory']
    df['browser.os'] = df['device.browser'] + '_' + df[
            'device.operatingSystem']


# In[ ]:


for df in [train, test]:
    df['device_deviceCategory_channelGrouping'] = df[
            'device.deviceCategory'] + "_" + df['channelGrouping']
    df['channelGrouping_browser'] = df[
            'device.browser'] + "_" + df['channelGrouping']
    df['channelGrouping_OS'] = df[
            'device.operatingSystem'] + "_" + df['channelGrouping']
    
    for i in ['geoNetwork.city', 'geoNetwork.continent', 
              'geoNetwork.country','geoNetwork.metro', 
              'geoNetwork.networkDomain', 'geoNetwork.region',
              'geoNetwork.subContinent']:
        for j in ['device.browser','device.deviceCategory', 
                  'device.operatingSystem', 'trafficSource.source']:
            df[i + "_" + j] = df[i] + "_" + df[j]
    
    df['content.source'] = df[
            'trafficSource.adContent'].astype(str) + "_" + df['source.country']
    df['medium.source'] = df[
            'trafficSource.medium'] + "_" + df['source.country']

# User-aggregating features:

# In[ ]:


for feature in ["totals.hits", "totals.pageviews"]:
    info = pd.concat([train, test], 
                     sort=False).groupby("fullVisitorId")[feature].mean()
    train["usermean_" + feature] = train.fullVisitorId.map(info)
    test["usermean_" + feature] = test.fullVisitorId.map(info)
    
for feature in ["visitNumber"]:
    info = pd.concat([train, test], sort=False).groupby(
            "fullVisitorId")[feature].max()
    train["usermax_" + feature] = train.fullVisitorId.map(info)
    test["usermax_" + feature] = test.fullVisitorId.map(info)


# # Encoding features

# In[ ]:


excluded = ['date', 'fullVisitorId', 'sessionId', 'totals_transactionRevenue', 
            'visitId', 'visitStartTime', 'month', 'day', 'help',
            'visitDate', 'visitTime']

cat_cols = [f for f in train.columns if (
        train[f].dtype == 'object' and f not in excluded)]
real_cols = [f for f in train.columns if (
        not f in cat_cols and f not in excluded)]


# In[ ]:


train[cat_cols].nunique()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
for col in cat_cols:
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(
            test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))


# In[ ]:


for col in real_cols:
    train[col] = train[col].astype(float)
    test[col] = test[col].astype(float)


# In[ ]:


train[real_cols + cat_cols].head()


# In[ ]:


for to_del in ["date", "sessionId", "visitId", "day"]:
    del train[to_del]
    del test[to_del]


# In[]
# Preparing validation
# Function to tell us the score using the metric we actually care about
    
from sklearn.metrics import mean_squared_error
def score(data, y):
    validation_res = pd.DataFrame(
    {"fullVisitorId": data["fullVisitorId"].values,
     "transactionRevenue": data["totals.transactionRevenue"].values,
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
    def __init__(self, data, n_splits=5):
        unique_vis = np.array(sorted(data['fullVisitorId'].astype(str).unique()))
        folds = GroupKFold(n_splits)
        ids = np.arange(data.shape[0])
        
        self.fold_ids = []
        for trn_vis, val_vis in folds.split(
                X=unique_vis, y=unique_vis, groups=unique_vis):
            self.fold_ids.append([
                    ids[data['fullVisitorId'].astype(str).isin(unique_vis[trn_vis])],
                    ids[data['fullVisitorId'].astype(str).isin(unique_vis[val_vis])]
                ])
            
    def validate(self, train, test, features, model, name="", prepare_stacking=False, 
                 fit_params={"early_stopping_rounds": 50, "verbose": 100, 
                             "eval_metric": "rmse"}):
        model.FI = pd.DataFrame(index=features)
        full_score = 0
        
        if prepare_stacking:
            test[name] = 0
            train[name] = np.NaN
        
        for fold_id, (trn, val) in enumerate(self.fold_ids):
            devel = train[features].iloc[trn]
            y_devel = np.log1p(train["totals.transactionRevenue"].iloc[trn])
            valid = train[features].iloc[val]
            y_valid = np.log1p(train["totals.transactionRevenue"].iloc[val])
                       
            print("Fold ", fold_id, ":")
            model.fit(devel, y_devel, eval_set=[(valid, y_valid)], **fit_params)
            
            if len(model.feature_importances_) == len(features):  # some bugs in catboost?
                model.FI['fold' + str(fold_id)] = model.feature_importances_ / model.feature_importances_.sum()

            predictions = model.predict(valid)
            predictions[predictions < 0] = 0
            print("Fold ", fold_id, " error: ", mean_squared_error(y_valid, predictions)**0.5)
            
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


# In[ ]:


lgbmodel = lgb.LGBMRegressor(n_estimators=1000, objective="regression", metric="rmse", num_leaves=31, min_child_samples=100,
                      learning_rate=0.03, bagging_fraction=0.7, feature_fraction=0.5, bagging_frequency=5, 
                      bagging_seed=2019, subsample=.9, colsample_bytree=.9, use_best_model=True)


# In[ ]:


Kfolder.validate(train, test, real_cols + cat_cols, lgbmodel, "lgbpred", prepare_stacking=True)


# In[ ]:


lgbmodel.FI.mean(axis=1).sort_values()[:30].plot(kind="barh")


# # User-level

# Make one user one object:
# * all features are averaged
# * we hope, that categorical features do not change for one user (that's not true :/ )
# * categoricals labels are averaged (!!!) and are treated as numerical features (o_O)
# * predictions are averaged in multiple ways...

# In[ ]:


def create_user_df(df):
    agg_data = df[real_cols + cat_cols + ['fullVisitorId']].groupby('fullVisitorId').mean()
    
    pred_list = df[['fullVisitorId', 'lgbpred']].groupby('fullVisitorId').apply(lambda visitor_df: list(visitor_df.lgbpred))        .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})
    all_predictions = pd.DataFrame(list(pred_list.values), index=agg_data.index)
    feats = all_predictions.columns

    all_predictions['t_mean'] = all_predictions.mean(axis=1)
    all_predictions['t_median'] = all_predictions.median(axis=1)   # including t_mean as one of the elements? well, ok
    all_predictions['t_sum_log'] = all_predictions.sum(axis=1)
    all_predictions['t_sum_act'] = all_predictions.fillna(0).sum(axis=1)
    all_predictions['t_nb_sess'] = all_predictions.isnull().sum(axis=1)

    full_data = pd.concat([agg_data, all_predictions], axis=1).astype(float)
    full_data['fullVisitorId'] = full_data.index
    del agg_data, all_predictions
    gc.collect()
    return full_data


# In[ ]:

user_train = create_user_df(train)
user_test = create_user_df(test)


# In[ ]:


features = list(user_train.columns)[:-1]  # don't include "fullVisitorId"
user_train["totals.transactionRevenue"] = train[['fullVisitorId', 'totals.transactionRevenue']].groupby('fullVisitorId').sum()


# In[ ]:


for f in features:
    if f not in user_test.columns:
        user_test[f] = np.nan


# # Meta-models

# In[ ]:


Kfolder = KFoldValidation(user_train)


# In[ ]:


lgbmodel = lgb.LGBMRegressor(
        n_estimators=1000, objective="regression", metric="rmse", 
        num_leaves=31, min_child_samples=100,learning_rate=0.03, 
        bagging_fraction=0.7, feature_fraction=0.5, bagging_frequency=5, 
        bagging_seed=2019, subsample=.9, colsample_bytree=.9,use_best_model=True)


# In[ ]:


Kfolder.validate(user_train, user_test, features, lgbmodel, 
                 name="lgbfinal", prepare_stacking=True)


# In[ ]:


xgbmodel = xgb.XGBRegressor(
        max_depth=22, learning_rate=0.02, n_estimators=1000, 
        objective='reg:linear', gamma=1.45, seed=2019, silent=False,
        subsample=0.67, colsample_bytree=0.054, colsample_bylevel=0.50)


# In[ ]:


Kfolder.validate(user_train, user_test, features, 
                 xgbmodel, name="xgbfinal", prepare_stacking=True)


# In[ ]:


catmodel = cat.CatBoostRegressor(iterations=500, learning_rate=0.2, 
                                 depth=5, random_seed=2019)


# In[ ]:


Kfolder.validate(user_train, user_test, features, catmodel, name="catfinal", 
                 prepare_stacking=True,
                fit_params={"use_best_model": True, "verbose": 100})


# # Ensembling dragons

# In[ ]:


user_train['PredictedLogRevenue'] = user_train["lgbfinal"] #+                                     0.2 * user_train["xgbfinal"] +                                     0.4 * user_train["catfinal"]
score(user_train, user_train.PredictedLogRevenue)


# In[ ]:


user_test['PredictedLogRevenue'] = user_test["lgbfinal"]# +  0.4 * user_test["catfinal"] + 0.2 * user_test["xgbfinal"]
user_test[['PredictedLogRevenue']].to_csv('leaky submission.csv', index=True)










