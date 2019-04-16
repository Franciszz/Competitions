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
from sklearn.preprocessing import LabelEncoder
from pipeline_tools import from_pickle
train, test = from_pickle('train_df.csv'), from_pickle('test_df.csv')

excluded = ['date', 'fullVisitorId', 'sessionId', 'totals_transactionRevenue', 
            'visitId', 'visitStartTime', 'month', 'day', 'help',
            'visitDate', 'visitTime']

cat_cols = [f for f in train.columns if (
        train[f].dtype == 'object' and f not in excluded)]
real_cols = [f for f in train.columns if (
        not f in cat_cols and f not in excluded)]

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

for feature in ['date', 'fullVisitorId', 'sessionId', 
                'totals_transactionRevenue', 'visitId', 'visitStartTime', 
                'visitDate', 'visitTime']:
    del train[feature]
    del test[feature]


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


# In[ ]:


lgbmodel = lgb.LGBMRegressor(
        n_estimators=1000, objective="regression", metric="rmse", 
        num_leaves=31, min_child_samples=100, learning_rate=0.03, 
        bagging_fraction=0.7, feature_fraction=0.5, bagging_frequency=5, 
        bagging_seed=2019, subsample=.9, colsample_bytree=.9, 
        use_best_model=True)


# In[ ]:


Kfolder.validate(train, test, real_cols + cat_cols, 
                 lgbmodel, "lgbpred", prepare_stacking=True)


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
    agg_data = df[real_cols + cat_cols + ['fullVisitorId']].groupby(
            'fullVisitorId').mean()
    
    pred_list = df[['fullVisitorId', 'lgbpred']].groupby(
            'fullVisitorId').apply(lambda visitor_df: list(visitor_df.lgbpred))        .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})
    all_predictions = pd.DataFrame(list(pred_list.values), index=agg_data.index)
#    feats = all_predictions.columns

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
user_train["totals_transactionRevenue"] = train[[
        'fullVisitorId', 'totals_transactionRevenue']].groupby(
        'fullVisitorId').sum()


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
        num_leaves=31, min_child_samples=100,learning_rate=0.025, 
        bagging_fraction=0.7, feature_fraction=0.5, bagging_frequency=5, 
        bagging_seed=2019, subsample=.9, colsample_bytree=.9,
        use_best_model=True)


# In[ ]:


Kfolder.validate(user_train, user_test, features, lgbmodel, 
                 name="lgbfinal", prepare_stacking=True)


# In[ ]:
#
#
#xgbmodel = xgb.XGBRegressor(
#        max_depth=22, learning_rate=0.02, n_estimators=1000, 
#        objective='reg:linear', gamma=1.45, seed=2019, silent=False,
#        subsample=0.67, colsample_bytree=0.054, colsample_bylevel=0.50)
#

# In[ ]:


#Kfolder.validate(user_train, user_test, features, 
#                 xgbmodel, name="xgbfinal", prepare_stacking=True)


# In[ ]:

#
#catmodel = cat.CatBoostRegressor(iterations=500, learning_rate=0.2, 
#                                 depth=5, random_seed=2019)
#

# In[ ]:

#
#Kfolder.validate(user_train, user_test, features, catmodel, name="catfinal", 
#                 prepare_stacking=True,
#                fit_params={"use_best_model": True, "verbose": 100})


# # Ensembling dragons

# In[ ]:


user_train['PredictedLogRevenue'] = user_train["lgbfinal"] #+                                     0.2 * user_train["xgbfinal"] +                                     0.4 * user_train["catfinal"]
score(user_train, user_train.PredictedLogRevenue)


# In[ ]:


user_test['PredictedLogRevenue'] = user_test["lgbfinal"]# +  0.4 * user_test["catfinal"] + 0.2 * user_test["xgbfinal"]
user_test[['PredictedLogRevenue']].to_csv('submission/leaky_submission_2.csv', index=True)










