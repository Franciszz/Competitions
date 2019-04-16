# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 21:36:47 2018

@author: Franc
"""

import numpy as np
from steppy.base import BaseTransformer
from steppy.utils import get_logger

import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

logger = get_logger()

class LightGBM(BaseTransformer):
    def __init__(self, lgb_params):
        self.df = None
        self.estimator = None
        self.params = lgb_params['params']
        #self.seed = lgb_params['seed']
        self.eval_ratio = lgb_params['eval_ratio']
        self.test_ratio = lgb_params['test_ratio']
        self.nfold = lgb_params['nfold']
        self.num_boost_round = lgb_params['num_boost_round']
        self.learning_rates = lgb_params['params']['learning_rate']
        self.early_stopping_rounds = lgb_params['early_stopping_rounds']
        
    def loglikelood(self, preds, train_data):
        labels = train_data.get_label()
        preds = 1. / (1. + np.exp(-preds))
        grad = preds - labels
        hess = preds * (1. - preds)
        return grad, hess
    
    def roc_auc_error(self, preds, train_data):
        labels = train_data.get_label()
        return 'error', 1-roc_auc_score(labels, preds), False
    
    def fit(self, df):
        feature_name = list(df.columns[2:])
        
        categorical_feature = list(df.dtypes.index[df.dtypes=='object'])
#        df.loc[:,categorical_feature] = \
#            df.loc[:,categorical_feature].replace({np.nan:100}).\
#            astype('int64').replace({100:np.nan})
#            
        df_x, df_y = df.iloc[:,2:], df.TARGET
            
        x_train, x_eval, y_train, y_eval = \
                train_test_split(df_x, df_y,         
                                 test_size = self.test_ratio)
            
#        x_train, x_eval, y_train, y_eval = \
#            train_test_split(x_train_eval, y_train_eval, 
#                             test_size = self.eval_ratio,
#                             random_state = self.seed)
            
        lgb_train = \
                lgb.Dataset(x_train, y_train, 
                            feature_name = feature_name,
                            categorical_feature = categorical_feature,
                            free_raw_data = False)
        lgb_eval = \
                lgb.Dataset(x_eval, y_eval, 
                            feature_name = feature_name,
                            categorical_feature = categorical_feature,
                            free_raw_data = False)
        
#        categorical_feature = \
#            list(df.dtypes.reset_index().index[df.dtypes=='object']-2)
#            
        self.estimator = \
                lgb.train(params = self.params,
                          train_set = lgb_train,
                          num_boost_round = self.num_boost_round,
                          init_model = None,
                          valid_sets = lgb_eval,
                          fobj = self.loglikelood,
                          feval = self.roc_auc_error,
                          early_stopping_rounds = self.early_stopping_rounds,
                          learning_rates = \
                              lambda x: self.learning_rates*(0.99**x),
                          verbose_eval = True,
                          feature_name = feature_name,
                          categorical_feature = categorical_feature)
        
#        y_test_predict = lgb_clf.predict(x_test.values)
#        lgb_clf_score = roc_auc_score(y_test.values, y_test_predict)
        
        return self
    
    def cv(self, df):
        feature_name = list(df.columns[2:])
        categorical_feature = list(df.dtypes.index[df.dtypes=='object'])
        
        df.loc[:,categorical_feature] = \
                df.loc[:,categorical_feature].replace({np.nan:100}).\
                astype('int64').replace({100:np.nan})
            
        df_x, df_y = \
                df.iloc[:,2:], df.TARGET
        
        lgb_train = \
                lgb.Dataset(df_x.values, df_y.values, 
                            feature_name = feature_name,
                            categorical_feature = categorical_feature)
        lgb_cv_hist = \
                lgb.cv(params = self.params,
                       nfold = self.nfold,
                       train_set = lgb_train,
                       num_boost_round = self.num_boost_round,
                       fobj = self.loglikelood,
                       feval = self.roc_auc_error,
                       verbose_eval = True,
                       feature_name = feature_name,
                       categorical_feature = categorical_feature)
        return lgb_cv_hist
    
    def transform(self, x_test):
        prediction = self.estimator.predict(x_test)
        prediction = 1/(1+np.exp(-prediction))
        return {'prediction':prediction}

class LightGBMClassifier(BaseTransformer):
    def __init__(self, lgb_params):
        self.df = None
        self.estimator = None
        self.params = lgb_params['params']
        self.params_gs = lgb_params['params_gs']
        self.params_fit = lgb_params['params_fit']
        #self.seed = lgb_params['seed']
        self.eval_ratio = lgb_params['eval_ratio']
        self.test_ratio = lgb_params['test_ratio']
        self.nfold = lgb_params['nfold']
        self.num_boost_round = lgb_params['num_boost_round']
        self.learning_rates = lgb_params['params']['learning_rate']
        self.early_stopping_rounds = lgb_params['early_stopping_rounds']
        
    def loglikelood(self, preds, train_data):
        labels = train_data.get_label()
        preds = 1. / (1. + np.exp(-preds))
        grad = preds - labels
        hess = preds * (1. - preds)
        return grad, hess
    
    def roc_auc_error(self, preds, train_data):
        labels = train_data.get_label()
        return 'error', 1-roc_auc_score(labels, preds), False
    
    def fit(self, df):
        feature_name = list(df.columns[2:])
        
        categorical_feature = list(df.dtypes.index[df.dtypes=='object'])
#        df.loc[:,categorical_feature] = \
#            df.loc[:,categorical_feature].replace({np.nan:100}).\
#            astype('int64').replace({100:np.nan})
#            
        df_x, df_y = df.iloc[:,2:], df.TARGET
            
        x_train, x_eval, y_train, y_eval = \
                train_test_split(df_x, df_y,         
                                 test_size = self.test_ratio)
            
#        x_train, x_eval, y_train, y_eval = \
#            train_test_split(x_train_eval, y_train_eval, 
#                             test_size = self.eval_ratio,
#                             random_state = self.seed)
        self.estimator = LGBMClassifier(boosting_type = self.params['boosting_type'],
                                        num_leaves = self.params['num_leaves'],
                                        max_depth = self.params['max_depth'],
                                        learning_rate = self.params['learning_rate'],
                                        n_estimators = self.params['n_estimators'],
                                        max_bin = self.params['max_bin'],
                                        objective = self.params['objective'],
                                        min_child_weight=self.params['min_child_weight'],
                                        min_child_sample=self.params['min_child_sample'],
                                        subsample=self.params['subsample'],
                                        colsample_bytree=self.params['colsample_bytree'],
                                        #'reg_alpha':0.01
                                        reg_lambda=self.params['reg_lambda'],
                                        random_state=self.params['random_state'],
                                        n_jobs=self.params['n_jobs'],
                                        silent=self.params['silent'])
        self.estimator.fit(x_train, y_train, eval_set = [(x_eval, y_eval)],
                            eval_names = ['eval data'], eval_metric = 'auc',
                            early_stopping_rounds = self.early_stopping_rounds,
                            feature_name = feature_name,
                            categorical_feature = categorical_feature)
        return self
    
    def GridSearch(self, df):
        feature_name = list(df.columns[2:])
        
        categorical_feature = list(df.dtypes.index[df.dtypes=='object'])
#        df.loc[:,categorical_feature] = \
#            df.loc[:,categorical_feature].replace({np.nan:100}).\
#            astype('int64').replace({100:np.nan})
#            
        df_x, df_y = df.iloc[:,2:], df.TARGET
            
        x_train, x_eval, y_train, y_eval = \
                train_test_split(df_x, df_y,         
                                 test_size = self.test_ratio)
        lgb_clf =  LGBMClassifier(boosting_type = self.params['boosting_type'],
                                  num_leaves = self.params['num_leaves'],
                                  max_depth = self.params['max_depth'],
                                  learning_rate = self.params['learning_rate'],
                                  n_estimators = self.params['n_estimators'],
                                  max_bin = self.params['max_bin'],
                                  objective = self.params['objective'],
                                  min_child_weight=self.params['min_child_weight'],
                                  min_child_sample=self.params['min_child_sample'],
                                  subsample=self.params['subsample'],
                                  colsample_bytree=self.params['colsample_bytree'],
                                  #'reg_alpha':0.01
                                  reg_lambda=self.params['reg_lambda'],
                                  random_state=self.params['random_state'],
                                  n_jobs=self.params['n_jobs'],
                                  silent=self.params['silent']),
        self.params_fit['eval_set'] = [(x_eval, y_eval)]
        self.params_fit['feature_name'] = feature_name
        self.params_fit['categorical_feature'] = categorical_feature
        self.estimator = GridSearchCV(lgb_clf, param_grid = self.params_gs,
                                      scoring = 'auc', fit_params = self.params_fit,
                                      cv = self.nfold)
        self.estimator.fit(x_train, y_train)
        return self.estimator
    
    def transform(self, x_test):
        prediction = self.estimator.predict(x_test)
        return {'prediction':prediction}
        
        
            
    