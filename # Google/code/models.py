# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 11:06:09 2018

@author: Franc
"""
import datetime

from steppy.base import BaseTransformer
from steppy.utils import get_logger

import catboost as cat
import lightgbm as lgb

#from sklearn.model_selection import train_test_split

from pipeline_cache import setseed
from pipeline_config import parameters, features
logger = get_logger()

class CatBoost(BaseTransformer):
    def __init__(self):
        self.estimator = None
        self.params = parameters['CatParams']
        self.nfold = parameters['CatParams']['nfold']
        self.seed = parameters['CatParams']['seed']
        self.eval_ratio = parameters['CatParams']['eval_ratio']
        self.num_boost_round = parameters['CatParams']['num_boost_round']
        self.learning_rates = parameters['CatParams']['params_clf']['learning_rate']
        self.categorical_feature = sorted(features['categorical_feature'])
        self.numerical_feature = sorted(features['numerical_feature'])
        self.features_name = sorted(features['categorical_feature']+\
                                    features['numerical_feature'])
#    def loglikelood(self, preds, train_data):
#        labels = train_data.get_label()
#        preds = 1. / (1. + np.exp(-preds))
#        grad = preds - labels
#        hess = preds * (1. - preds)
#        return grad, hess
#    
#    def roc_auc_error(self, preds, train_data):
#        labels = train_data.get_label()
#        return 'error', 1 - roc_auc_score(labels, preds)
    
    def fit(self, data, clf = None):
        setseed(self.seed)
        train_df = data[data['visitId'].\
                        apply(lambda x: x.date()) >= datetime.date(2016,9,30)]
        val_df = data[data['visitId'].\
                      apply(lambda x: x.date()) < datetime.date(2016,9,30)]
        if clf:
            train_X, train_y = \
                train_df[self.features_name].values,\
                train_df['validRevenue'].values\
                
            val_X, val_y = \
                val_df[self.features_name].values,\
                val_df['validRevenue'].values
        else:
            train_X, train_y = \
                train_df[self.features_name].values,\
                train_df['totals_transactionRevenue'].values
                
            val_X, val_y = \
                val_df[self.features_name].values,\
                val_df['totals_transactionRevenue'].values
            
        
#        x_train, x_eval, y_train, y_eval = \
#                train_test_split(df_x, df_y,         
#                             test_size = self.test_ratio,
#                             random_state = self.seed)
            
        cat_train = cat.Pool(
                data = train_X, label = train_y,
                feature_names = self.features_name,
                cat_features = self.categorical_feature)
        
        cat_eval = cat.Pool(
                data = val_X, label = val_y,
                feature_names = self.features_name,
                cat_features = self.categorical_feature)
        
        model_param = self.params['params_clf'] if clf else self.params['params_reg']
        
        self.estimator = cat.train(
                params = model_param,
                pool = cat_train,
                eval_set = cat_eval,
#                num_boost_round = self.num_boost_round,
#                learning_rate = self.params['learning_rate'],
#                max_depth = self.params['max_depth'],
#                l2_leaf_reg = self.params['l2_leaf_reg'],
#                rsm = self.params['colsample_ratio'],
#                subsample = self.params['subsample_ratio'],
#                class_weights = self.params['class_weights'],
#                loss_function = self.loglikeloss,
#                custom_loss = self.loglikeloss,
#                custom_metric = self.roc_auc_error,
#                eval_metric = self.roc_auc_error
                )
        
        return self
    
    def cv(self, data, clf=None):
        setseed(self.seed)
        if clf:
            train_X, train_y = \
                data[self.features_name].values,\
                data['validRevenue'].values
        else:
            train_X, train_y = \
                data[self.features_name].values,\
                data['totals_transactionRevenue'].values
                
        cat_train = cat.Pool(data = train_X, label = train_y,
                             feature_names = self.features_name,
                             cat_features = self.categorical_feature)
        
        cat_cv_hist = cat.cv(pool = cat_train, 
                             params = self.params,
#                             num_boost_round = self.num_boost_round,
                             nfold = self.nfold,
                             seed = self.seed)
                
        return cat_cv_hist
    
    def predict(self, data):
        test_X = data[self.features_name]
        cat_test = cat.Pool(data = test_X,
                            feature_names = self.features_name,
                            cat_features = self.categorical_feature)
        prediction = self.estimator.predict(cat_test)
        return prediction
    
class LightGBM(BaseTransformer):
    def __init__(self):
        self.estimator = None
        self.params = parameters['LgbParams']
        self.nfold = parameters['LgbParams']['nfold']
        self.seed = parameters['LgbParams']['seed']
        self.eval_ratio = parameters['LgbParams']['eval_ratio']
        self.num_boost_round = parameters['LgbParams']['num_boost_round']
        self.learning_rates = parameters['LgbParams']['params_clf']['learning_rate']
        self.early_stopping_rounds = parameters['LgbParams']['early_stopping_rounds']
        self.categorical_feature = features['categorical_feature']
        self.numerical_feature = features['numerical_feature']
        self.features_name = sorted(features['categorical_feature']+\
                                    features['numerical_feature'])
        
#    def loglikelood(self, preds, train_data):
#        labels = train_data.get_label()
#        preds = 1. / (1. + np.exp(-preds))
#        grad = preds - labels
#        hess = preds * (1. - preds)
#        return grad, hess
#    
#    def roc_auc_error(self, preds, train_data):
#        labels = train_data.get_label()
#        return 'error', 1-roc_auc_score(labels, preds), False
    
    def fit(self, data, clf=None):
        setseed(self.seed)
        train_df = data[data['visitId'].\
                        apply(lambda x: x.date()) >= datetime.date(2016,9,30)]
        val_df = data[data['visitId'].\
                      apply(lambda x: x.date()) < datetime.date(2016,9,30)]
        if clf:
            train_X, train_y = \
                train_df[self.features_name],\
                train_df['validRevenue'].values\
                
            val_X, val_y = \
                val_df[self.features_name],\
                val_df['validRevenue'].values
        else:
            train_X, train_y = \
                train_df[self.features_name],\
                train_df['totals_transactionRevenue'].values
                
            val_X, val_y = \
                val_df[self.features_name],\
                val_df['totals_transactionRevenue'].values
            
#        x_train, x_eval, y_train, y_eval = \
#            train_test_split(x_train_eval, y_train_eval, 
#                             test_size = self.eval_ratio,
#                             random_state = self.seed)
            
        lgb_train = \
                lgb.Dataset(train_X, train_y, 
                            feature_name = self.features_name,
                            categorical_feature = self.categorical_feature)
        lgb_eval = \
                lgb.Dataset(val_X, val_y, 
                            feature_name = self.features_name,
                            categorical_feature = self.categorical_feature)
        model_param = self.params['params_clf'] if clf else self.params['params_reg']
        self.estimator = \
                lgb.train(params = model_param,
                          train_set = lgb_train,
                          num_boost_round = self.num_boost_round,
#                          init_model = None,
                          valid_sets = lgb_eval,
#                          fobj = self.loglikelood,
#                          feval = self.roc_auc_error,
                          early_stopping_rounds = self.early_stopping_rounds,
#                          learning_rates = self.learning_rates,
#                              lambda x: self.learning_rates*(0.99**x),
                          verbose_eval = True,
                          feature_name = self.features_name,
                          categorical_feature = self.categorical_feature)
        
        return self
    
    def cv(self, data, clf=None):
        setseed(self.seed)
        if clf:
            train_X, train_y = \
                data[self.features_name],\
                data['validRevenue'].values\

        else:
            train_X, train_y = \
                data[self.features_name],\
                data['totals_transactionRevenue'].values
                
        lgb_train = lgb.Dataset(train_X, train_y, 
                                feature_name = self.features_name,
                                categorical_feature = self.categorical_feature,
                                free_raw_data = False)
        model_param = self.params['params_clf'] if clf else self.params['params_reg']
        lgb_cv_hist = \
                lgb.cv(params = model_param,
                       nfold = self.nfold,
                       train_set = lgb_train,
#                       num_boost_round = self.num_boost_round,
#                       fobj = self.loglikelood,
#                       feval = self.roc_auc_error,
                       verbose_eval = True,
                       feature_name = self.features_name,
                       categorical_feature = self.categorical_feature)
        return lgb_cv_hist
    
    def transform(self, data):
        test_X = data[self.features_name]
        prediction = self.estimator.predict(test_X, \
                            num_iteration = self.estimator.best_iteration)
        return prediction      