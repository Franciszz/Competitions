# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 07:47:27 2018

@author: Franc
"""
# Regressor Parameters
params = {
        'xgb_params':{
                'objective': 'reg:linear',
                'booster': 'gbtree',
                'learning_rate': 0.02,
                'max_depth': 10,
                'min_child_weight': 57,
                'gamma' : 1.45,
                'alpha': 0.0,
                'lambda': 0.0,
                'subsample': 0.67,
                'colsample_bytree': 0.054,
                'colsample_bylevel': 0.50,
                'n_jobs': -1,
                'random_state': 9611},
        'cat_params':{
                'learning_rate' :0.03,
                'depth' :10,
                'eval_metric' :'RMSE',
                'od_type' :'Iter',
                'metric_period ' : 50,
                'od_wait' : 20,
                'seed' : 9611},
        'lgb_params':{
                'learning_rate': 0.025,
                'objective':'regression',
                'metric':'rmse',
                'num_leaves': 31,
                'verbose': 1,
                'min_child_samples':100,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'random_state':9611,
                'use_best_model':True,
#                'max_depth': 10,
#                'lambda_l2': 0.02085548700474218,
#                'lambda_l1': 0.004107624022751344,
                'bagging_fraction': 0.7934712636944741,
                'bagging_frequency':5,
                'feature_fraction': 0.686612409641711,
#                'min_child_samples': 21
                }
        }
# validRevenue 
validRevenueRatio = [892138, 11515]
# 
parameters = {
        'cat_params':
            {'params_clf':{
#                    """ 核心参数 """
                    'objective': 'Logloss',# CrossEntropy, Logloss, MAE...
                    'bootstrap_type': 'Bayesian',
                    'iterations': 2000,
                    'learning_rate': 0.01, 
                    
#                    """ 学习控制参数 """
                    'eval_metric': 'Logloss',
                    'random_seed': 960101,
#                    'l2_leaf_reg': 1e-2,
                    'subsample': 0.9,
                    'use_best_model': True,
                    'depth': 9,
#                    'one_hot_max_size': 8,
                    'rsm':1, # colsample_bylevel
                        
#                    """ 目标参数 """
                    'bagging_temperature': 1, # only in bayesian
                    'scale_pos_weight': 75, # only in binary
                    
#                    """ 度量参数 """
                    'custom_metric':'Accuracy'#AUC, F1, Accuracy, Logloss, MAE...
                    },
             'params_reg':{
#                    """ 核心参数 """
                    'objective': 'RMSE',# CrossEntropy, Logloss, MAE...
#                    'bootstrap_type': 'Bayesian',
                    'iterations': 2000,
                    'learning_rate': 0.1, 
                    
#                    """ 学习控制参数 """
                    'random_seed': 960101,
                    'l2_leaf_reg': 1,
                    'subsample': 0.9,
                    'use_best_model': True,
                    'depth': 9,
#                    'one_hot_max_size': 8,
                    'rsm':1, # colsample_bylevel
                        
#                    """ 目标参数 """
#                    'bagging_temperature': 1, # only in bayesian
#                    'scale_pos_weight': 75, # only in binary
#                    
#                    """ 度量参数 """
                    'custom_metric':'RMSE'#AUC, F1, Accuracy, Logloss, MAE...
                    },
             'seed': 960101,
             'eval_ratio': 0.25,
             'num_boost_round': 2000,
             'nfold': 5},
        'lgb_params':
            {'params_clf':{
#                    """ 核心参数 """
                    'objective': 'binary', #binary, regression_l1, regression_l2
                    'boosting_type': 'goss', #rf, dart, goss
                    'num_iterations': 2000,
                    'learning_rate': 0.1,
#                    'num_leaves': 48,
                    'num_threads': 4,
#                    'max_bin': 250,
                    
#                    """ 学习控制参数 """
                    'max_depth': 9,
                    'min_data_in_leaf': 25,
                    'min_sum_hessian_in_leaf':1e-3,
                    'feature_fraction':1,
                    'feature_fraction_seed':960101,
#                    'bagging_fraction':1,
#                    'bagging_freq':0,
#                    'bagging_seed':960101,
                    'early_stopping_round':200,
#                    'lambda_l1':1e-2,
#                    'lambda_l2':1e-2,
#                    'min_split_gain':0,
#                    'drop_rate':0.05,# only in dart
#                    'skip_drop':0.25,# only in dart
#                    'max_drop':50,# only in dart
#                    'uniform_drop':False,# only in dart
#                    'xgboost_dart_mode':False,# only in dart
#                    'top_rate': 0.3,# only in goss, ratio of large gradient
#                    'other_rate': 0.1, # only in goss, ratio of small gradient
#                    'min_data_per_group':100,
#                    'max_cat_threshold':32,
                    'cat_smooth':10,
                    'max_cat_to_onehot':8, # threshold to use one_hot encoding method
#                    'top_k': 20, # parallel
                    
#                    """ 目标参数 """
                    'scale_pos_weight': 75, # only in binary classification
#                    'boosting_from_average': True, # only in regression
#                    'is_unbalance':True, # in binary classification
                    
#                    """ 度量参数 """
                    'metric':'binary_logloss'# mse, mae, auc, binary_logloss, binary_error, cross_entrypy
                    },
            'params_reg':{
                    #""" 核心参数 """
                    'objective': 'regression', #binary, regression_l1, regression_l2
#                    'boosting_type': 'goss', #rf, dart, goss
                    'num_iterations': 2000,
                    'learning_rate': 0.02,
                    'num_leaves': 30,
#                    'num_threads': 4,
#                    'max_bin': 250,
                    
                    # 学习控制参数
                    'max_depth': 8,
#                    'min_data_in_leaf': 10,
#                    'min_sum_hessian_in_leaf':1e-3,
                    'feature_fraction':0.9,
#                    'feature_fraction_seed':960101,
#                    'bagging_fraction':0.7,
#                    'bagging_freq':5,
#                    'bagging_seed':960101,
                    'early_stopping_round':200,
#                    'lambda_l1':1e-2,
#                    'lambda_l2':1e-2,
#                    'min_split_gain':0,
#                    'drop_rate':0.05,# only in dart
#                    'skip_drop':0.25,# only in dart
#                    'max_drop':50,# only in dart
#                    'uniform_drop':False,# only in dart
#                    'xgboost_dart_mode':False,# only in dart
#                    'top_rate': 0.3,# only in goss, ratio of large gradient
#                    'other_rate': 0.3, # only in goss, ratio of small gradient
#                    'min_data_per_group':100,
#                    'max_cat_threshold':32,
#                    'cat_smooth':10,
#                    'max_cat_to_onehot':8, # threshold to use one_hot encoding method
#                    'top_k': 20, # parallel
                    
                    #""" 目标参数 """
#                    'scale_pos_weight': 75, # only in binary classification
#                    'boosting_from_average': True, # only in regression
#                    'is_unbalance':True, # in binary classification
                    
                    #""" 度量参数 """
                    'metric':'rmse'# mse, mae, auc, binary_logloss, binary_error, cross_entrypy
                    },
            'seed': 960101,
            'eval_ratio': 0.25,
            'num_boost_round': 2000,
            'early_stopping_rounds': 200,
            'nfold':5},
        'XGBParams':{},
        'GBDTParams':{},
        'AdaParams':{}
        }

# 
features = {
        'target_feature':
            ['totals_transactionRevenue','validRevenue'],
        'id_feature':
            ['fullVisitorId','visitId','date','visitDate','visitTime'],
        'numerical_feature':
            ['device_browser_ratio','device_is_desktop','device_operatingSystem_ratio',
             'device_browser_tail','device_operatingSystem_dominant',
             'device_operatingSystem_revise',
             'geoNetwork_networkDomain_count','geoNetwork_networkDomain_count_ratio',
             'geoNetwork_networkDomain_ratio','totals_hits','totals_hits_views_ratio',
             'totals_pageviews','trafficSource_referralPath_count',
             'trafficSource_referralPath_count_ratio','visitNumber',
             'totals_hits_tail','totals_pageviews_tail',
             'visitMultiple','visit_timedelta_days','visit_timedelta_seconds',
             'visit_timedelta_hours','visit_timedelta_minutes','visit_inday',
             'visitDate_count','channelGrouping_is_changed','visitWeekday_weekend',
             'device_browser_is_changed','device_deviceCategory_is_changed',
             'device_operatingSystem_is_changed','geoNetwork_city_is_changed',
             'geoNetwork_continent_is_changed','geoNetwork_country_is_changed',
             'geoNetwork_metro_is_changed','geoNetwork_networkDomain_is_changed',
             'geoNetwork_region_is_changed','geoNetwork_subContinent_is_changed',
             'totals_hits_interval_is_changed','totals_pageviews_interval_is_changed',
             'trafficSource_campaign_is_changed','trafficSource_keyword_is_changed',
             'trafficSource_medium_is_changed','trafficSource_referralPath_is_changed',
             'trafficSource_source_is_changed','visitHour_interval_is_changed',
             'visit_hits_delta','visit_hits_ratio','visit_pageviews_delta',
             'visit_pageviews_ratio','visit_hits_views_ratio_delta',
             'visit_hits_views_ratio_ratio','geoNetwork_city_count',
             'geoNetwork_city_count_ratio','geoNetwork_city_ratio',
             'geoNetwork_country_count','geoNetwork_country_count_ratio',
             'geoNetwork_country_ratio','geoNetwork_region_count',
             'geoNetwork_region_count_ratio','geoNetwork_region_ratio',
             'visit_is_local_holiday','visit_is_us_holiday',
             'visit_is_us_holiday_delta_1','visit_is_us_holiday_delta_2',
             'visit_is_us_holiday_delta_3','visit_is_us_holiday_delta_4',
             'visit_is_us_holiday_delta_5','external_usdx_index',
             'external_employment','external_unemployment','external_rate'],
        'categorical_feature':
            ['channelGrouping','device_browser','device_browser_revise',
             'device_deviceCategory','device_operatingSystem',
             'geoNetwork_city','geoNetwork_continent','geoNetwork_country',
             'geoNetwork_metro','geoNetwork_networkDomain_revise',
             'geoNetwork_networkDomain_suffix','geoNetwork_region',
             'geoNetwork_subContinent','totals_hits_interval',
             'totals_pageviews_interval','trafficSource_campaign',
             'trafficSource_medium','visitMonth','trafficSource_referralPath_revise',
             'trafficSource_source','trafficSource_source_revise','visitHour',
             'visitHour_interval','visitWeekday','trafficSource_keyword_tail',
             'geoNetwork_city_revise','geoNetwork_country_revise',
             'geoNetwork_region_revise']}
