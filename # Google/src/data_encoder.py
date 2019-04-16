# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:12:58 2018

@author: Franc
"""

import pandas as pd
import numpy as np
from steppy.base import BaseTransformer
from steppy.utils import get_logger
logger = get_logger()

class googleLabelEncoder(BaseTransformer):
    
    def __init__(self):
        super(googleLabelEncoder,self).__init__()
    
    def featureCategoryShrink(self, train, test, limits=1600):
        df = pd.concat([train, test], axis=0, sort = False).fillna(0)
        feature_to_shrink = [
                'device_operatingSystem','geoNetwork_country',
                'geoNetwork_city','geoNetwork_metro',
                'geoNetwork_networkDomain','geoNetwork_region',
                'geoNetwork_subContinent','trafficSource_adContent',
                'trafficSource_campaign','trafficSource_keyword',
                'trafficSource_medium','trafficSource_referralPath',
                'trafficSource_source']
        for feature in feature_to_shrink:
            component = df[feature].value_counts()
            component =  set(component.index[component>limits].values)
            df[feature] = np.where(
                    df[feature].isin(component),df[feature],'others')
            
        for feature in ['totals_hits', 'totals_pageviews']:
            info = df.groupby("fullVisitorId")[feature].mean()
            df['usermean_' + feature] = df.fullVisitorId.map(info)
            
        for feature in ['visitNumber','Avg. Session Duration']:
            info = df.groupby('fullVisitorId')[feature].max()
            df["usermax_" + feature] = df.fullVisitorId.map(info)
            
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
        return df.iloc[:len(train),:], df.iloc[len(train):,:]
            
    def transform(self, train, test):
        train, test = train.copy(), test.copy()
        train.replace({-2147483648:1},inplace=True)
        test.replace({-2147483648:1},inplace=True)
        feature_to_drop = [
                'geoNetwork_networkDomain','device_isMobile',
                'trafficSource_adwordsClickInfo.gclId',
                'trafficSource_adwordsClickInfo.page',
                'trafficSource_adwordsClickInfo.adNetworkType',
                'trafficSource_adwordsClickInfo.slot',
                'trafficSource_referralPath']
        self.feature_id = test.columns.tolist()[:6]
        self.feature_df = list(
                set(train.columns)-set(self.feature_id)-\
                set(feature_to_drop)-set(['totals_transactionRevenue'])) 
        self.feature_category = test.columns[
                ((test.dtypes=='object')|
                (test.columns.str.endswith('interval')))&
                ~(test.columns.isin(
                        self.feature_id+feature_to_drop))].tolist()+\
                ['visitHour','visitMonth','visitWeekday']
        feature_to_factorize = [
                _feature for _feature in self.feature_category if \
                test[_feature].dtype=='object']
        for _feature in feature_to_factorize:
            train[_feature], _feature_indexer = pd.factorize(train[_feature])
            test[_feature] = _feature_indexer.get_indexer(test[_feature])
        cols = self.feature_id+sorted(self.feature_df)
        train = train.reindex(columns=['totals_transactionRevenue']+cols)
        test = test.reindex(columns = cols)
        return train, test
            