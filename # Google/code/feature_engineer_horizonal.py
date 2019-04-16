# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:14:34 2018

@author: Franc
"""
import numpy as np
import pandas as pd
from steppy.base import BaseTransformer

class googleFeatureEngineerHorizonal(BaseTransformer):
    
    def __init__(self):
        super(googleFeatureEngineerHorizonal,self).__init__()
        self.id_feature = [
                'fullVisitorId','sessionId','visitId',
                'date','visitDate','visitTime']
        self.numerical_feature = [
                'totals_hits','totals_pageviews','visitNumber']
    
    def transform(self, data, train, train_mode=True):
        data = data.copy()        
        self.categorical_feature = list(set(data.columns)-\
            set(['totals_transactionRevenue','validRevenue'])-\
            set(self.id_feature+self.numerical_feature))
        data = data.sort_values(['sessionId'])
        
        # device_browser
        data['device_browser_tail'] = data['device_browser'].isin([
                'Chrome', 'Safari', 'Firefox', 'Internet Explorer',
                'Edge', 'Android Webview', 'Safari (in-app)']).astype('int32')
        component = self.valid_ratio(
                train[['validRevenue','device_browser']],'device_browser')
        data = data.merge(component, on=['device_browser'], how='left')
        data['device_browser'] = data['device_browser'].map(
                lambda x: self.browser_transform(str(x).lower())).astype('str')
        data['device_browser_revise'] = np.where(
                data['device_browser'].isin([
                        'Chrome', 'Safari', 'Firefox', 'Internet Explorer',
                        'Edge', 'Android Webview', 'Safari (in-app)']),
                data['device_browser'],
                'others')
        data['device_is_desktop_bool'] = \
            (data['device_deviceCategory']=='desktop').astype('int')
        
        component = self.valid_ratio(
                train[['validRevenue','device_operatingSystem']],'device_operatingSystem')
        data = data.merge(component, on=['device_operatingSystem'], how='left')
        
        # device_operatingSystem
        data['device_operatingSystem_bool'] = data['device_operatingSystem'].isin([
                'Windows','Macintosh','Android','iOS','Linux','Chrome OS']).astype('int32')
        data['device_operatingSystem_dominant_bool'] = data['device_operatingSystem'].isin([
                'Macintosh', 'Linux', 'Chrome OS']).astype('int32')
            
        # geoNetwork
        data['totals_hits_views_ratio'] = \
            (data['totals_hits'] + 1)/(data['totals_pageviews'] + 1)
        data['totals_hits_tail_bool'] = \
            (data['totals_hits']>100).astype('int32')
        data['totals_hits_interval'] = np.where(
                data['totals_hits']==1, 0,
                np.where(data['totals_hits']<4,1,
                         np.where(data['totals_hits']<7,2,
                                  np.where(data['totals_hits']<11,3,\
                np.where(data['totals_hits']<22,4,
                         np.where(data['totals_hits']<33,5,
                                  np.where(data['totals_hits']<49,6,7)))))))
        
        data['totals_pageviews_tail_bool'] = (data['totals_pageviews']>100).astype('int32')
        
        data['totals_pageviews_interval'] = np.where(
                data['totals_pageviews']==1, 0,
                np.where(data['totals_pageviews']<4,1,
                         np.where(data['totals_pageviews']<7,2,
                                  np.where(data['totals_pageviews']<10,3,\
                np.where(data['totals_pageviews']<16,4,
                         np.where(data['totals_pageviews']<24,5,
                                  np.where(data['totals_pageviews']<49,6,7)))))))
        # geoNetwork_networkDomain
        component = self.valid_ratio(
                train[['validRevenue','geoNetwork_networkDomain']],'geoNetwork_networkDomain')
        data = data.merge(component, on=['geoNetwork_networkDomain'], how='left')
        
        component = self.count_ratio(
                data, 'geoNetwork_networkDomain',500)
        data = data.merge(component, on=['geoNetwork_networkDomain'], how='left')
        
        data['geoNetwork_networkDomain_suffix_revise'] = \
            data['geoNetwork_networkDomain_revise'].apply(lambda x: x.split('.')[-1])
            
        data['trafficSource_adContent_tail'] = np.where(
                data['trafficSource_adContent'].notnull(),
                np.where(data['trafficSource_adContent'].isin([
                        'Google Merchandise Collection', 
                        'Google Online Store', 
                        'Full auto ad IMAGE ONLY', '20% discount',
                        '{KeyWord:Google Branded Gear}', 
                        'Swag with Google Logos',
                        '{KeyWord:Want Google Stickers?}']),
                        data['trafficSource_adContent'],'others'),np.nan)
            
        data['trafficSource_adContent'] = data['trafficSource_adContent'].map(
                lambda x: self.adContent_transform(str(x).lower())).astype('str')
        data['trafficSource_keyword_tail'] = np.where(
                data['trafficSource_keyword'].notnull(),
                np.where(data['trafficSource_keyword'].isin([
                        '(not provided)', '6qEhsCssdK0z36ri', 
                        '1hZbAqLCbjwfgOH7', '(Remarketing/Content targeting)',
                        '1X4Me6ZKNV0zg-jV']), data['trafficSource_keyword'],
                    np.where(data['trafficSource_keyword'].isin([
                            'google merchandise store','google store',
                            'Google Merchandise','+Google +Merchandise']),
                            'Google','others')),np.nan)
        ## trafficSource_referralPath
        component = self.count_ratio(data, 'trafficSource_referralPath',50)
        data = data.merge(component, on=['trafficSource_referralPath'], how='left')
        
        data['trafficSource_source_revise'] = np.where(
                data['trafficSource_source'].isin([
                        'mall.googleplex.com','google','(direct)',
                        'dfa','mail.google.com','sites.google.com',
                        'dealspotr.com','groups.google.com','yahoo','bing',
                        'facebook.com','gdeals.googleplex.com','l.facebook.com',
                        'youtube.com','Partners','t.co','m.facebook.com',
                        'siliconvalley.about.com','phandroid.com',
                        'google.com','plus.google.com','ask','mg.mail.yahoo.com',
                        'googleux.perksplus.com','connect.googleforwork.com',
                        'pinterest.com','trainup.withgoogle.com','keep.google.com',
                        'basecamp.com','outlook.live.com','search.myway.com',
                        'search.xfinity.com','us-mg5.mail.yahoo.com',
                        'gatewaycdi.com','seroundtable.com','chat.google.com',
                        'calendar.google.com','l.messenger.com','quora.com',
                        'mail.aol.com','moma.corp.google.com','reddit.com',
                        'docs.google.com']),
                data['trafficSource_source'],
                'others')
            
        data['trafficSource_source'] = data['trafficSource_source'].map(
                lambda x: self.source_transform(str(x).lower())).astype('str')
        
        data['Extra_source_country'] = \
            data['trafficSource_source'] + '_' + data['geoNetwork_country']
        data['Extra_campaign_medium'] = \
            data['trafficSource_campaign'] + '_' + data['trafficSource_medium']
        data['Extra_browser_category'] = \
            data['device_browser'] + '_' + data['device_deviceCategory']
        data['Extra_browser_os'] = \
            data['device_browser'] + '_' + data['device_operatingSystem']
        
        data['Extra_device_deviceCategory_channelGrouping'] = \
            data['device_deviceCategory'] + "_" + data['channelGrouping']
        data['Extra_channelGrouping_browser'] = \
            data['device_browser'] + "_" + data['channelGrouping']
        data['Extra_channelGrouping_OS'] = \
            data['device_operatingSystem'] + "_" + data['channelGrouping']
    
        for region in ['geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country',
                       'geoNetwork_metro', 'geoNetwork_networkDomain', 
                       'geoNetwork_region','geoNetwork_subContinent']:
            for device in ['device_browser','device_deviceCategory', 
                           'device_operatingSystem', 'trafficSource_source']:
                data['Extra_' + region + '_' + device] = data[region] + '_' + data[device]
        
        data['Extra_content_source'] = \
            data['trafficSource_adContent'] + "_" + data['Extra_source_country']
        data['Extra_medium_source'] = \
            data['trafficSource_medium'] + "_" + data['Extra_source_country']
        
        """ visit time """
        data['visitHour_interval'] = np.where(
                (0<data['visitHour'])&(data['visitHour']<9),1,
                np.where((13<data['visitHour'])&(data['visitHour']<21),3,2))
        data['visitWeekday_weekend_bool'] = \
            ((0<data['visitWeekday'])&(data['visitWeekday']<7)).astype('int32')

        self.manual_categorical = \
            data.columns[data.columns.str.startswith('Extra')].tolist()+\
            data.columns[data.columns.str.endswith(('tail','revise','interval'))].tolist()
            
        self.manual_numerical = \
            data.columns[data.columns.str.endswith(('ratio','bool'))].tolist()
            
        self.categorical_feature = \
            list(set(self.categorical_feature+self.manual_categorical))
        self.numerical_feature = \
            list(set(self.numerical_feature+self.manual_numerical))
        cols = self.id_feature + \
            sorted(self.categorical_feature+self.numerical_feature)
        if train_mode:
            cols = ['totals_transactionRevenue','validRevenue']+cols
        data = data.reindex(columns = cols)
        return data
            
    def valid_ratio(self, data, feature):
        component = pd.crosstab(
                data['validRevenue'], data[feature], margins=True).T.reset_index()
        component[f'{feature}_ratio'] = np.log(component[1]+1/(component[0]+1))
        return component[[feature,f'{feature}_ratio']]
    
    def feature_count_ratio(self, data, feature, topK):
        component = data[feature].value_counts().sort_values(ascending=False).\
            reset_index().rename(
                    columns={'index':feature, feature:f'{feature}_count'})
        component[f'{feature}_count_ratio'] = \
        component[f'{feature}_count']/len(data)
        feature_include = component[feature].tolist()[:topK]
        component[f'{feature}_revise'] = np.where(
                component[feature].isin(feature_include),
                component[feature], 'others')
        return component
    
    def browser_transform(self, x):
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
        
        
    def adContent_transform(self, x):
        if  ('google' in x):
            return 'google'
        elif  ('placement' in x) | ('placememnt' in x):
            return 'placement'
        elif '(not set)' in x or 'nan' in x:
            return x
        elif 'ad' in x:
            return 'ad'
        else:
            return 'others'
        
    def source_transform(self, x):
        if  ('google' in x):
            return 'google'
        elif  ('youtube' in x):
            return 'youtube'
        elif '(not set)' in x or 'nan' in x:
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