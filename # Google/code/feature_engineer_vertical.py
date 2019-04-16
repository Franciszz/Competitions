# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 16:46:06 2018

@author: Franc
"""

import numpy as np
from steppy.base import BaseTransformer
from datetime import datetime

class googleFeatureEngineerVertical(BaseTransformer):
    
    def __init__(self):
        super(googleFeatureEngineerVertical,self).__init__()
        self.manual_feature = []

    def transform(self, data):
        data = data.copy()
        data['visitId'] = data['visitId'].apply(datetime.fromtimestamp)
        data = data.sort_values(['fullVisitorId','visitId','date'])
        
        judge_id_multi = data['fullVisitorId'].value_counts().\
            sort_values(ascending=False)
        id_multi_category = judge_id_multi.index[judge_id_multi>1].tolist()
        
        data_multivisit = data[data['fullVisitorId'].isin(id_multi_category)]
        data_multivisit = self.transform_multi(data_multivisit)

        data_univisit = data[~(data['fullVisitorId'].isin(id_multi_category))]
        data_univisit = data_univisit.reindex(
                columns = data_multivisit.columns.tolist())
        data_univisit[self.manual_numerical]=0
        data = data_multivisit.append(data_univisit)
        return data
    
    def transform_multi(self, data):
        data['visitMultiple_bool'] = 1
        data = data.groupby(['fullVisitorId']).apply(self.transform_indivisual)
        return data
    
    def transform_indivisual(self, data):
        visit_timedelta = self.transform_delta(data, 'visitId')
        data['visit_days_timedelta'] = \
            visit_timedelta.apply(lambda x: x.days if x else np.nan)
        data['visit_seconds_timedelta'] = \
            visit_timedelta.apply(lambda x: x.seconds if x else np.nan)
        data['visit_hours_timedelta'] = \
            (data['visit_seconds_timedelta']/3600).fillna(0).astype('int32')
        data['visit_timedelta_minutes'] = \
            (data['visit_seconds_timedelta']/60).fillna(0).astype('int32')
        
        data['visit_inday_bool'] = self.transform_delta(
                data,'visitId', delta='equals')
        
        component = data['visitDate'].value_counts().reset_index().\
            rename(columns={'index':'visitDate', 
                            'visitDate':'visitDate_count_delta'})
        data = data.merge(component, on=['visitDate'], how='left')
        
        feature_to_judge = [
                'channelGrouping','device_browser','device_deviceCategory',
                'device_operatingSystem','geoNetwork_city',
                'geoNetwork_continent','geoNetwork_country',
                'geoNetwork_metro','geoNetwork_region',
                'totals_hits_interval','totals_pageviews_interval',
                'trafficSource_source','visitHour_interval']
        for feature in feature_to_judge:
            data[f'{feature}_changed'] = \
                self.transform_delta(data, feature, delta='equals')
        
        data['visit_hits_delta'] = \
            self.transform_delta(data,'totals_hits')
        data['visit_hits_ratio'] = \
            self.transform_delta(data,'totals_hits', delta = 'div')
        data['visit_pageviews_delta'] = \
            self.transform_delta(data,'totals_pageviews')
        data['visit_pageviews_ratio'] = \
            self.transform_delta(data,'totals_pageviews', delta = 'div')
            
        data['visit_hits_views_ratio_delta'] = \
            self.transform_delta(data,'totals_hits_views_ratio')
        data['visit_hits_views_ratio_ratio'] = \
            self.transform_delta(data,'totals_hits_views_ratio', delta = 'div')
        return data
    
    def transform_delta(self, data, feature, delta=None, shiftdelta=1):
        df = data.copy()
        df[f'{feature}_adient'] = df[feature].shift(shiftdelta)
        if not delta:
            return df[feature]-df[f'{feature}_adient']
        elif delta == 'equals':
            return (df[f'{feature}_adient']==df[feature]).astype('int32')
        elif delta == 'div':
            return df[feature]/(df[f'{feature}_adient']+1)
        