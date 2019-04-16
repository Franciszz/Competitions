# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 17:18:44 2018

@author: Franc
"""

from datetime import datetime
from feature_horizonal import googleHorizonal

class googleVertical(googleHorizonal):
    
    def __init__(self):
        super(googleHorizonal, self).__init__()
        
    def featureVertical(self, df):
        df = df.copy()
        df['visitId'] = df['visitId'].apply(datetime.fromtimestamp)
        df = df.sort_values(['fullVisitorId','visitId','date'])
        
        id_multi_category = df.loc[df['fullVisitorId'].duplicated(),
                                   'fullVisitorId'].unique().tolist()
        df_multivisit = df[df['fullVisitorId'].isin(id_multi_category)]
        df_multivisit = self.transform_multi(df_multivisit)

        df_univisit = df[~(df['fullVisitorId'].isin(
                id_multi_category))].reset_index()
        df_univisit = df_univisit.reindex(
                columns = df_multivisit.columns.tolist())
        df_univisit[self.vertical_feature] = 0
        
        df = df_multivisit.append(df_univisit).sort_values(
                ['fullVisitorId','visitId','date'])
        return df
    
    def transform_multi(self, df):
        cols_origin = df.columns
        df['visitMultiple'] = 1
        groupby_df = df.groupby(['fullVisitorId'],as_index=False)
        feature_to_judge = [
                'channelGrouping','device_browser','device_deviceCategory',
                'device_operatingSystem','geoNetwork_city',
                'geoNetwork_continent','geoNetwork_country',
                'geoNetwork_metro','geoNetwork_region','visitDate',
                'totals_hits_interval','totals_pageviews_interval',
                'trafficSource_source','visitHour_interval']
        shift_df = groupby_df[[
                'visitId','totals_hits','totals_pageviews','totals_high_visit',
                'totals_hits_pageviews_ratio']+feature_to_judge].\
                shift(1).reset_index(drop=True)
        
        """ visitTime """
        visit_timedelta = df['visitId']-shift_df['visitId']
        df['visit_days_timedelta'] = visit_timedelta.dt.days
        df['visit_seconds_timedelta'] = visit_timedelta.dt.seconds
        df['visit_hours_timedelta'] = \
            (df['visit_seconds_timedelta']/3600).fillna(0).astype('int32')
        df['visit_minutes_timedelta'] = \
            (df['visit_seconds_timedelta']/60).fillna(0).astype('int32')
        
        """ visit inday """
        component = df.groupby(['fullVisitorId','visitDate']).size().\
            reset_index().rename(columns={0:'visit_nums_inday'})
        df = df.merge(component, on=['fullVisitorId','visitDate'], how='left')
        
        """ visit hits & pageviews """
        df['visit_totals_high_visit_last'] = shift_df['totals_high_visit']
        df['visit_hits_delta'] = df['totals_hits']-shift_df['totals_hits']
        df['visit_hits_ratio'] = df['totals_hits']/shift_df['totals_hits']
        df['visit_pageviews_delta'] = \
            df['totals_pageviews']-shift_df['totals_pageviews']
        df['visit_pageviews_ratio'] = \
            df['totals_pageviews']/shift_df['totals_pageviews']
        df['visit_hits_pageviews_ratio_delta'] = \
            df['totals_hits_pageviews_ratio']-shift_df[
                    'totals_hits_pageviews_ratio']       
        df['visit_hits_pageviews_ratio_ratio'] = \
            df['totals_hits_pageviews_ratio']/shift_df[
                    'totals_hits_pageviews_ratio'] 
        """ other categorical features """
        judge_equal_df = (df[feature_to_judge]==shift_df[feature_to_judge]).\
            astype('int32')
        judge_equal_df.columns = [_f+'_changed' for _f in feature_to_judge]
        df = df.join(judge_equal_df)
        self.vertical_feature = list(set(df.columns)-set(cols_origin))
        return df