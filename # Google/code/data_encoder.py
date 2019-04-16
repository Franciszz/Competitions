# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 18:26:22 2018

@author: Franc
"""

import pandas as pd

from steppy.base import BaseTransformer
from steppy.utils import get_logger
logger = get_logger()

class googleLabelEncoder(BaseTransformer):
    
    def __init__(self):
        super(googleLabelEncoder,self).__init__()
    
    def transform(self, train, test):
        train, test = train.copy(), test.copy()
        train.replace({-2147483648:1},inplace=True)
        test.replace({-2147483648:1},inplace=True)
        self.feature_id = test.columns[:6].tolist()
        self.feature_df = list(set(test.columns)-set(self.feature_id)) 
        self.feature_category = test.columns[
                ((test.dtypes=='object')|
                (test.columns.str.endswith('interval')))&
                ~(test.columns.isin(self.feature_id))].tolist()+\
                ['visitHour','visitMonth','visitWeekday']
        feature_to_factorize = [
                _feature for _feature in self.feature_category if \
                test[_feature].dtype=='object']
        for _feature in feature_to_factorize:
            train[_feature], _feature_indexer = pd.factorize(train[_feature])
            test[_feature] = _feature_indexer.get_indexer(test[_feature])
        return train, test
            