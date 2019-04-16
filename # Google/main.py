# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 23:04:14 2018

@author: Franc
"""
# In[]
import sys
sys.path.append('src')
import warnings
warnings.simplefilter('ignore')
import gc
gc.enable()
from feature_external import googleExternal
from data_encoder import googleLabelEncoder
labelEncoderProcessor = googleLabelEncoder()
from pipeline_tools import to_pickle, VarDesc
# In[]
featureExternalProcessor = googleExternal()
train = featureExternalProcessor.ExtractInput(train_mode=True)
test = featureExternalProcessor.ExtractInput(train_mode=False)

# In[]
train = featureExternalProcessor.DataCleaner(train,train_mode=True)
test = featureExternalProcessor.DataCleaner(test,train_mode=False)

# In[]
train = featureExternalProcessor.featureHorizonal(train, train_mode=True)
test = featureExternalProcessor.featureHorizonal(test, train_mode=False)
train, test = labelEncoderProcessor.featureCategoryShrink(train, test)
#del train, test
#gc.collect()
# In[]
train = featureExternalProcessor.featureVertical(train)
test = featureExternalProcessor.featureVertical(test)
#del train_hrt, test_hrt
#gc.collect()

# In[]
train_df = featureExternalProcessor.featureExternal(train, train_mode=True)
test_df = featureExternalProcessor.featureExternal(test, train_mode=False)
#del train_vrt, test_vrt
to_pickle(train_df,'train_df.csv')
to_pickle(test_df,'test_df.csv')
#gc.collect()

# In[]
#train_google, test_google = labelEncoderProcessor.transform(train_ext, test_ext)
#feature_categorical = sorted(labelEncoderProcessor.feature_category)
vardesc_google = VarDesc(train_df)
#gc.collect()