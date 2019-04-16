# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 20:28:46 2018

@author: Franc
"""

import pickle

def from_pickle(filename):
    with open('cache/'+filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def to_pickle(obj, filename):
    with open('cache/'+filename, 'wb') as f:
        pickle.dump(obj, f, -1)
    


