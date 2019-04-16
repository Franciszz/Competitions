# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:56:02 2018

@author: Franc
"""

import pandas as pd
from datetime import datetime
from data_input import googleInput

class googleCleaner(googleInput):
    
    def __init__(self):
        super(googleCleaner, self).__init__()