# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:53:19 2018

@author: Franc
"""
import numpy as np
import os 
files = os.listdir('prediction10')
files_test = [x for x in files if x.startswith('a')]
files_sample = [x for x in files if x.startswith('s')]
obj = pd.read_csv('prediction10/%s'%files_test[0]).sort_values('SK_ID_CURR').reset_index(drop=True)
y_true = df_all.loc[df_all.SK_ID_CURR.isin(obj.SK_ID_CURR),['SK_ID_CURR','TARGET']]

#for i in range(96,110):  
for file in files_test:
    pred = pd.read_csv('prediction10/%s'%file).sort_values('SK_ID_CURR')
#    pred = np.where(pred.TARGET+pred.TARGET_CAT>1,
#                    pred[['TARGET','TARGET_CAT']].max(axis=1),
#                    pred[['TARGET','TARGET_CAT']].min(axis=1), 
#                             pred[['TARGET','TARGET_CAT']].mean(axis=1)))
    #y_true[file] = pred.TARGET
    y_true = y_true.merge(pred, on = 'SK_ID_CURR', how='left')
    #print(roc_auc_score(y_true.iloc[:,1:2], pred.TARGET))
y_true['mean'] = y_true.iloc[:,2:].mean(axis=1)

#roc_auc_score(y_true.iloc[:,1:2], y_true['mean'])

submit = pd.DataFrame(dict(SK_ID_CURR = y_true['SK_ID_CURR'].astype('int'),
                           TARGET = y_true['mean']))
submit.to_csv('prediction10/asubmit_3.csv',index=False)
