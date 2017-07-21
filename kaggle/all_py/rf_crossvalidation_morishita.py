
# coding: utf-8

# In[2]:

import numpy as np
import xgboost as xgb
import pandas as pd
import math

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from ml_metrics import rmsle

print ('')
print ('Loading Data...')

def evalerror(preds, dtrain):

    labels = dtrain.get_label()
    assert len(preds) == len(labels)
    labels = labels.tolist()
    preds = preds.tolist()
    terms_to_sum = [(math.log(labels[i] + 1) - math.log(max(0,preds[i]) + 1)) ** 2.0 for i,pred in enumerate(labels)]
    return 'error', (sum(terms_to_sum) * (1.0/len(preds))) ** 0.5

train = pd.read_csv('../kaggle/train_downsample_50000.csv')
test = pd.read_csv('../kaggle/test.csv')

print ('')
print ('Training_Shape:', train.shape)

ids = test['id']
test = test.drop(['id'],axis = 1)

y = train['Demanda_uni_equil']
X = train[test.columns.values]


# In[13]:

from sklearn.cross_validation import KFold

n_folds = 5
num_rounds = 100

rmsle_scores = []

for train_index, test_index in KFold(n=len(X), n_folds=n_folds, shuffle=True, random_state=1729):
    # Xをnparrayへ変更
    x_train = X.as_matrix()[train_index]
    y_train = y[train_index]
    x_test = X.as_matrix()[test_index]
    y_test = y[test_index]
    
    clf = RandomForestRegressor(n_estimators=30 , max_features='log2', random_state=1729)
    clf = clf.fit(x_train, y_train)
    
    preds = clf.predict(x_test)
    rmsle_scores.append(rmsle(y_test, preds))


# In[14]:

print(sum(rmsle_scores)/n_folds)


# In[ ]:



