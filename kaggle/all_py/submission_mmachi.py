
# coding: utf-8

# In[1]:

import numpy as np
import xgboost as xgb
import pandas as pd
import math

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

#train = pd.read_csv('./kaggle/train_downsample_5000000.csv')
train = pd.read_csv('./kaggle/train_downsample.csv')
test = pd.read_csv('./kaggle/test.csv')

freq_dict = {}
freq_file = open('./kaggle/frequent.csv')

first = True
for line in freq_file:
    if first:
        first = False
        continue
    key, value = line.split(',')
    key = int(key)
    value = int(value)
    freq_dict[key] = value
    
#print(freq_dict)
    
train['id_f'] = train['Producto_ID'].map(freq_dict)
test['id_f'] = test['Producto_ID'].map(freq_dict)

print ('')
print ('Training_Shape:', train.shape)

ids = test['id']
test = test.drop(['id'],axis = 1)

y = train['Demanda_uni_equil']
X = train[test.columns.values]

params = {}
params['objective'] = "reg:linear"
params['eta'] = 0.1
params['max_depth'] = 10
params['subsample'] = 0.85
params['colsample_bytree'] = 0.7
params['silent'] = True

from sklearn.cross_validation import KFold

n_folds = 5
num_rounds = 500
rmsle_scores = []

#for train_index, test_index in KFold(n=len(X), n_folds=n_folds, shuffle=True, random_state=1729):
    # Xをnparrayへ変更
x_train = X.as_matrix()
y_train = y

test_preds = np.zeros(test.shape[0])
xg_train = xgb.DMatrix(x_train, label=y_train)


watchlist = [(xg_train, 'train')]

xgclassifier = xgb.train(params, xg_train, num_rounds, watchlist, feval = evalerror, early_stopping_rounds= 20, verbose_eval = 10)
#preds = xgclassifier.predict(xg_test, ntree_limit=xgclassifier.best_iteration)

#print ('RMSLE Score:', rmsle(y_test, preds))
rmsle_scores.append

#test_preds = np.zeros(test.shape[0])
#unlabeled_test = xgb.DMatrix(test)
#fold_preds = np.around(xgclassifier.predict(unlabeled_test, ntree_limit=xgclassifier.best_iteration), decimals = 1)
#test_preds += fold_preds

#submission
fxg_test = xgb.DMatrix(test)
fold_preds = np.around(xgclassifier.predict(fxg_test, ntree_limit=xgclassifier.best_iteration), decimals = 1)
test_preds += fold_preds

submission = pd.DataFrame({'id':ids, 'Demanda_uni_equil': test_preds})
submission.to_csv('submission.csv', index=False)


# In[2]:

train


# In[ ]:



