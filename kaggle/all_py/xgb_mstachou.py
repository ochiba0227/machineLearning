
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import xgboost as xgb
from ml_metrics import rmsle as metric
from sklearn import preprocessing as ppr
from sklearn.cross_validation import train_test_split

def rmsle(true, labels):
    pred = labels.get_label()
    if len(pred)==len(true):
        pred[pred<0] = 0
        rmsle = np.sqrt((sum((np.log(pred+1) - np.log(true+1))**2))/len(true))
        return 'rmsle', rmsle
        
print('start!')
#train = pd.read_csv('./kaggle/train.csv', nrows = 500000)
train = pd.read_csv('./kaggle/train_downsample_50000.csv')
test = pd.read_csv('./kaggle/test.csv')
freq = pd.read_csv('./kaggle/frequent.csv',  dtype={'freq': np.int32})

test_id = test['id']
test = test.drop(['id'],axis = 1)
y = train['Demanda_uni_equil']
X = train[test.columns.values]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4518)

params = {'objective': "reg:linear",
          'eta'      : 0.03,
          'max_depth': 10}
params['subsample'] = 0.85
params['colsample_bytree'] = 0.7
rounds = 100

xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_test = xgb.DMatrix(X_test)

watchlist = [(xgb_train, 'train')]

xgb_reg = xgb.train(params, xgb_train, rounds, watchlist, feval = rmsle, early_stopping_rounds= 20, verbose_eval = 10)
preds = xgb_reg.predict(xgb_test, ntree_limit=xgb_reg.best_iteration)

print('rmsle:', metric(y_test, preds))

test_preds = np.zeros(test.shape[0])
unlabeled_test = xgb.DMatrix(test)
fold_preds = np.around(xgb_reg.predict(unlabeled_test, ntree_limit=xgb_reg.best_iteration), decimals = 1)
test_preds += fold_preds


# In[4]:

submission = pd.DataFrame({'id':test_id, 'Demanda_uni_equil': test_preds})
submission.to_csv('submission.csv', index=False)


# In[5]:

fold_preds


# In[6]:

unlabeled_test.feature_names


# In[7]:

xgb_train.feature_names


# In[ ]:



