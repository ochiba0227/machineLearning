
# coding: utf-8

# In[164]:

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

train = pd.read_csv('../kaggle/train_downsample_50000.csv')
test = pd.read_csv('../kaggle/test.csv')


# In[165]:

print ('')
print ('Training_Shape:', train.shape)

ids = test['id']
test = test.drop(['id'],axis = 1)

y = train['Demanda_uni_equil']
X_orig = train[test.columns.values]


# In[166]:

# パンの出現頻度を追加
frequent = pd.read_csv('../kaggle/frequent_producto.csv')
X_orig = pd.merge(left=X_orig,right=frequent, how='left', left_on='Producto_ID', right_on='ids')


# In[167]:

# Agencia_IDの出現頻度を追加
frequent = pd.read_csv('../kaggle/frequent_agencia.csv')
X_orig = pd.merge(left=X_orig,right=frequent, how='left', left_on='Agencia_ID', right_on='ids')


# In[168]:

# Canal_IDの出現頻度を追加
frequent = pd.read_csv('../kaggle/frequent_canal.csv')
X_orig = pd.merge(left=X_orig,right=frequent, how='left', left_on='Canal_ID', right_on='ids')


# In[169]:

# Ruta_SAKの出現頻度を追加
frequent = pd.read_csv('../kaggle/frequent_route.csv')
X_orig = pd.merge(left=X_orig,right=frequent, how='left', left_on='Ruta_SAK', right_on='ids')


# In[170]:

# Cliente_IDの出現頻度を追加
frequent = pd.read_csv('../kaggle/frequent_client.csv')
X_orig = pd.merge(left=X_orig,right=frequent, how='left', left_on='Cliente_ID', right_on='ids')


# In[171]:

# 前週の売上平均値を与える
mean_equil = pd.DataFrame(columns=[])
mean_equil['Semana'] = list(map(lambda x: x+1,train.loc[:,["Semana","Demanda_uni_equil"]].groupby('Semana')['Demanda_uni_equil'].mean().index.tolist()))
mean_equil['uni_equil'] = train.loc[:,["Semana","Demanda_uni_equil"]].groupby('Semana')['Demanda_uni_equil'].mean().as_matrix()


# In[172]:

# プロダクトの売上平均値を与える
mean_equil_prd = pd.DataFrame(columns=[])
mean_equil_prd['Producto_ID'] = list(map(lambda x: x+1,train.loc[:,["Producto_ID","Demanda_uni_equil"]].groupby('Producto_ID')['Demanda_uni_equil'].mean().index.tolist()))
mean_equil_prd['uni_equil_prd'] = train.loc[:,["Producto_ID","Demanda_uni_equil"]].groupby('Producto_ID')['Demanda_uni_equil'].mean().as_matrix()


# In[173]:

X_orig = pd.merge(left=X_orig,right=mean_equil, how='left', left_on='Semana', right_on='Semana')
X_orig = pd.merge(left=X_orig,right=mean_equil_prd, how='left', left_on='Producto_ID', right_on='Producto_ID')

X_orig = X_orig[X_orig["Semana"]>3]

X_orig[0:1]


# In[190]:

#X = X_orig.loc[:,['Semana','producto_freq','agencia_freq','canal_freq','route_freq','client_freq']]
X = X_orig.drop(['ids','ids_x','ids_y'],axis=1)
#X = X.drop(['agencia_freq','canal_freq','route_freq'],axis=1)
X = X.loc[:,["Semana","Cliente_ID","Producto_ID","uni_equil","uni_equil_prd"]].reset_index()
X[0:1]


# In[ ]:




# In[191]:

params = {}
params['objective'] = "reg:linear"
params['eta'] = 0.1
params['max_depth'] = 10
params['subsample'] = 0.85
params['colsample_bytree'] = 0.7
params['silent'] = True


# In[192]:

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
    
    test_preds = np.zeros(test.shape[0])
    xg_train = xgb.DMatrix(x_train, label=y_train)
    xg_test = xgb.DMatrix(x_test)
    
    watchlist = [(xg_train, 'train')]
    
    xgclassifier = xgb.train(params, xg_train, num_rounds, watchlist, feval = evalerror, early_stopping_rounds= 20, verbose_eval = 10)
    preds = xgclassifier.predict(xg_test, ntree_limit=xgclassifier.best_iteration)

    rmsle_scores.append(rmsle(y_test, preds))


# In[177]:

print(sum(rmsle_scores)/n_folds)


# In[178]:

rmsle_scores


# In[ ]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#plt.hist(train.loc[:,['Semana','Venta_hoy','Dev_proxima']].groupby('Semana').sum())
train.loc[:,['Semana','Venta_hoy','Dev_proxima']].groupby('Semana').sum()


# In[ ]:



