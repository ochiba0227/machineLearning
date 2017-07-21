# coding: utf-8

import pandas as pd
import tensorflow as tf
import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 末尾の行に目的とするデータがある
X = train_data.iloc[:,:-5]
Y = train_data.iloc[:,-1:]

clf = RandomForestRegressor(n_estimators=10)

# １次元配列の形で渡す
clf = clf.fit(X, Y.values.ravel())
# 学習結果を保存
joblib.dump(clf, './model/rf_'+dt.datetime.today().strftime("%Y%m%d%H%M%S")+'.pkl')

# テストデータはidを持っているので省く
T = test_data.iloc[:,1:]

print(clf.predict(T))
