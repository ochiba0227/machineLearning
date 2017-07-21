
# coding: utf-8

# In[1]:



import pandas as pd
import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib



# In[2]:

train_data = pd.read_csv("../kaggle/train.csv")


# In[3]:

import xgboost


# In[4]:

train_data[0:20]


# In[1]:

import io
import base64
from IPython.display import HTML

#video = io.open('/home/ubuntu/Downloads/test.mp4', 'r+b').read()
encoded = base64.b64encode(video)

HTML(data='''<video alt="test" controls>
     <source src="data:video/mp4;base64,{0}" type="video/mp4" />
     </video>'''.format(encoded.decode('ascii')))


# In[ ]:



