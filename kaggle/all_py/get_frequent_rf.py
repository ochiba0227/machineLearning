
# coding: utf-8

# In[ ]:

print('Importing libraries...')
import random
import pandas as pd

original_file = '../kaggle/train.csv'
target_file = '../kaggle/frequent.csv'

print('Counting original file rows...')
f = open(original_file,'r')
n_lines = sum(1 for l in f)
f.close()
print('Total number of rows: ', str(n_lines))

print('Creating dataframe...')
data = pd.read_csv(original_file)

data['idf'] = data[:]["Producto_ID"].value_counts()

print('Saving sampled CSV...')
data.to_csv(target_file, index=False)
print('CSV saved > ', target_file)


# In[ ]:



