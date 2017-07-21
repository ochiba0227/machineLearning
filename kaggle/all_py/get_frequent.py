
# coding: utf-8

# In[24]:

print('Importing libraries...')
import random
import pandas as pd

original_file = '../kaggle/train.csv'
target_file = '../kaggle/frequent_header.csv'

print('Counting original file rows...')
f = open(original_file,'r')
n_lines = sum(1 for l in f)
f.close()
print('Total number of rows: ', str(n_lines))

print('Creating dataframe...')
data = pd.read_csv(original_file)


# In[25]:

output_data = pd.DataFrame(columns=['ids','freq'])


# In[31]:

#data = data[:]["Producto_ID"].value_counts()
output_data['ids'] = data[:]["Producto_ID"].value_counts().index.tolist()
output_data['freq'] = data[:]["Producto_ID"].value_counts().as_matrix()


# In[32]:

print('Saving sampled CSV...')
output_data.to_csv(target_file, index=False)
print('CSV saved > ', target_file)


# In[33]:

print(output_data)


# In[ ]:



