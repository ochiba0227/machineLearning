
# coding: utf-8

# In[1]:

print('Importing libraries...')
import random
import pandas as pd

original_file = '../kaggle/train.csv'

print('Counting original file rows...')
f = open(original_file,'r')
n_lines = sum(1 for l in f)
f.close()
print('Total number of rows: ', str(n_lines))

print('Creating dataframe...')
data = pd.read_csv(original_file)


# In[2]:

target_file = '../kaggle/frequent_producto.csv'
output_data = pd.DataFrame(columns=[])
#data = data[:]["Producto_ID"].value_counts()
output_data['ids'] = data[:]["Producto_ID"].value_counts().index.tolist()
output_data['producto_freq'] = data[:]["Producto_ID"].value_counts().as_matrix()
print('Saving sampled CSV...')
output_data.to_csv(target_file, index=False)
print('CSV saved > ', target_file)


# In[3]:

print(output_data)


# In[4]:

target_file = '../kaggle/frequent_agencia.csv'
output_data = pd.DataFrame(columns=[])
output_data['ids'] = data[:]["Agencia_ID"].value_counts().index.tolist()
output_data['agencia_freq'] = data[:]["Agencia_ID"].value_counts().as_matrix()
print('Saving sampled CSV...')
output_data.to_csv(target_file, index=False)
print('CSV saved > ', target_file)


# In[5]:

target_file = '../kaggle/frequent_canal.csv'
output_data = pd.DataFrame(columns=[])
output_data['ids'] = data[:]["Canal_ID"].value_counts().index.tolist()
output_data['canal_freq'] = data[:]["Canal_ID"].value_counts().as_matrix()
print('Saving sampled CSV...')
output_data.to_csv(target_file, index=False)
print('CSV saved > ', target_file)


# In[7]:

target_file = '../kaggle/frequent_route.csv'
output_data = pd.DataFrame(columns=[])
output_data['ids'] = data[:]["Ruta_SAK"].value_counts().index.tolist()
output_data['route_freq'] = data[:]["Ruta_SAK"].value_counts().as_matrix()
print('Saving sampled CSV...')
output_data.to_csv(target_file, index=False)
print('CSV saved > ', target_file)


# In[8]:

target_file = '../kaggle/frequent_client.csv'
output_data = pd.DataFrame(columns=[])
output_data['ids'] = data[:]["Cliente_ID"].value_counts().index.tolist()
output_data['client_freq'] = data[:]["Cliente_ID"].value_counts().as_matrix()
print('Saving sampled CSV...')
output_data.to_csv(target_file, index=False)
print('CSV saved > ', target_file)


# In[ ]:



