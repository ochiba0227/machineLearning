
# coding: utf-8

# In[4]:

print('Importing libraries...')
import random
import pandas as pd

original_file = '../kaggle/train.csv'
target_lines = 500000
target_file = '../kaggle/train_downsample_'+str(target_lines)+'_s.csv'

print('Counting original file rows...')
f = open(original_file,'r')
n_lines = sum(1 for l in f)
f.close()
print('Total number of rows: ', str(n_lines))


print('Creating skip indexes...')
skip = random.sample(range(1, n_lines), n_lines - target_lines)

print('Creating dataframe...')
data = pd.read_csv(original_file, header=0, skiprows=skip)
skip = None
data = data[data['Semana'] > 5]
print('Saving sampled CSV...')
data.to_csv(target_file, index=False)
print('CSV saved > ', target_file)


# In[ ]:



