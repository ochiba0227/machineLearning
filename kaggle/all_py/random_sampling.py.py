
# coding: utf-8

# In[ ]:

print('Importing libraries...')
import random
import pandas as pd

original_file = '../kaggle/train.csv'
target_file = '../kaggle/train_sample_5000000.csv'
target_lines = 5000000

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

print('Saving sampled CSV...')
data.to_csv(target_file, index=False)
print('CSV saved > ', target_file)


# In[ ]:



