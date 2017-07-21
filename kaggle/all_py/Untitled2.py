
# coding: utf-8

# In[ ]:

import csv

with open('../kaggle/train.csv', 'r') as f:
    with open('../kaggle/train_short.csv', 'w') as fw:
        reader = csv.reader(f)
        writer = csv.writer(fw)

        for i, row in enumerate(reader):
            if i % 5 == 0:
                writer.writerow(row)

