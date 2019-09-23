import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
data_dir = './train_set/'
result_dir = './result/'
files = ['train_114501.csv', 
         'train_115601.csv', 
         'train_115701.csv',
         'train_108401.csv',
         'train_111801.csv',
         'train_111901.csv'
         ]
for f in files:
    data = pd.read_csv(os.path.join(data_dir, f))
    freq = data['Frequency Band'][0]
    true_label = data['RSRP'].values
    result_file = f+'_result.txt'
    with open(os.path.join(result_dir, result_file)) as rf:
        pred = rf.readline().strip()
        pred = eval(pred)
        pred = np.array(pred['RSRP']).reshape(-1)
    mse = mean_squared_error(true_label, pred)
    print('freq: {}'.format(freq))
    print('mse: {}'.format(mse))