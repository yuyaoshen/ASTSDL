import os
import pandas as pd
import numpy as np

path='./'
data=pd.read_pickle(path+'dataset.pkl')
labels=data['label']
classes=np.unique(labels)
for i in classes:
	samples=data[data['label'].isin([i])]
	data_num=len(samples)
	print(data_num)
