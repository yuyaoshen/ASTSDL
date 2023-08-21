import os
import pandas as pd
import numpy as np

path='./'
data=pd.read_pickle(path+'data/OJ/dataset.pkl')
ratio='6:2:2'
ratios=[int(r) for r in ratio.split(':')]
labels=data['label']
classes=np.unique(labels)
train_samples=pd.DataFrame(columns=['filename','label','astens'])
dev_samples=pd.DataFrame(columns=['filename','label','astens'])
test_samples=pd.DataFrame(columns=['filename','label','astens'])
for i in classes:
	samples=data[data['label'].isin([i])]
	data_num=len(samples)
	samples=samples.sample(frac=1, random_state=666)
	train_split=int(ratios[0]/sum(ratios)*data_num)
	val_split=train_split + int(ratios[1]/sum(ratios)*data_num)
	train=samples.iloc[:train_split]
	train_samples=train_samples.append(train)
	dev=samples.iloc[train_split:val_split]
	dev_samples=dev_samples.append(dev)
	test=samples.iloc[val_split:]
	test_samples=test_samples.append(test)
train_samples=train_samples.reset_index(drop=True)
train_samples.to_pickle(path+'Samples-train.pkl')
dev_samples=dev_samples.reset_index(drop=True)
dev_samples.to_pickle(path+'Samples-dev.pkl')
test_samples=test_samples.reset_index(drop=True)
test_samples.to_pickle(path+'Samples-test.pkl')
