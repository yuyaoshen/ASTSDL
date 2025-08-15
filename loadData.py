import os
import pandas as pd

path='./'
file_dir=path+'dataset/'
output='dataset'
files=os.listdir(file_dir)
data=pd.DataFrame(columns=['filename','label','astens'])
count=0
for filename in files:
	label=filename.split('@')[0]
	label=int(label) #BCB数据集需 -1
	astens=[]
	with open(file_dir+filename,mode='r',encoding='utf-8') as f:
		content=f.readline()
		while content:
			lineContent=content.split()
			astens.append(lineContent[0])
			content=f.readline()
	data.loc[count]=[filename,label,astens]
	count=count+1
data.to_pickle(path+output+'.pkl')
source=pd.read_pickle(path+output+'.pkl')
print(source)
