import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def splitASTENS(astens):
	num=len(astens)
	i=0
	while i<num:
		vector=np.array(astens[i]).astype(int)
		structCoding=np.trunc(vector/1000)
		structCoding=structCoding.astype(int)
		typeCoding=abs(vector-structCoding*1000)
		typeCoding=typeCoding.astype(int)
		astens[i]=(list(zip(structCoding,typeCoding)))
		i=i+1

maxSeqLen=1000
path='./'
test_samples=pd.read_pickle(path+'testset.pkl')
splitASTENS(test_samples['astens'])
data=test_samples['astens']
data=pad_sequences(data,maxlen=maxSeqLen,dtype='int32',padding='post',truncating='post',value=0.0)
data=data[1]
print(data)
np.save('data-npy',data)
