import pandas as pd
import numpy as np
import time,os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking,Bidirectional,Dropout,Dense,Activation
from tensorflow.compat.v1.keras.layers import CuDNNLSTM,CuDNNGRU
from tensorflow.keras.preprocessing.sequence import pad_sequences
	
def splitASTENS(astens):
	num=len(astens)
	i=0
	while i<num:
		vector=np.array(astens[i]).astype(int)
		structCoding=np.trunc(vector/1000)
		structCoding=structCoding.astype(int)
#		idx=np.where(structCoding<0)
#		structCoding[idx]=-1
#		idx=np.where(structCoding>0)
#		structCoding[idx]=1
		typeCoding=abs(vector-structCoding*1000)
		typeCoding=typeCoding.astype(int)
		astens[i]=(list(zip(structCoding,typeCoding)))
		i=i+1

maxSeqLen=1000
classNum=104

hiddenUnits=400
dropout=0.2
batchSize=16
MaxEpoch=50
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

path='./'
train_samples=pd.read_pickle(path+'Samples-train.pkl')
splitASTENS(train_samples['astens'])
dev_samples=pd.read_pickle(path+'Samples-dev.pkl')
splitASTENS(dev_samples['astens'])
test_samples=pd.read_pickle(path+'Samples-test.pkl')
splitASTENS(test_samples['astens'])

train_x=train_samples['astens']
train_x=pad_sequences(train_x,maxlen=maxSeqLen,dtype='int32',padding='post',truncating='post',value=0.0)
train_y=train_samples['label']
train_y=np.array(train_y).astype(int)
train_y.reshape(len(train_y),1)
train_y=keras.utils.to_categorical(train_y)

dev_x=dev_samples['astens']
dev_x=pad_sequences(dev_x,maxlen=maxSeqLen,dtype='int32',padding='post',truncating='post',value=0.0)
dev_y=dev_samples['label']
dev_y=np.array(dev_y).astype(int)
dev_y.reshape(len(dev_y),1)
dev_y=keras.utils.to_categorical(dev_y)

test_x=test_samples['astens']
test_x=pad_sequences(test_x,maxlen=maxSeqLen,dtype='int32',padding='post',truncating='post',value=0.0)
test_y=test_samples['label']
test_y=np.array(test_y).astype(int)
test_y.reshape(len(test_y),1)
test_y=keras.utils.to_categorical(test_y)

model=Sequential([
	Bidirectional(CuDNNLSTM(hiddenUnits, return_sequences=False), input_shape=(maxSeqLen,2)),
	Dropout(dropout),
	Dense(classNum+1),
	Activation('softmax'),
])

lr_schedule=keras.optimizers.schedules.ExponentialDecay(
	initial_learning_rate=0.01,
	decay_steps=2,
	decay_rate=0.1)
optimizer=keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x=train_x,y=train_y,validation_data=(dev_x, dev_y),epochs=MaxEpoch,shuffle=True,batch_size=batchSize)
model.evaluate(test_x,test_y)
currentTime=time.strftime("%m-%d_%H_%M",time.localtime())
print("Saving model to: ASTSDL-BiLSTM-"+currentTime+".h5")
model.save(path+"ASTSDL-BiLSTM-"+currentTime+".h5")
