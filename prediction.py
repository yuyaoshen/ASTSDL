import pandas as pd
import numpy as np
import time,os
from tensorflow import keras
from tensorflow.keras.models import load_model
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

def predictLabel(samples,samplesLabel,model):
	prediction=model.predict(samples)
	prediction=pd.DataFrame(prediction)
	return prediction

def predicting(model,samples,maxSeqLen):
	test_x=samples['astens']
	test_x=pad_sequences(test_x,maxlen=maxSeqLen,dtype='int32',padding='post',truncating='post',value=0.0)
	test_y=samples['label']
	test_y=np.array(test_y).astype(int)
	prediction=predictLabel(test_x,test_y,model)
	results=pd.concat([samples['filename'],prediction],axis=1)
	return results
	
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
model_path='models/BCB/'
results_path='results/'
maxSeqLen=1000
modelName='BiLSTM'
hiddenUnits=400
roundNum=5

path='./data/BCB/'
test_samples=pd.read_pickle(path+'BCB_Full.pkl')
splitASTENS(test_samples['astens'])

for i in range(0,roundNum):
	model=load_model(model_path+modelName+'-'+str(hiddenUnits)+'/ASTSDL-'+modelName+'-'+str(i)+'.h5')
	currentTime=time.strftime("%m-%d_%H_%M_%S",time.localtime())
	print(currentTime)
	results=predicting(model,test_samples,maxSeqLen)
	currentTime=time.strftime("%m-%d_%H_%M_%S",time.localtime())
	print(currentTime)
	print(results)
	results.to_pickle(results_path+'ASTSDL-'+modelName+'-'+str(i)+'.pkl')
	results.to_csv(results_path+'ASTSDL-'+modelName+'-'+str(i)+'.txt',sep='\t', index=False)

