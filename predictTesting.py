import pandas as pd
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plot

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
	num=len(prediction)
	predictedPosition=pd.DataFrame(columns=['label','labelPosition'])
	i=0
	while i<num:
		pred=prediction[i]
		sortedID=np.argsort((-pred))
		predictedPosition.loc[i]=[sortedID[0],((np.argwhere(sortedID == samplesLabel[i]))[0])[0]]
		i=i+1
	return predictedPosition

def predicting(model,samples,maxSeqLen):
	test_x=samples['astens']
	test_x=pad_sequences(test_x,maxlen=maxSeqLen,dtype='int32',padding='post',truncating='post',value=0.0)
	test_y=samples['label']
	test_y=np.array(test_y).astype(int)
	prediction=predictLabel(test_x,test_y,model)
	return prediction
	
def analysisPrediction(prediction):
	num=len(prediction)
	acc_1=(len((np.where(prediction['labelPosition']<1))[0]))/num
	acc_1=round(acc_1*100,2)
	acc_2=(len((np.where(prediction['labelPosition']<2))[0]))/num
	acc_2=round(acc_2*100,2)
	acc_3=(len((np.where(prediction['labelPosition']<3))[0]))/num
	acc_3=round(acc_3*100,2)
	acc_5=(len((np.where(prediction['labelPosition']<5))[0]))/num
	acc_5=round(acc_5*100,2)
	acc_10=(len((np.where(prediction['labelPosition']<10))[0]))/num
	acc_10=round(acc_10*100,2)
	acc_20=(len((np.where(prediction['labelPosition']<20))[0]))/num
	acc_20=round(acc_20*100,2)
	acc_50=(len((np.where(prediction['labelPosition']<50))[0]))/num
	acc_50=round(acc_50*100,2)	
	return acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50

def accuracyMeasure(model,test_samples,maxSeqLen):
	selected=test_samples[test_samples['filename'].str.contains("cutted-9")]
	prediction=predicting(model,selected,maxSeqLen)
	acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50=analysisPrediction(prediction)
	accuracy_9=np.array([acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50])
	selected=test_samples[test_samples['filename'].str.contains("cutted-8")]
	prediction=predicting(model,selected,maxSeqLen)
	acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50=analysisPrediction(prediction)
	accuracy_8=np.array([acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50])
	selected=test_samples[test_samples['filename'].str.contains("cutted-7")]
	prediction=predicting(model,selected,maxSeqLen)
	acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50=analysisPrediction(prediction)
	accuracy_7=np.array([acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50])
	selected=test_samples[test_samples['filename'].str.contains("cutted-6")]
	prediction=predicting(model,selected,maxSeqLen)
	acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50=analysisPrediction(prediction)
	accuracy_6=np.array([acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50])
	selected=test_samples[test_samples['filename'].str.contains("cutted-5")]
	prediction=predicting(model,selected,maxSeqLen)
	acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50=analysisPrediction(prediction)
	accuracy_5=np.array([acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50])
	selected=test_samples[test_samples['filename'].str.contains("cutted-4")]
	prediction=predicting(model,selected,maxSeqLen)
	acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50=analysisPrediction(prediction)
	accuracy_4=np.array([acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50])
	selected=test_samples[test_samples['filename'].str.contains("cutted-3")]
	prediction=predicting(model,selected,maxSeqLen)
	acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50=analysisPrediction(prediction)
	accuracy_3=np.array([acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50])
	selected=test_samples[test_samples['filename'].str.contains("cutted-2")]
	prediction=predicting(model,selected,maxSeqLen)
	acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50=analysisPrediction(prediction)
	accuracy_2=np.array([acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50])
	selected=test_samples[~test_samples['filename'].str.contains("cutted")]
	prediction=predicting(model,selected,maxSeqLen)
	acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50=analysisPrediction(prediction)
	accuracy_all=np.array([acc_1,acc_2,acc_3,acc_5,acc_10,acc_20,acc_50])
	accuracy=np.r_[accuracy_9,accuracy_8,accuracy_7,accuracy_6,accuracy_5,accuracy_4,accuracy_3,accuracy_2,accuracy_all]
	accuracy=accuracy.reshape((9,7))
	return accuracy

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
maxSeqLen=1000
modelName='BiLSTM'
hiddenUnits=400
roundNum=11
print("Accuracy for "+modelName+"-"+str(hiddenUnits)+":")

datapath='./data/'
dataset='OJ/'
test_samples=pd.read_pickle(datapath+dataset+'testset.pkl')
splitASTENS(test_samples['astens'])
accuracy_all=np.zeros((roundNum,9,7))
meanAccuracy=np.zeros((9,7))
for i in range(0,roundNum):
	model=load_model('models/'+dataset+modelName+'-'+str(hiddenUnits)+'/ASTSDL-'+modelName+'-'+str(i)+'.h5')
	accuracy=accuracyMeasure(model,test_samples,maxSeqLen)
	accuracy_all[i,:,:]=accuracy
	meanAccuracy=meanAccuracy+accuracy
meanAccuracy=meanAccuracy/roundNum
print(accuracy_all)
print(meanAccuracy)
x=np.arange(roundNum)
for i in range(0,8):
	plot.plot(x,accuracy_all[:,i,5])
plot.show()

