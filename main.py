import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import datetime as dt
from sklearn import linear_model 
from sklearn.metrics import mean_absolute_error,accuracy_score,confusion_matrix
import os
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional
from sklearn.preprocessing import MinMaxScaler

lenWin=1

def readFiles():
	data = []
	os.chdir('./Data/Stocks/')
	stockFileNames=os.listdir('./')
	stockFileNames = ['prk.us.txt', 'bgr.us.txt', 'jci.us.txt', 'aa.us.txt',
					 'fr.us.txt', 'star.us.txt', 'sons.us.txt', 'ipl_d.us.txt',
					  'sna.us.txt', 'utg.us.txt']
	c=0
	for f in stockFileNames:
		fileName=f
		fp=open(fileName,'r')
		l=len(fp.read())
		if(l>50000):
			df=pd.read_csv(fileName,encoding="utf-8-sig")
			label, _, _ = fileName.split('.')
			df['Label'] = fileName
			df['Date'] = pd.to_datetime(df['Date'])
			data.append(df)        
		fp.close()
		c+=1
	print('File reading Done')
	return data

def plotData(data):
    plt.plot(data['Date'], data['Close'])
    plt.show()


def apply_smoothing(trainData,testData):
	scaler = MinMaxScaler()
	trainData = trainData.reshape(-1,1)
	testData = testData.reshape(-1,1)
	smoothing_window_size = 2500
	for di in range(0,10000,smoothing_window_size):
	    scaler.fit(trainData[di:di+smoothing_window_size,:])
	    trainData[di:di+smoothing_window_size,:] = scaler.transform(trainData[di:di+smoothing_window_size,:])
	scaler.fit(trainData[di+smoothing_window_size:,:])
	trainData[di+smoothing_window_size:,:] = scaler.transform(trainData[di+smoothing_window_size:,:])
	trainData = trainData.reshape(-1)
	testData = scaler.transform(testData).reshape(-1)
	EMA = 0.0
	gamma = 0.1
	for ti in range(11000):
	  EMA = gamma*trainData[ti] + (1-gamma)*EMA
	  trainData[ti] = EMA
	return trainData,testData

def process_inputs(tSet,lenWin):
	inputs = []
	for i in range(len(tSet)-lenWin):
	    temp_set = tSet[i:(i+lenWin)].copy()
	    
	    for col in list(temp_set):
	        temp_set[col] = temp_set[col]/temp_set[col].iloc[0] - 1
	    
	    inputs.append(temp_set)
	outputs = (tSet['Close'][lenWin:].values/tSet['Close'][:-lenWin].values)-1

	inputs = [np.array(tInput) for tInput in inputs]
	inputs = np.array(inputs)

	return inputs,outputs

def trainLSTMModel(inputs,outputs):
    print('Training LSTM Model')
    model = Sequential()
    model.add(LSTM(32, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    model.compile(loss='mae', optimizer='adam')
    model.fit(inputs,outputs,epochs=5, batch_size=1, verbose=0, shuffle=True)
    print('Model Trained')
    return model

def trainBiLSTMModel(inputs,outputs):
    print('Training BiLSTM Model')
    model = Sequential()
    model.add(Bidirectional(LSTM(32, input_shape=(inputs.shape[1], inputs.shape[2]))))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation('linear'))
    model.compile(loss='mae', optimizer='adam')
    model.fit(inputs,outputs,epochs=5, batch_size=1, verbose=0, shuffle=True)
    print('Model Trained')
    return model

def pred(model, data, windowSize):
    win=data[0]
    predictions=[]
    for i in range(len(data)):
        predictions.append(model.predict(win[np.newaxis,:,:])[0,0])
        win=win[1:]
        win=np.insert(win, [windowSize-1], predictions[-1], axis=0)
    return predictions

data=readFiles()

df=data[0]
# plotData(df)

splitInterval=list(data[0]["Date"][-(1000+1):])[0]


trainSet=df[df['Date']<splitInterval]
trainSet=trainSet.drop(['Date','Label', 'OpenInt'], 1)

testSet=df[df['Date']>=splitInterval]
testSet=testSet.drop(['Date','Label','OpenInt'], 1)


print(trainSet.shape)
print(testSet.shape)
# apply_smoothing(trainSet,testSet)

trainInputs,trainOutputs=process_inputs(trainSet,lenWin)
test_inputs,test_outputs=process_inputs(testSet,lenWin)

# model=trainLSTMModel(trainInputs,trainOutputs)

model=trainBiLSTMModel(trainInputs,trainOutputs)

predictions = pred(model, test_inputs, lenWin)

# plt.plot(test_outputs, label="actual")
plt.plot(predictions, label="predicted")
plt.legend()
plt.show()
inc_pred=[]
inc_act=[]
for i in range(1,len(predictions)):
	if(predictions[i]-predictions[i-1]>=0):
		inc_pred.append(1)
	else:
		inc_pred.append(0)
	if(test_outputs[i]-test_outputs[i-1]>=0):
		inc_act.append(1)
	else:
		inc_act.append(0)


mae=mean_absolute_error(test_outputs,predictions)
print('The Mean Absolute Error is:',mae)

acc=accuracy_score(inc_act,inc_pred)
print('Accuracy:',acc)

print(confusion_matrix(inc_act,inc_pred))