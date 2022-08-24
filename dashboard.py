# -*- coding: utf-8 -*-
"""Dashboard.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YjmtJR_nLxi2w725ahniH9kUSPZ76cBY
"""

!pip install streamlit
!pip install pyngrok

from keras.models import load_model
import streamlit as st

from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose

!pip install yfinance

import yfinance as yf

st.title("Stock Trend Prediction")

user_input=st.text_input("Enter Strock Ticker","GOOG")

start="2018-01-01"
end="2022-01-01"
df=yf.download('GOOG',start,end)

st.subheader('Data from 2018-2020')
st.write(df.describe())

df_close = df[['Close']]


result = seasonal_decompose(df_close, model='multiplicative', period=365)
fig = result.plot()
fig.set_size_inches(20, 8)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.plot(df_close)

ma100=df_close.rolling(100).mean()
ma200=df_close.rolling(200).mean()

plt.figure(figsize=(12,6))
plt.plot(df_close)
plt.plot(ma100,'red',label="100 day Moving Average")
plt.plot(ma200,'green',label="200 day Moving Average")
plt.legend()

"""IF 100 MA>200 MA: UPTREND <font color='green'> buy stock.

IF 100 MA<200 MA : DOWNTREND <font color='red'> sell stock
"""

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df_close).reshape(-1,1))
df1.shape

# Splitting data into Training and Testing
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing= pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

data_training_array=scaler.fit_transform(data_training)

x_train=[]
y_train=[]

for i in range (100,data_training_array.shape[0]):
  x_train.append(data_training_array[i-100:i])
  y_train.append(data_training_array[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)

# ML Model
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model=Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60,activation='relu',return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1)) # Since our prediction is only one number hence we are adding onlu one unit in dense layer

model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(x_train,y_train, epochs=20, batch_size=16, verbose=2)
#verbose shows the number of epochs it has completerd

model.save('LSTM.h5')

past_100_days=data_training.tail(100)
final_df= past_100_days.append(data_testing,ignore_index=True)

input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

# Making Predictions

y_predicted=model.predict(x_test)

scaler.scale_

scale_factor=1/0.01208985

y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# Model evaluation
mape = np.mean(np.abs(y_test-y_predicted))
mape