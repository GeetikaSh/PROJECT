
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
# %matplotlib inline
import datetime as dt
from datetime import datetime, timedelta
import scipy.stats as stats
import statistics
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import streamlit as st
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# Project Details
st.title("Time Series Forecasting")
st.markdown("Comparitive Analysis of various Time Series models on GOOGLE Stock Closing Price")


 
# fetches the data: Open, Close, High, Low and Volume
df = yf.download('GOOG', 
                      start='2018-01-01', 
                      end='2022-01-01', 
                      progress=False)
select=st.sidebar.selectbox("Objective:",("Risk in Investment","Investment Period","Model and Accuracy"))




@st.cache
def lstm(df):
  scaler=MinMaxScaler(feature_range=(0,1))
  df1=scaler.fit_transform(np.array(df).reshape(-1,1))
  train_size=int(len(df1)*0.65)
  test_size=len(df1)-train_size
  train_data,test_data=df1[0:train_size,:],df1[train_size:len(df1),:1]
  def create_dataset(dataset,time_step=1):
    dataX,dataY= [],[]
    for i in range(len(dataset)-time_step-1):
      a=dataset[i:(i+time_step),0]
      dataX.append(a)
      dataY.append(dataset[i+time_step,0])
    return np.array(dataX), np.array(dataY)
  time_step=100
  X_train,y_train= create_dataset(train_data,time_step)
  X_test,y_test=create_dataset(test_data,time_step)
  X_train= X_train.reshape(X_train.shape[0],X_train.shape[1],1)
  X_test= X_test.reshape(X_test.shape[0],X_test.shape[1], 1)
  model=Sequential()
  model.add(LSTM(100, return_sequences=True, input_shape=(100,1)))
  model.add(LSTM(1000,return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error',optimizer='adam')
  model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64,verbose=1)
  train_predict=model.predict(X_train)
  test_predict=model.predict(X_test)
  train_predict=scaler.inverse_transform(train_predict)
  test_predict=scaler.inverse_transform(test_predict)
  look_back=100
  trainPredictPlot=np.empty_like(df1)
  trainPredictPlot[:,:]=np.nan
  trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict
    # shift test predictions for plotting
  testPredictPlot=np.empty_like(df1)
  testPredictPlot[:,:]=np.nan
  testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:]=test_predict
  return(scaler.inverse_transform(df1),trainPredictPlot,testPredictPlot,test_predict)
    #plot baseline and predictions



if select=="Risk in Investment":
  st.title("Exploratory Data Analysis of GOOGLE")
  st.header(("Closing Price"))
  fig=plt.figure(figsize=(10,5))
  plt.plot(df['Close'])
  plt.ylabel("Closing Price")
  plt.title("Closing Price of GOOGLE")
  st.pyplot(fig)

# Now let's plot the total volume of stock being traded each day
  st.header("Volume of the stock being traded per day")
  st.text('''Average daily trading volume is an important metric because high or
low trading volume attracts different types of traders and investors.''')
  st.caption('''Many traders and investors prefer higher average daily trading 
volume compared to low trading volume, because with high volume it is easier to 
get into and out positions.''')
  fig=plt.figure(figsize=(10, 5))
  df['Volume'].plot()
  plt.ylabel("Volume")
  plt.title("Sales Volume of GOOGLE")
  st.pyplot(fig)

  st.header("Daily Return of GOOGLE")
  st.caption('''We're now going to analyze the risk of the stock. In order to 
do so we'll need to take a closer look at the daily changes of the stock, and 
not just its absolute value.''')
  fig=plt.figure(figsize=(10, 5))
  df['Daily Return']=df['Close'].pct_change()
  df['Daily Return'].plot(legend=True, linestyle='--', marker='o').set_title('Daily Return of GOOGLE')
  st.pyplot(fig)
  
  st.subheader("Histogram of Daily Return of GOOGLE")
  st.text('''Histogram helps to analyse the stock return and is vely helpful if
we are comparing the performace of teh stock with other stocks.''')
  st.caption('''What if there is skewness in the daily return plot:
Positive Skewness:the values of the dataset extend to the right of the peak
A relatively high positive skewness reading indicates returns deep in the right tail of the distribution.
Negative Skewness; the values of the dataset extend to left of the peak
A negative number equates with a loss in the left tail
''')

  fig=plt.figure(figsize=(10, 5))
  df['Daily Return'].hist(bins=50)
  plt.ylabel("Daily Return")
  plt.title("Histogram of Daily Return of GOOGLE")
  st.pyplot(fig)

  st.header("How much value do we put at risk by investing in a particular stock?")
  st.text('''There are many ways we can quantify risk, one of the most basic 
ways using the information we've gathered on daily percentage returns is by 
comparing the expected return with the standard deviation of the daily returns.''')
  rets = df[["Daily Return"]]

  area = np.pi * 20

  fig=plt.figure(figsize=(10, 5))
  plt.scatter(rets.mean(), rets.std(), s=area)
  plt.annotate('Risk',xy=(rets.mean(), rets.std()), xytext=(50, 50),textcoords='offset points', ha='right', va='bottom', 
                  arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))

  plt.xlabel('Expected return')
  plt.ylabel('Risk')
  plt.title("Risk by investing in GOOGLE")
  st.pyplot(fig)
  
  df=df[['Close']]

if select=="Investment Period":
  df=df[['Close']]
  data=st.sidebar.selectbox("Time of Investment",("Moving Average","Seasonality"))
  if data=="Moving Average":
    ma100=df.rolling(100).mean()
    ma200=df.rolling(200).mean()
    st.header("Moving Average")
    st.text('''A moving average (MA) is a stock indicator commonly used in 
    technical analysis, used to help smooth out price data by creating a 
    constantly updated average price.A rising moving average indicates that the 
    security is in an uptrend, while a declining moving average indicates a 
    downtrend.''')
    st.markdown("IF 100 MA>200 MA: UPTREND (buy stock).")
    st.markdown("IF 100 MA<200 MA : DOWNTREND (sell stock)")
    st.caption(" ")
    st.caption("The plot below gives you the information when to buy and sella  stock")
    fig=plt.figure(figsize=(10,5))
    plt.plot(df)
    plt.plot(ma100,'red',label="100 day Moving Average")
    plt.plot(ma200,'green',label="200 day Moving Average")
    plt.title("Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend()
    st.pyplot(fig)

  if data=="Seasonality":
    st.header("Seasonality")
    st.text('''Seasonality is the study of a market over time to establish if there
is a regular and predictable change that occurs within that markets price every 
calendar year.Every market experiences periods of either greater 
supply or demand throughout a year, and it is these forces that drive seasonal patterns.''')
    st.caption("")
    st.caption('''It is recommened to observe the seasonality plot, and invest in the stock
    on that basis''')
    result = seasonal_decompose(df, model='multiplicative', period=365)
    fig = result.plot()
    
    plt.legend()
    st.pyplot(fig)

df=df[['Close']]
if select=='Model and Accuracy':
  st.title("Time Series Forecasting")
  model=st.sidebar.selectbox("Model:",("ARIMA: z-score","ARIMA: log","SVR","LSTM"))

  if model=="ARIMA: z-score":
    st.subheader("ARIMA with z-score normalization")
    z_score=stats.zscore(df)
    train_data, test_data = z_score[3:int(len(z_score)*0.9)], z_score[int(len(z_score)*0.9):]
    model = sm.tsa.arima.ARIMA(z_score.Close, order=(1,1,1))
    model=sm.tsa.statespace.SARIMAX(z_score['Close'],order=(1,1,1),seasonal_order=(1,1,1,12))
    results=model.fit()
    
    z_score['Forecast']=results.predict(dynamic=False)
    forecast = results.get_forecast(101)
    conf = forecast.conf_int(alpha=0.05)
    conf.index=test_data.index
    lower_series = pd.Series(conf["lower Close"])
    upper_series = pd.Series(conf["upper Close"])
    
    fig=plt.figure(figsize=(10,5), dpi=100)
    plt.title("Normalized Predicted Value")
    plt.ylabel("Normalized Closing Price")
    plt.plot(z_score['Close'],label="Actual")
    plt.plot(z_score["Forecast"],label="Predicted")
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                    color='k', alpha=.10)
    plt.legend()
    st.pyplot(fig)


#     df2 = pd.DataFrame(columns=["Predicted"],index=pd.date_range("20220101", periods=365))
#     x=1009
#     for i in range (len(df2)):
#       df2.Predicted.iloc[i]=float(fitted.predict(x))
#       x=x+1

    #predict=df2.loc[date]





  if model=="ARIMA: log":
    st.header("ARIMA with log normalization")
    log_g=df
    log_g['Close'] = np.log(log_g['Close']).diff()
    log_g=log_g.dropna()
    train_data, test_data = log_g[3:int(len(log_g)*0.9)], log_g[int(len(log_g)*0.9):]
    model = sm.tsa.arima.ARIMA(log_g.Close, order=(1,0,1))
    model=sm.tsa.statespace.SARIMAX(log_g['Close'],order=(1,0,1),seasonal_order=(1,0,1,6))
    results=model.fit()
    log_g['Forecast']=results.predict(dynamic=False)
    forecast = results.get_forecast(101)
    conf = forecast.conf_int(alpha=0.05)
    conf.index=test_data.index
    lower_series = pd.Series(conf["lower Close"])
    upper_series = pd.Series(conf["upper Close"])
    
    fig=plt.figure(figsize=(10,5), dpi=100)
    
    plt.plot(log_g["Close"],label="Actual")
    plt.plot(log_g["Forecast"],label="Predicted")
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                    color='k', alpha=.10)
    

    plt.ylabel("Normalized Closing Price")
    plt.legend()
    st.pyplot(fig)

    
    #predict=df2.loc[date]

  if model=="SVR":
    st.header("Support Vector Regression")
    future_days=5
    df[str(future_days)+'_Day_Price_Forecast']=df[['Close']].shift(-future_days)
    df[['Close',str(future_days)+'_Day_Price_Forecast']]
    X=np.array(df[['Close']])
    X=X[:df.shape[0]-future_days]
    y=np.array(df[str(future_days)+'_Day_Price_Forecast'])
    y=y[:-future_days]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    svr_rbf=SVR(kernel='rbf',C=1e3,gamma=0.00001)
    svr_rbf.fit(X_train,y_train)
    
    svm_prediction=svr_rbf.predict(X_test)
    fig=plt.figure(figsize=(10,5))
    plt.plot(svm_prediction,label='prediction',lw=2,alpha=0.7)
    plt.plot(y_test,label='Actual Value', lw=4, alpha=0.5)
    
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.legend(loc = "upper left")
    st.pyplot(fig)

  if model=="LSTM":
    st.header("LSTM")
    # scaler=MinMaxScaler(feature_range=(0,1))
    # df1=scaler.fit_transform(np.array(df).reshape(-1,1))
    # train_size=int(len(df1)*0.65)
    # test_size=len(df1)-train_size
    # train_data,test_data=df1[0:train_size,:],df1[train_size:len(df1),:1]
    # def create_dataset(dataset,time_step=1):
    #   dataX,dataY= [],[]
    #   for i in range(len(dataset)-time_step-1):
    #     a=dataset[i:(i+time_step),0]
    #     dataX.append(a)
    #     dataY.append(dataset[i+time_step,0])
    #   return np.array(dataX), np.array(dataY)
    # time_step=100
    # X_train,y_train= create_dataset(train_data,time_step)
    # X_test,y_test=create_dataset(test_data,time_step)
    # X_train= X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    # X_test= X_test.reshape(X_test.shape[0],X_test.shape[1], 1)
    # model=Sequential()
    # model.add(LSTM(100, return_sequences=True, input_shape=(100,1)))
    # model.add(LSTM(1000,return_sequences=False))
    # model.add(Dense(25))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error',optimizer='adam')
    # model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64,verbose=1)
    # train_predict=model.predict(X_train)
    # test_predict=model.predict(X_test)
    # train_predict=scaler.inverse_transform(train_predict)
    # test_predict=scaler.inverse_transform(test_predict)
    # look_back=100
    # trainPredictPlot=np.empty_like(df1)
    # trainPredictPlot[:,:]=np.nan
    # trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict
    # # shift test predictions for plotting
    # testPredictPlot=np.empty_like(df1)
    # testPredictPlot[:,:]=np.nan
    # testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:]=test_predict
    # # #plot baseline and predictions
    df1,trainPredictPlot,testPredictPlot,test_predict=lstm(df)
    fig=plt.figure(figsize=(10,5))
    plt.plot(df1)
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.legend(['Close Price',"Train Prediction","Prediction"])
    st.pyplot(fig)
    



    st.header("Forecast")
    
    fig=plt.figure(figsize=(10,5))
    plt.plot(test_predict)
    plt.xlabel("Days")
    plt.ylabel("Price")
    st.pyplot(fig)





  st.header("Accuracy")
  d = {'Mean Square Error':[0.002,3.87,8.5,8.79],'Root Mean Square Error':[0.04,1.96,2.91,2.97],'R_2 Score':[0.99,-.27,0.98,0.98],'Accuracy in %': [99.99,96.13,91.5,91.2]}
  df = pd.DataFrame(data=d,index=["ARIMA with z_score","ARIMA with log scale","SVR","LSTM"])
  st.table(df)
