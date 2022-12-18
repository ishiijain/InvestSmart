import numpy as np 
import pandas as pd
import pandas_datareader as data
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st

start='2010-01-01'
end='2021-12-31'

st.title('InvestSmart')
user_input=st.text_input('Enter Asset Name','MRF.NS')
df=data.DataReader(user_input,'yahoo',start,end)

#Describing Data

#st.subheader('Data From 2010-2021')
st.write(df.describe())

#Visualization

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close)
st.pyplot(fig)

data_train = pd.DataFrame(df['Close'][ 0:int(len(df)*0.70) ])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scal=MinMaxScaler(feature_range=(0,1))

data_train_arr=scal.fit_transform(data_train)

#Load My Model
model=load_model('keras_model.h5')

#Testing Part

past_100_days=data_train.tail(100)
final_df=past_100_days.append(data_test,ignore_index=True)
input_data=scal.fit_transform(final_df) 

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test=np.array(x_test),np.array(y_test)

y_pred=model.predict(x_test)

scal=scal.scale_
scal_fac=1/scal[0]
y_pred=y_pred*scal_fac
y_test=y_test*scal_fac


#Final Graph
st.subheader('Predicated vs Original')

fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_pred,'r',label='Predicated Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)





