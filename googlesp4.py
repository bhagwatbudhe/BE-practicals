import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


data=pd.read_csv("C:\\Users\\HP\\Downloads\\Google_Stock_Price_Train.csv")
print(data)


train = data.iloc[:1260,:]
test = data.iloc[1260:,:]


print(train)


trainset = train.iloc[:,1:2].values


print(trainset)


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_scaled = sc.fit_transform(trainset)


print(training_scaled)


x_train = []
y_train = []


for i in range(60,1259):
    x_train.append(training_scaled[i-60:i, 0])
    y_train.append(training_scaled[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)


print(x_train.shape)


x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()
regressor.add(LSTM(units = 50,return_sequences = True,input_shape = (x_train.shape[1],1)))


regressor.add(Dropout(0.2))


regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))


regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))


regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


regressor.add(Dense(units = 1))


regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')


regressor.fit(x_train,y_train,epochs = 10, batch_size = 32)


print(test)


real_stock_price = test.iloc[:,1:2].values


dataset_total = pd.concat((train['Open'],test['Open']),axis = 0)
print(dataset_total)


inputs = dataset_total[len(dataset_total) - len(test)-60:].values
print(inputs)


inputs = inputs.reshape(-1,1)

print(inputs)

inputs = sc.transform(inputs)
print(inputs.shape)

x_test = []
for i in range(60,185):
    x_test.append(inputs[i-60:i,0])

x_test = np.array(x_test)
print(x_test.shape)

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
print(x_test.shape)

predicted_price = regressor.predict(x_test)

predicted_price = sc.inverse_transform(predicted_price)
print(predicted_price)

plt.plot(real_stock_price,color = 'red', label = 'Real Price')
plt.plot(predicted_price, color = 'blue', label = 'Predicted Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()