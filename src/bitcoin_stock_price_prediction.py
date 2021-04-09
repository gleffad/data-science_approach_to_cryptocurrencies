# importing required tools
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

plot.style.use('fivethirtyeight')

# other features data and csv export
'''
gold = web.DataReader('GC=F', data_source='yahoo',start='2016-01-01', end='2020-06-08')
silver = web.DataReader('SI=F', data_source='yahoo',start='2016-01-01', end='2020-06-08')
crude_oil = web.DataReader('CL=F', data_source='yahoo', start='2016-01-01', end='2020-06-08')
natural_gas = web.DataReader('NG=F', data_source='yahoo', start='2016-01-01', end='2020-06-08')
euros_usd = web.DataReader('EURUSD=X', data_source='yahoo', start='2016-01-01', end='2020-06-08')

gold.to_csv('/Users/x/_/git/github/data-science_approach_to_cryptocurrencies/builded_datasets/gold_dataset.csv', index=False, sep=',')
silver.to_csv('/Users/x/_/git/github/data-science_approach_to_cryptocurrencies/builded_datasets/silver_dataset.csv', index=False, sep=',')
crude_oil.to_csv('/Users/x/_/git/github/data-science_approach_to_cryptocurrencies/builded_datasets/crude_oil_dataset.csv', index=False, sep=',')
natural_gas.to_csv('/Users/x/_/git/github/data-science_approach_to_cryptocurrencies/builded_datasets/natural_gas_dataset.csv', index=False, sep=',')
euros_usd.to_csv('/Users/x/_/git/github/data-science_approach_to_cryptocurrencies/builded_datasets/euro_usd.csv', index=False, sep=',')
bitcoin.to_csv('/Users/x/_/git/github/data-science_approach_to_cryptocurrencies/builded_datasets/bitcoin_dataset.csv', index=False, sep=',')

print(gold)
print(silver)
print(crude_oil)
print(natural_gas)
print(euros_usd)
'''
# retreving data from yahoo finance
print("...retreving data from yahoo finance...")
bitcoin = web.DataReader('BTC-USD', data_source='yahoo',
                         start='2010-01-01', end='2020-06-08')

# overview of the datas
print("\n...overview of the datas...\n")
print(bitcoin)

# getting info on the dataframe
print("\n...getting info on the dataframe...\n")
bitcoin.info()

# getting the number of columns and rows
print("\n...getting the number of columns and rows\...\n")
bitcoin.shape

# visualization of the bitcoin closing course
print("\n...visualization of the bitcoin closing course...\n")
plot.figure(figsize=(16, 8))
plot.title('Bitcoin close price history')
plot.plot(bitcoin['Close'])
plot.xlabel('Date')
plot.ylabel('Close Price USD')
plot.savefig('../graphs/original_stock.png')

# fitering column close of the dataframe
print("\n...fitering column close of the dataframe...\n")
data = bitcoin.filter(['Close'])

# converting the dataframe to a numpy array
print("\n...converting the dataframe to a numpy array...\n")
dataset = data.values

# choosing the % of value where the model is going to train, and get number of rows
print("\n...get the number of rows to train the model on...\n")
training_data_len = math.ceil(len(dataset) * .7)
training_data_len

# scaling the data between 0 and 1
print("\n...scaling the data between 0 and 1...\n")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

# creating the trained dataset and scaling it
print("\n...creating the trained dataset and scaling it...\n")
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0])
    y_train.append(train_data[i, 0])
    if i <= 60:
        print(x_train)
        print(y_train)
        print()

# convert x_train and y_train to a numpy array
print("\n...convert x_train and y_train to a numpy array...\n")
x_train, y_train = np.array(x_train), np.array(y_train)

# reshape the data
print("\n...reshape the data...\n")
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

# bulding the LSTM model
print("\n...bulding the LSTM model...\n")
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# model compilation
print("\n...model compilation...\n")
model.compile(optimizer='adam', loss='mean_squared_error')

# Model training
print("\n...Model training...\n")
model.fit(x_train, y_train, batch_size=1, epochs=1)

# creating testing dataset and a new array containing scaled values
print("\n...creating testing dataset and a new array containing scaled values...\n")
test_data = scaled_data[training_data_len - 60:, :]

# create the data sets x_test and y_test
print("\n...create the data sets x_test and y_test...\n")
x_test = []
y_test = dataset[training_data_len]
for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, 0])

# convert the data to a numpy array
print("\n...convert the data to a numpy array...\n")
x_test = np.array(x_test)

# reshape data
print("\n...reshape data...\n")
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# get forecast
print("\n...get forecast...\n")
forecast = model.predict(x_test)
forecast = scaler.inverse_transform(forecast)

# get root mean squared error (RMSE)
print("\n...get root mean squared error (RMSE)...\n")
rmse = np.sqrt(np.mean(forecast - y_test) ** 2)
print(rmse)

# visualize datas
print("\n..visualize datas...\n")
train = data[:training_data_len]
valid = data[training_data_len:]
valid['forecast'] = forecast
plot.figure(figsize=(16, 8))
plot.title('Model')
plot.xlabel('Date')
plot.ylabel('Bitcoin price USD')
plot.plot(train['Close'])
plot.plot(valid[['Close', 'forecast']])
plot.legend(['Train', 'Val', 'forecast'], loc='lower right')
plot.savefig('../graphs/forecast_result.png')

print("...\ncomparing forecast to original datas\n...")
print(valid)
