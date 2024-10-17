

# IMPORTANT NOTE:
# I used Jupyter Notebook as my editor
# I just copied it from each cell and pasted it into this python file

#===========================================================================================

import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# Libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, LayerNormalization, Input
from keras.optimizers import SGD
import math

#===========================================================================================

# Useful Functions
def plot_predictions(test,predicted):
    plt.clf()
    plt.plot(test, color='red',label='Real AMZN Stock Price')
    plt.plot(predicted, color='blue',label='Predicted AMZN Stock Price')
    plt.title('AMZN Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('AMZN Stock Price')
    plt.legend()
    plt.show()
    
def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print(f"The root mean squared error is {rmse}")
    
def plot_loss(history, epochs):
    plt.clf()
    plt.plot(history.history['loss'], color='red', label='Loss')
    plt.title(f'Loss After {epochs} Epochs')
    plt.ylabel('Epoch')
    plt.xlabel('Loss')
    plt.legend()
    plt.show()

#===========================================================================================

# Load Data
dataset = pd.read_csv('archive\\AMZN_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
dataset.head()

#===========================================================================================

# Split Data
# Checking for missing values
training_set = dataset[:'2016'].iloc[:,1:2].values
test_set = dataset['2017':].iloc[:,1:2].values

# Visualization of data
# We have chosen 'High' attribute for prices. Let's see what it looks like
dataset["High"][:'2016'].plot(figsize=(16,4), legend=True)
dataset["High"]['2017':].plot(figsize=(16,4), legend=True)

plt.legend(['Training set (Before 2017)', 'Test set (2017 and beyond)'])
plt.title('AMZN stock price')
plt.show()

#===========================================================================================

# Preprocessing of Data
# Scaling the training set
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

# Since LSTMs store long term memory state, we create a data structure with 60 timesteps and 1 output
# So for each element of training set, we have 60 previous training set elements
X_train = []
y_train = []

window_size = 16

for i in range(window_size, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-window_size:i,0])
    y_train.append(training_set_scaled[i,0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping X_train for efficient modelling
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Preparing X_test 
dataset_total = pd.concat((dataset["High"][:'2016'], dataset["High"]['2017':]), axis=0)
inputs = dataset_total[len(dataset_total) - len(test_set) - window_size:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(window_size, len(inputs)):
    X_test.append(inputs[i-window_size:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

#===========================================================================================

# The LSTM architecture
regressor = Sequential([
    # Input Layer
    Input(shape=(window_size,1)),
    # First LSTM layer
    LSTM(64, return_sequences=True),
    # Second LSTM layer
    Bidirectional(LSTM(64)),
    # Hidden Layer
    Dense(128),
    # The output layer
    Dense(units=1)
])

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.summary()

#===========================================================================================

# Fitting to training set
epochs = 50
history = regressor.fit(
    X_train, 
    y_train, 
    epochs=epochs, 
    batch_size=32,
)

#===========================================================================================

plot_loss(history, epochs)

#===========================================================================================

# Predicting the Prices
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# Visualizing Prediction
plot_predictions(test_set, predicted_stock_price)
# Evaluating our model
return_rmse(test_set, predicted_stock_price)
print(f'Window Size = {window_size}')
print(f'Epochs = {epochs}')

#===========================================================================================
