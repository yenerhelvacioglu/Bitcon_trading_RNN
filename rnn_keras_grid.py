# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n_inputs = 99
n_neurons = 100
n_layers = 2
n_windows = 10
n_outputs = 1
start_train = 10000
end_train = 14000
end_test = 14400
batch_size = 100

# Importing the training set
data = pd.read_csv('data.csv', index_col=0)
len(data.columns)
#data = data.iloc[:,0:1].values

#dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
#training_set = dataset_train.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range = (0, 1))
sc = MinMaxScaler(feature_range = (0, 1))
#training_set_scaled = sc.fit_transform(training_set)
data_scaled = sc.fit_transform(data)

sc_x = MinMaxScaler(feature_range = (0, 1))
x_scaled = sc_x.fit_transform(data.iloc[:,0:1].values)

# Creating a data structure with 60 timesteps and t+1 output
X_train = []
y_train = []
for i in range(start_train, end_train):
    X_train.append(data_scaled[i-n_windows:i, 0:98])
    y_train.append(data_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 98))
# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from time import time
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit

def build_regressor(optimizer):
    # Initialising the RNN
    regressor = Sequential()
    # Adding the input layer and the LSTM layer
    regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (n_windows,98)))
    regressor.add(Dropout(0.2))
    # Adding a second LSTM layer
    regressor.add(LSTM(units = 3, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Adding a third LSTM layer
    regressor.add(LSTM(units = 3, return_sequences = True))
    regressor.add(Dropout(0.2))
    # Adding a fourth LSTM layer
    regressor.add(LSTM(units = 100))
    regressor.add(Dropout(0.2))
    # Adding the output layer
    regressor.add(Dense(units = 1))
    # Compiling the RNN
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error',metrics=['mse', 'mae', 'mape', 'cosine'])
    return regressor

regressor = KerasRegressor(build_fn = build_regressor)
parameters = {'batch_size': [25, 32],
              'epochs': [1, 2],
              'optimizer': ['Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam','SGD','rmsprop']
#              'units' : [100,50]
#              'loss' : ['mean_squared_error','mean_squared_logarithmic_error','mean_absolute_error']
              }
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           cv=ShuffleSplit(n_splits=1))
# Part 3 - Making the predictions and visualising the results

# Getting the predicted BTC price
#scaled_real_stock_price = sc.fit_transform(real_stock_price)
inputs = []
for i in range(end_train, end_test):
    inputs.append(data_scaled[i-n_windows:i, 0:98])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 98))

# Visualising the results
real_BTC=data_scaled[end_train:end_test,0]
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()), histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')


#regressor = load_model('kerasmodel.h5')
# Fitting the RNN to the Training set
grid_search.fit(X_train, y_train, callbacks=[tensorboard])
#regressor.save('kerasmodel.h5')
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print('Best Parameters are :',best_parameters)
print('Best Score is :',best_accuracy)


"""
predicted_BTC = regressor.predict(inputs)

plt.plot(real_BTC, color = 'red', label = 'Real BTC value')
plt.plot(predicted_BTC, color = 'blue', label = 'Predicted BTC value')
plt.title('BTC Price Prediction')
plt.xlabel('Time')
plt.ylabel('BTC Price')
plt.legend()
plt.show()
"""