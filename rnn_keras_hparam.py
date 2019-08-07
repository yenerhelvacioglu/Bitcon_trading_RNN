# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from time import time

n_inputs = 99 # total number of inputs minus 1 as it starts from 0
n_outputs = 99
n_windows = 5
start_train = 1000
end_train = 11000
end_test = 12000
batch_size = 1000

# Importing the training set
data = pd.read_csv('data.csv', index_col=0)
#data = pd.DataFrame(data).rolling(window=50).mean()
#pd.DataFrame(data).to_csv('data_ma.csv')

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
data_scaled = sc.fit_transform(data)

# Creating a data structure with n_windows timesteps
X_train = []
y_train = []
for i in range(start_train, end_train, n_windows):
    X_train.append(data_scaled[i-n_windows:i, 0:n_inputs])
    y_train.append(data_scaled[i:i+n_windows, 0:n_outputs])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_inputs))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], n_outputs))
print(X_train.shape,y_train.shape)

X_test = []
y_test = []
for i in range(end_train, end_test, n_windows):
    X_test.append(data_scaled[i-n_windows:i, 0:n_inputs])
    y_test.append(data_scaled[i:i+n_windows, 0:n_outputs])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_inputs))
y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], n_outputs))
pd.DataFrame(np.reshape(X_test, (end_test-end_train,n_inputs))).to_csv('X.csv')
pd.DataFrame(np.reshape(y_test, (end_test-end_train,n_outputs))).to_csv('y.csv')

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
#from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorboard.plugins.hparams import api as hp

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([100]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1,0.11))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_OUTPUT = hp.HParam('output_number',hp.Discrete(list(range(n_outputs-1))))
METRIC_ACCURACY = 'loss'

hp.hparams_config(hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_OUTPUT], metrics=[hp.Metric(METRIC_ACCURACY, display_name='loss')],)

def run(run_dir, hparams):
    hp.hparams(hparams)  # record the values used in this trial
    predicted = train_test_model(run_dir,hparams)
    return predicted

optimizers.RMSprop(lr=0.001,rho=0.9,decay=0.9)

def train_test_model(run_dir,hparams):
    hp.hparams(hparams)
    tf.compat.v1.keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(hparams[HP_NUM_UNITS], return_sequences=True ,input_shape=(n_windows, n_inputs) ,activation='relu' ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1), dropout=hparams[HP_DROPOUT] ,recurrent_dropout=hparams[HP_DROPOUT]))
    model.add(LSTM(hparams[HP_NUM_UNITS], activation='relu' ,return_sequences=True ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1) ,dropout=hparams[HP_DROPOUT], recurrent_dropout=hparams[HP_DROPOUT]))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=hparams[HP_OPTIMIZER], loss='mse') # metrics=['mae'])
    model.summary()
    #model.load_weights('model.h5')
    #model.fit(X_train, y_train[:,:,hparams[HP_OUTPUT]:hparams[HP_OUTPUT]+1], epochs=1, batch_size=batch_size, validation_data=(X_test, y_test[:,:,hparams[HP_OUTPUT]:hparams[HP_OUTPUT]+1]))
    model.fit(X_train, y_train[:,:,hparams[HP_OUTPUT]:hparams[HP_OUTPUT]+1], epochs=1, batch_size=batch_size, validation_data=(X_test, y_test[:,:,hparams[HP_OUTPUT]:hparams[HP_OUTPUT]+1]), callbacks=[
         TensorBoard(log_dir=run_dir, histogram_freq=50, write_graph=True, write_grads=True, update_freq='epoch'),
         hp.KerasCallback(writer=run_dir, hparams=hparams)])
    #model.save_weights('model.h5')
    predicted = model.predict(X_test)
    return predicted

# Part 3 - Running the model

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in np.arange(HP_DROPOUT.domain.min_value,HP_DROPOUT.domain.max_value,0.1):
        for optimizer in HP_OPTIMIZER.domain.values:
            for output_number in HP_OUTPUT.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: round(dropout_rate,1),
                    HP_OPTIMIZER: optimizer,
                    HP_OUTPUT: output_number,
                }
                run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                predicted_BTC = run('logs/hparam_tuning/' + run_name, hparams)
                pd.DataFrame(np.reshape(predicted_BTC, ((end_test-end_train),1))).to_csv('pred' +run_name+'.csv')

# plt.plot(real_BTC, color = 'red', label = 'Real BTC value')
# plt.plot(predicted_BTC, color = 'blue', label = 'Predicted BTC value')
# plt.title('BTC Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('BTC Price')
# plt.legend()
# plt.show()
