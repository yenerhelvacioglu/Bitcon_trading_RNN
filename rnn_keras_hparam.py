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
start_train = 13900
end_train = 14000
end_test = 14400
batch_size = 100

# Importing the training set
data = pd.read_csv('data.csv', index_col=0)
len(data.columns)
#data = data.iloc[:,0:1].values

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
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 98))

X_test = []
y_test = []
for i in range(end_train, end_test):
    X_test.append(data_scaled[i-n_windows:i, 0:98])
    y_test.append(data_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 98))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
'''
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from time import time
from keras.callbacks import TensorBoard
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from time import time
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

#tf.enable_eager_execution()

#logdir = 'logs/hparam_tuning'
#tensorboard = TensorBoard(log_dir='logs/{}'.format(time()), histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([3, 6]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
METRIC_ACCURACY = 'val_loss'

#with tf.compat.v1.summary.FileWriter('logs/hparam_tuning'):
hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='val_loss')],
    )

def run(run_dir, hparams):
#    with tf.compat.v1.summary.FileWriter(run_dir):
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(run_dir,hparams)
        #tf.compat.v1.summary.scalar(METRIC_ACCURACY, accuracy)
    return 0

session_num = 0
                        
# Compiling the RNN
#regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

def train_test_model(run_dir,hparams):
    hp.hparams(hparams)
    model = Sequential()
    model.add(LSTM(hparams[HP_NUM_UNITS], return_sequences=True, input_shape=(n_windows, 98)))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(LSTM(units=3))
    model.add(Dense(units=1))
    model.compile(optimizer=hparams[HP_OPTIMIZER], loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=(X_test, y_test),callbacks=[TensorBoard(log_dir=run_dir, histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch'), hp.KerasCallback(writer=run_dir, hparams=hparams)])
    return history.history['val_loss']

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                HP_NUM_UNITS: num_units,
                HP_DROPOUT: dropout_rate,
                HP_OPTIMIZER: optimizer,
            }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/hparam_tuning/' + run_name, hparams)
            session_num += 1

#predicted_BTC = regressor.predict(inputs)

# plt.plot(real_BTC, color = 'red', label = 'Real BTC value')
# plt.plot(predicted_BTC, color = 'blue', label = 'Predicted BTC value')
# plt.title('BTC Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('BTC Price')
# plt.legend()
# plt.show()
