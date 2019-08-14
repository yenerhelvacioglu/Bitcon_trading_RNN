# LSTM Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from time import time
from sklearn.preprocessing import MinMaxScaler
# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
#from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import backend as K
from functools import partial

# Part 1 - Data Preprocessing

n_inputs = 3828 # total number of inputs minus 1 as it starts from 0
n_outputs = 3828
start_train = 6
end_train = 96
end_test = 126
batch_size = 12

# Importing the training set
data = pd.read_csv('rating_data.csv', index_col=0)

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
data_scaled = sc.fit_transform(data)

# Creating a data structure with n_windows timesteps
def create_data(data_scaled, start_train ,end_train ,n_windows):
    X_train = []
    y_train = []
    for i in range(start_train, end_train, n_windows):
        X_train.append(data_scaled[i-n_windows:i, 0:n_inputs])
        y_train.append(data_scaled[i:i+n_windows, 0:n_outputs])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_inputs))
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], n_outputs))
    print(X_train.shape,y_train.shape)
    return X_train, y_train

# Part 2 - Building the RNN

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([100]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1,0.11))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['rmsprop']))
HP_OUTPUT = hp.HParam('output_number',hp.Discrete(list(range(1))))
HP_WINDOW = hp.HParam('window_size',hp.Discrete([6]))
#HP_OUTPUT = hp.HParam('output_number',hp.Discrete(list(range(n_outputs-1))))
METRIC_ACCURACY = 'loss'

hp.hparams_config(hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_OUTPUT, HP_WINDOW], metrics=[hp.Metric(METRIC_ACCURACY, display_name='loss')],)

def run(run_dir, hparams):
    hp.hparams(hparams)  # record the values used in this trial
    predicted = train_test_model(run_dir,hparams)
    return predicted

optimizers.RMSprop(lr=0.001,rho=0.9,decay=0.9)
optimizers.SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)

def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(K.square(y_true - y_pred)*weights,axis=-1)
    return weighted_loss

def train_test_model(run_dir,hparams):
    hp.hparams(hparams)
    [X_train, y_train] = create_data(data_scaled=data_scaled, start_train=start_train, end_train=end_train, n_windows=hparams[HP_WINDOW])
    [X_test, y_test] = create_data(data_scaled=data_scaled, start_train=end_train, end_train=end_test, n_windows=hparams[HP_WINDOW])
    pd.DataFrame(np.reshape(X_test, (end_test - end_train, n_inputs))).to_csv('X.csv')
    pd.DataFrame(np.reshape(y_test, (end_test - end_train, n_outputs))).to_csv('y.csv')
    tf.compat.v1.keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(hparams[HP_NUM_UNITS], return_sequences=True ,input_shape=(hparams[HP_WINDOW], n_inputs) ,activation='tanh' ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1), dropout=hparams[HP_DROPOUT] ,recurrent_dropout=hparams[HP_DROPOUT]))
    model.add(LSTM(hparams[HP_NUM_UNITS], activation='tanh' ,return_sequences=False ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1) ,dropout=hparams[HP_DROPOUT], recurrent_dropout=hparams[HP_DROPOUT]))
    model.add(Dense(units=n_outputs, activation='linear'))
    weights = np.full([1,1,n_outputs],0.01)
    weights[0,0,0:57] = 1
    model.compile(optimizer=hparams[HP_OPTIMIZER], loss=get_weighted_loss(weights=weights)) # metrics=['mae'])
    model.summary()
    #model.load_weights('model.h5')
    model.fit(X_train, y_train[:, 0:1, :], epochs=10, batch_size=batch_size, validation_data=(X_test, y_test[:, 0:1, :]))
    #model.fit(X_train, y_train[:,0,hparams[HP_OUTPUT]:hparams[HP_OUTPUT]+1], epochs=1, batch_size=batch_size, validation_data=(X_test, y_test[:,0,hparams[HP_OUTPUT]:hparams[HP_OUTPUT]+1]))
    #model.fit(X_train, y_train[:,0,:], epochs=100, batch_size=batch_size, validation_data=(X_test, y_test[:,0,:]), callbacks=[
    #    TensorBoard(log_dir=run_dir, histogram_freq=50, write_graph=True, write_grads=True, update_freq='epoch'),
    #     hp.KerasCallback(writer=run_dir, hparams=hparams)])
    #model.save_weights('model.h5')
    X_test_extended = X_test
    for iteration in range(hparams[HP_WINDOW]):
        predicted = model.predict(X_test_extended)
        X_test_extended = np.concatenate((X_test_extended[:, -hparams[HP_WINDOW]+1:, :], np.reshape(predicted, (X_test.shape[0], 1, X_test.shape[2]))), axis=1)

    predicted = X_test_extended
    return predicted

# Part 3 - Running the model

for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in np.arange(HP_DROPOUT.domain.min_value,HP_DROPOUT.domain.max_value,0.1):
        for optimizer in HP_OPTIMIZER.domain.values:
            for output_number in HP_OUTPUT.domain.values:
                for n_windows in HP_WINDOW.domain.values:
                    hparams = {
                        HP_NUM_UNITS: num_units,
                        HP_DROPOUT: round(dropout_rate,1),
                        HP_OPTIMIZER: optimizer,
                        HP_OUTPUT: output_number,
                        HP_WINDOW: n_windows,
                    }
                    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    predicted = run('logs/hparam_tuning/' + run_name, hparams)
                    pd.DataFrame(np.reshape(predicted, ((end_test-end_train),n_outputs))).to_csv('pred' +run_name+'.csv')
