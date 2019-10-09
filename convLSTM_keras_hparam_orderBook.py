# LSTM Neural Network

# Importing the libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from time import time
from sklearn.preprocessing import MinMaxScaler
# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential, load_model
#from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Reshape, Dense, LSTM, Dropout, Input, ConvLSTM2D, Conv1D, Flatten, AveragePooling1D, MaxPooling1D, Embedding, GlobalMaxPooling1D, TimeDistributed, BatchNormalization, MaxPooling3D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras.utils import to_categorical
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import backend as K
from functools import partial
from scipy import optimize

# Part 1 - Data Preprocessing

start_train = 300
end_train = 600
end_test = 900
batch_size = 100

# Importing the training set
#data = pd.read_csv('5minsData.csv', index_col=0)
#data = data.values[:,::2]

dataUSDT_BTC = np.load('USDT_BTC_orderbook.npy',allow_pickle=True)
dataUSDT_ETH = np.load('USDT_ETH_orderbook.npy',allow_pickle=True)
dataBTC_ETH = np.load('BTC_ETH_orderbook.npy',allow_pickle=True)
data = np.transpose([dataUSDT_BTC,dataUSDT_ETH,dataBTC_ETH],axes=(1,0,2,3))
data = np.concatenate((data[:, :, :,::2],np.cumsum(data[:, :, :, 1::2],axis=2)),axis=3)

ydataUSDT_BTC = np.load('USDT_BTC_close.npy',allow_pickle=True)
ydataUSDT_ETH = np.load('USDT_ETH_close.npy',allow_pickle=True)
ydataBTC_ETH = np.load('BTC_ETH_close.npy',allow_pickle=True)
ydata = np.transpose([ydataUSDT_BTC,ydataUSDT_ETH,ydataBTC_ETH])

#y_cat = to_categorical(data[49:-10,0,0]>data[59:,0,0])
#y_cat = np.reshape([y_cat]*50,(941,50,2))
# def func(x, a, b):
#     return a * np.sin(b * x)
#
# xdata = np.linspace(1, len(data), len(data), dtype=int)
# popt, pcov = optimize.curve_fit(func, xdata, data[:,0])
#
# plt.plot(xdata, func(data, *popt))
# #plt.plot(xdata, data)
# plt.show()

# Feature Scaling
X_sc = MinMaxScaler(feature_range=(0, 1))
y_sc = MinMaxScaler(feature_range=(0, 1))

#This scales the data for x and y and creates overlapping data
def create_data(data_unscaled, ydata_unscaled, start_train ,end_train ,n_windows,n_outputs):
    X = []
    y = []
    for i in range(start_train, end_train):
        X.append(data_unscaled[i-n_windows:i, :, :, 0:n_inputs+1])
        y.append(ydata_unscaled[i:i+n_windows, 0:3])
    X, y = np.transpose(np.array(X,ndmin=5),axes=(0,1,2,3,4)), np.transpose(np.array(y,ndmin=3),axes=(0,1,2))
    X_scaled = X_sc.fit_transform(np.reshape(X, (X.shape[0]* X.shape[1] * X.shape[2] * X.shape[3], X.shape[4])))
    y_scaled = y_sc.fit_transform(np.reshape(y, (y.shape[0]* y.shape[1], y.shape[2])))
    X_scaled = np.reshape(X_scaled, (X.shape[0], X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
    y_scaled = np.reshape(y_scaled, (y.shape[0], y.shape[1], y.shape[2]))
    print(X_scaled.shape,y_scaled.shape)
    return X_scaled, y_scaled


# Part 2 - Building the RNN

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([100]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2,0.21))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
HP_OUTPUT = hp.HParam('output_number',hp.Discrete(list(range(3,4))))
HP_WINDOW = hp.HParam('window_size',hp.Discrete([300]))
HP_HORIZON = hp.HParam('horizon_size',hp.Discrete([300]))
#HP_OUTPUT = hp.HParam('output_number',hp.Discrete(list(range(n_outputs-1))))
METRIC_ACCURACY = 'loss'

hp.hparams_config(hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_OUTPUT, HP_WINDOW, HP_HORIZON], metrics=[hp.Metric(METRIC_ACCURACY, display_name='loss')],)
optimizers.RMSprop(lr=0.001,rho=0.9,decay=0.9)
optimizers.SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)

n_inputs = max(HP_OUTPUT.domain.values)# total number of inputs minus 1 as it starts from 0

def train_model(run_dir,hparams):

    hp.hparams(hparams)
    [X_train, y_train] = create_data(data_unscaled=data, ydata_unscaled=ydata, start_train=start_train, end_train=end_train, n_windows=hparams[HP_WINDOW], n_outputs=hparams[HP_OUTPUT])
    [X_test, y_test] = create_data(data_unscaled=data, ydata_unscaled=ydata, start_train=end_train, end_train=end_test, n_windows=hparams[HP_WINDOW], n_outputs=hparams[HP_OUTPUT])

    #X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1]/10), 10, X_train.shape[2])
    y_train = y_train[:, hparams[HP_WINDOW]-1, :] > y_train[:, 0, :]
    y_train = np.reshape(np.tile(y_train, hparams[HP_WINDOW]), (y_train.shape[0], hparams[HP_WINDOW], y_train.shape[1]))
    y_test = y_test[:, hparams[HP_WINDOW]-1, :] > y_test[:, 0, :]
    y_test = np.reshape(np.tile(y_test, hparams[HP_WINDOW]), (y_test.shape[0], hparams[HP_WINDOW], y_test.shape[1]))

    tf.compat.v1.keras.backend.clear_session()
    model = Sequential()
    #model.add(Input(shape=(None,10,40), dtype='float32'))
    #model.add(TimeDistributed(Conv1D(50,kernel_size=10,activation='relu',padding='same'),input_shape=(None,10,40)))
    model.add(ConvLSTM2D(kernel_size=(5,4), filters=6, padding = 'same', activation='relu', input_shape=(hparams[HP_WINDOW], 3, 50, 4), return_sequences=True, data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 5, 2), padding='same', data_format='channels_first'))
    model.add(ConvLSTM2D(kernel_size=(5,2), filters=3, padding = 'same', activation='relu', return_sequences=True, data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 10, 2), padding='same', data_format='channels_first'))
    #model.add(ConvLSTM2D(kernel_size=(4,1), filters=1, padding = 'same', activation='relu', return_sequences=False, data_format='channels_first'))
    model.add(TimeDistributed(Flatten()))
    #model.add(TimeDistributed(Dense(4)))
    #model.add(ConvLSTM2D(kernel_size=(10,4), filters=1, padding = 'same', activation='relu', return_sequences=False, data_format='channels_first'))
    #model.add(Conv1D(100, kernel_size=10, activation='relu', padding='same', input_shape=(50, 20)))
    #model.add(Conv1D(50, kernel_size=10, activation='relu', padding='same', input_shape=(50, 20)))
    #model.add(Conv1D(50,kernel_size=50,padding='causal',activation='relu',input_shape=(50,40),dilation_rate=1))
    #model.add(Conv1D(50,kernel_size=50,padding='causal',activation='relu',dilation_rate=2))
    #model.add(Conv1D(50,kernel_size=50,padding='causal',activation='relu',dilation_rate=4))
    #model.add(Conv1D(50,kernel_size=50,padding='causal',activation='relu',dilation_rate=8))
    #model.add(MaxPooling1D(pool_size=10))
    #model.add(Conv1D(50,kernel_size=10,padding='same',activation='relu'))
    #model.add(MaxPooling1D(pool_size=2))
    #model.add(LSTM(hparams[HP_NUM_UNITS], return_sequences=True ,input_shape=(hparams[HP_WINDOW], n_inputs+1) ,activation='tanh' ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1), dropout=hparams[HP_DROPOUT] ,recurrent_dropout=hparams[HP_DROPOUT]))
    #model.add(LSTM(40, return_sequences=True))
    #model.add(LSTM(20, return_sequences=True))
    #model.add(LSTM(2, return_sequences=False, input_shape=(50,2)))
    #model.add(LSTM(1, return_sequences=False))
    #model.add(Flatten())
    #model.add(LSTM(40, return_sequences=True, activation='relu'))
    #model.add(LSTM(20, return_sequences=False, activation='relu'))
    #model.add(Dense(100,activation='relu'))
    #model.add(Flatten())
    #model.add(Dropout(0.2))
    #model.add(GlobalMaxPooling1D())
    #model.add(Dense(20,activation='linear'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.compile(loss='mse', optimizer='adam')
    model.summary()
    #model.fit(X_train, y_cat[:750], epochs=1, batch_size=batch_size, validation_data=(X_test[:-50], y_cat[750:900]))

    model.fit(X_train, y_train, epochs=3, batch_size=batch_size, validation_data=(X_test, y_test))
    #model.fit(X_train, y_train[:, 49, :, :, :], epochs=1, batch_size=batch_size, validation_data=(X_test, y_test[:, 49, :, :, :]))
    #model.fit(X_train[:, :, :], to_categorical(y_train[:, 49:59, 0]>X_train[:, 49:59, 0]), epochs=10)

    # model.add(LSTM(hparams[HP_NUM_UNITS], return_sequences=False ,input_shape=(hparams[HP_WINDOW], n_inputs+1) ,activation='tanh' ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1), dropout=hparams[HP_DROPOUT] ,recurrent_dropout=hparams[HP_DROPOUT]))
    # model.add(LSTM(hparams[HP_NUM_UNITS], activation='tanh' ,return_sequences=False ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1) ,dropout=hparams[HP_DROPOUT], recurrent_dropout=hparams[HP_DROPOUT]))
    # model.add(Dense(units=1, activation='linear'))
    # model.compile(optimizer=hparams[HP_OPTIMIZER], loss='mse') #get_weighted_loss(weights=weights)) # metrics=['mae'])
    # model.summary()
    #model.fit(X_train, y_train[:, hparams[HP_WINDOW]-1:hparams[HP_WINDOW], 0], epochs=1, validation_data=(X_test, y_test[:, hparams[HP_WINDOW]-1:hparams[HP_WINDOW], 0]))
    #model.fit(X_train, y_train[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:], epochs=100, batch_size=batch_size, validation_data=(X_test, y_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:]), callbacks=[
    #    TensorBoard(log_dir=run_dir, histogram_freq=100, write_graph=True, write_grads=True, update_freq='epoch'),
    #    hp.KerasCallback(writer=run_dir, hparams=hparams)])
    model.save('model_' + str(hparams[HP_NUM_UNITS]) + '_' + str(hparams[HP_DROPOUT]) + '_' + str(hparams[HP_OPTIMIZER]) + '_' + str(hparams[HP_WINDOW]) + '_' + str(hparams[HP_OUTPUT]) + '.h5')
    #model.save_weights('model_weights_' + str(hparams[HP_NUM_UNITS]) + '_' + str(hparams[HP_DROPOUT]) + '_' + str(hparams[HP_OPTIMIZER]) + '_' + str(hparams[HP_WINDOW]) + '_' + str(hparams[HP_OUTPUT]) + '.h5')

    return 0

def test_model(hparams):

    [X_test, y_test] = create_data(data_unscaled=data, ydata_unscaled=ydata, start_train=end_train, end_train=end_test, n_windows=hparams[HP_WINDOW], n_outputs=0)
    #pd.DataFrame(X_sc.inverse_transform(np.reshape(X_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:], (X_test.shape[0],X_test.shape[2])))).to_csv('X_'+ str(hparams[HP_WINDOW]) +'.csv')
    #pd.DataFrame(y_sc.inverse_transform(np.reshape(y_test, (y_test.shape[0]*y_test.shape[1],y_test.shape[2])))).to_csv('y_'+ str(hparams[HP_WINDOW]) +'.csv')
    pd.DataFrame(y_sc.inverse_transform(np.reshape(y_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:], (y_test.shape[0], y_test.shape[2])))).to_csv('y_' + str(hparams[HP_WINDOW]) + '.csv')

    tf.compat.v1.keras.backend.clear_session()
    model = load_model('model_' + str(hparams[HP_NUM_UNITS]) + '_' + str(hparams[HP_DROPOUT]) + '_' + str(hparams[HP_OPTIMIZER]) + '_' + str(hparams[HP_WINDOW]) + '_' + '3' + '.h5')

    # X_test_extended = X_test
    # for iteration_w in range(hparams[HP_WINDOW]):
    #     predicted = []
    #     for iteration_o in range(hparams[HP_OUTPUT]):
    #         #tf.compat.v1.keras.backend.clear_session()
    #         #model.load_weights('model_weights_' + str(hparams[HP_NUM_UNITS]) + '_' + str(hparams[HP_DROPOUT]) + '_' + str(hparams[HP_OPTIMIZER]) + '_' + str(hparams[HP_WINDOW]) + '_' + str(iteration_o) + '.h5')
    #         print(str(iteration_w) + ' ' + str(iteration_o))
    #         predicted.append(model.predict(X_test_extended))
    #     #predicted = np.transpose(predicted,axes=(1,0,2))
    #     predicted = np.transpose(predicted,axes=(1,0,2,3,4))
    #     X_test_extended = np.concatenate((X_test_extended[:, -hparams[HP_WINDOW]+1:, :, :, :], predicted[:, :, :, :, :]), axis=1)

    #predicted = X_test_extended
    #predicted[:, :, :, :, 1].sort()

    # for iteration_h in hparams[HP_HORIZON]:
    #     pd.DataFrame(X_sc.inverse_transform(np.reshape(predicted[::iteration_h, :iteration_h, :,0,:], (predicted.shape[0]*predicted.shape[2], predicted.shape[4],)))).to_csv('pred_' + str(iteration_h) + '.csv')

    predicted = model.predict(X_test)
    #pd.DataFrame(y_sc.inverse_transform(np.reshape(predicted, (predicted.shape[0]*predicted.shape[1], predicted.shape[2])))).to_csv('pred.csv')
    pd.DataFrame(y_sc.inverse_transform(np.reshape(predicted[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:], (predicted.shape[0], predicted.shape[2])))).to_csv('pred.csv')

    return 0

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
                    run_dir = os.path.join("logs/hparam_tuning/", run_name)
                    train_model(run_dir, hparams)

test_model({HP_NUM_UNITS: 100, HP_DROPOUT: round(0.2,1), HP_OPTIMIZER: 'adam', HP_OUTPUT: 1, HP_WINDOW: 300, HP_HORIZON:[1,300]})
