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
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, RepeatVector, Flatten, TimeDistributed, ConvLSTM2D
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

start_train = 50
end_train = 23500
end_test = 24000
batch_size = 1000

# Importing the training set
data = pd.read_csv('5minsData.csv', index_col=0)
data = data.values#[:,3:5]#[:,::2]

# Feature Scaling
X_sc = MinMaxScaler(feature_range=(0, 1))
y_sc = MinMaxScaler(feature_range=(0, 1))

#This scales the data for x and y and creates overlapping data
def create_data(data_unscaled, start_train ,end_train ,n_windows,n_outputs):
    X = []
    y = []
    for i in range(start_train, end_train):
        X.append(data_unscaled[i-n_windows:i, 0:n_inputs+1])
        #y.append(data_unscaled[i-n_windows+1:i+1, n_outputs:n_outputs+1])
        y.append(data_unscaled[i-n_windows+1:i+1, 0:n_inputs+1])
    X, y = np.array(X), np.array(y)
    X_scaled = X_sc.fit_transform(np.reshape(X, (X.shape[0]* X.shape[1], X.shape[2])))
    #y_scaled = y_sc.fit_transform(np.reshape(y[:,:,::2], (y.shape[0]* y.shape[1], int(y.shape[2]/2))))
    y_scaled = y_sc.fit_transform(np.reshape(y, (y.shape[0] * y.shape[1], y.shape[2])))

    X_scaled = np.reshape(X_scaled, (X.shape[0], X.shape[1], X.shape[2], 1))
    X_scaled = np.transpose((X_scaled[:, :, ::2], X_scaled[:, :, 1::2]), axes=(1, 2, 0, 3, 4))
    #y_scaled = np.reshape(y_scaled, (y.shape[0], y.shape[1], 1, int(y.shape[2]/2)))
    y_scaled = np.reshape(y_scaled, (y.shape[0], y.shape[1], y.shape[2], 1))
    y_scaled = np.transpose((y_scaled[:, :, ::2], y_scaled[:, :, 1::2]), axes=(1, 2, 0, 3, 4))
    print(X_scaled.shape,y_scaled.shape)
    return X_scaled, y_scaled


# Part 2 - Building the RNN

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([100]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1,0.11))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['rmsprop']))
HP_OUTPUT = hp.HParam('output_number',hp.Discrete(list(range(0,40))))
HP_WINDOW = hp.HParam('window_size',hp.Discrete([50]))
HP_HORIZON = hp.HParam('horizon_size',hp.Discrete([50]))
#HP_OUTPUT = hp.HParam('output_number',hp.Discrete(list(range(n_outputs-1))))
METRIC_ACCURACY = 'loss'

hp.hparams_config(hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_OUTPUT, HP_WINDOW, HP_HORIZON], metrics=[hp.Metric(METRIC_ACCURACY, display_name='loss')],)
optimizers.RMSprop(lr=0.001,rho=0.9,decay=0.9)
optimizers.SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)

n_inputs = max(HP_OUTPUT.domain.values)# total number of inputs minus 1 as it starts from 0

def train_model(run_dir,hparams):

    hp.hparams(hparams)
    [X_train, y_train] = create_data(data_unscaled=data, start_train=start_train, end_train=end_train, n_windows=hparams[HP_WINDOW], n_outputs=hparams[HP_OUTPUT])
    [X_test, y_test] = create_data(data_unscaled=data, start_train=end_train, end_train=end_test, n_windows=hparams[HP_WINDOW], n_outputs=hparams[HP_OUTPUT])

    tf.compat.v1.keras.backend.clear_session()
    model = Sequential()
    model.add(ConvLSTM2D(kernel_size=(1,1), filters=1, activation='relu', input_shape=(hparams[HP_WINDOW], 2, int((n_inputs+1)/2), 1), return_sequences=False))
    #model.add(LSTM(hparams[HP_NUM_UNITS], return_sequences=False ,input_shape=(hparams[HP_WINDOW], n_inputs+1) ,activation='relu' ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1), dropout=hparams[HP_DROPOUT] ,recurrent_dropout=hparams[HP_DROPOUT]))
    #model.add(LSTM(hparams[HP_NUM_UNITS], activation='tanh' ,return_sequences=True ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1) ,dropout=hparams[HP_DROPOUT], recurrent_dropout=hparams[HP_DROPOUT]))
    #model.add(LSTM(hparams[HP_NUM_UNITS], activation='tanh' ,return_sequences=True ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1) ,dropout=hparams[HP_DROPOUT], recurrent_dropout=hparams[HP_DROPOUT]))
    #model.add(RepeatVector(50))
    #model.add(LSTM(hparams[HP_NUM_UNITS], activation='relu' ,return_sequences=True ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1) ,dropout=hparams[HP_DROPOUT], recurrent_dropout=hparams[HP_DROPOUT]))
    #model.add(TimeDistributed(Dense(100, activation='relu')))
    #model.add(TimeDistributed(Dense(1)))
    #model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=hparams[HP_OPTIMIZER], loss='mse') #get_weighted_loss(weights=weights)) # metrics=['mae'])
    model.summary()
    #model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))
    model.fit(X_train, np.reshape(y_train[:, hparams[HP_WINDOW]-1:hparams[HP_WINDOW], :,:,:],(23450,2,20,1)), epochs=10, validation_data=(X_test, np.reshape(y_test[:, hparams[HP_WINDOW]-1:hparams[HP_WINDOW], :,:,:],(500,2,20,1))))
    #model.fit(X_train, y_train[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],0], epochs=1, batch_size=batch_size, validation_data=(X_test, y_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],0]), callbacks=[
    #    TensorBoard(log_dir=run_dir, histogram_freq=10, write_graph=True, write_grads=True, update_freq='epoch'),
    #    hp.KerasCallback(writer=run_dir, hparams=hparams)])
    model.save('conv_model_' + str(hparams[HP_NUM_UNITS]) + '_' + str(hparams[HP_DROPOUT]) + '_' + str(hparams[HP_OPTIMIZER]) + '_' + str(hparams[HP_WINDOW]) + '_' + str(hparams[HP_OUTPUT]) + '.h5')

    return 0

def test_model(hparams):

    [X_test, y_test] = create_data(data_unscaled=data, start_train=end_train, end_train=end_test, n_windows=hparams[HP_WINDOW], n_outputs=0)
    #pd.DataFrame(X_sc.inverse_transform(np.reshape(X_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:], (X_test.shape[0],X_test.shape[2])))).to_csv('X_'+ str(hparams[HP_WINDOW]) +'.csv')
    #pd.DataFrame(y_sc.inverse_transform(np.reshape(y_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:], (y_test.shape[0],y_test.shape[2])))).to_csv('y_'+ str(hparams[HP_WINDOW]) +'.csv')

    tf.compat.v1.keras.backend.clear_session()
    model = load_model('conv_model_' + str(hparams[HP_NUM_UNITS]) + '_' + str(hparams[HP_DROPOUT]) + '_' + str(
        hparams[HP_OPTIMIZER]) + '_' + str(hparams[HP_WINDOW]) + '_0.h5')

    X_test_extended = X_test
    for iteration_w in range(hparams[HP_WINDOW]):
        predicted = []
        for iteration_o in range(hparams[HP_OUTPUT]):
            print(str(iteration_w) + ' ' + str(iteration_o))
            predicted.append(model.predict(X_test_extended))
        predicted = np.transpose(predicted,axes=(1,0,2,3,4))
        X_test_extended = np.concatenate((X_test_extended[:, -hparams[HP_WINDOW]+1:, :, :], predicted), axis=1)

    predicted = X_test_extended

    pd.DataFrame(X_sc.inverse_transform(np.reshape(predicted[::50, :50, :, :, 0], (500, 40)))).to_csv('conv_pred.csv')

    #for iteration_h in hparams[HP_HORIZON]:
    #    pd.DataFrame(X_sc.inverse_transform(np.reshape(predicted[::iteration_h, :iteration_h, :], (predicted.shape[0], predicted.shape[2])))).to_csv('pred_' + str(iteration_h) + '.csv')

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

test_model({HP_NUM_UNITS: 100, HP_DROPOUT: round(0.1,1), HP_OPTIMIZER: 'rmsprop', HP_OUTPUT: 1, HP_WINDOW: 50, HP_HORIZON:[1,5,10,25,50]})
