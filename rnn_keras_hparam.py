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

n_inputs = 3 # total number of inputs minus 1 as it starts from 0
#n_outputs = 1
start_train = 23000
end_train = 23500
end_test = 24000
batch_size = 1000

# Importing the training set
data = pd.read_csv('5minsData.csv', index_col=0)
data = data.values[:,::2]

# Feature Scaling
X_sc = MinMaxScaler(feature_range=(0, 1))
y_sc = MinMaxScaler(feature_range=(0, 1))

#Creating a data structure with n_windows timesteps
#for non-overlapping data
# def create_data(data_scaled, start_train ,end_train ,n_windows):
#     X_train = []
#     y_train = []
#     for i in range(start_train, end_train, n_windows):
#         X_train.append(data_scaled[i-n_windows:i, 0:n_inputs])
#         y_train.append(data_scaled[i:i+n_windows, 0:n_outputs])
#     X_train, y_train = np.array(X_train), np.array(y_train)
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_inputs))
#     y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], n_outputs))
#     print(X_train.shape,y_train.shape)
#     return X_train, y_train

#for overlapping data
# def create_data(data_scaled, start_train ,end_train ,n_windows):
#     X_train = []
#     y_train = []
#     for i in range(start_train, end_train):
#         X_train.append(data_scaled[i-n_windows:i, 0:n_inputs])
#         y_train.append(data_scaled[i-n_windows+1:i+1, 0:n_outputs])
#     X_train, y_train = np.array(X_train), np.array(y_train)
#     X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_inputs))
#     y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], n_outputs))
#     print(X_train.shape,y_train.shape)
#     return X_train, y_train

#This scales the data for x and y and creates overlapping data
def create_data(data_scaled, start_train ,end_train ,n_windows,n_outputs):
    X_train = []
    y_train = []
    for i in range(start_train, end_train):
        X_train.append(data_scaled[i-n_windows:i, 0:n_inputs])
        y_train.append(data_scaled[i-n_windows+1:i+1, n_outputs:n_outputs+1])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train_scaled = X_sc.fit_transform(np.reshape(X_train, (X_train.shape[0]* X_train.shape[1], X_train.shape[2])))
    y_train_scaled = y_sc.fit_transform(np.reshape(y_train, (y_train.shape[0]* y_train.shape[1], y_train.shape[2])))
    X_train_scaled = np.reshape(X_train_scaled, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    y_train_scaled = np.reshape(y_train_scaled, (y_train.shape[0], y_train.shape[1], y_train.shape[2]))
    print(X_train_scaled.shape,y_train_scaled.shape)
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_inputs))
    # y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], n_outputs))
    # print(X_train.shape,y_train.shape)
    return X_train_scaled, y_train_scaled

# Part 2 - Building the RNN

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([100]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1,0.11))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['rmsprop']))
HP_OUTPUT = hp.HParam('output_number',hp.Discrete(list(range(0,3))))
HP_WINDOW = hp.HParam('window_size',hp.Discrete([2]))
#HP_OUTPUT = hp.HParam('output_number',hp.Discrete(list(range(n_outputs-1))))
METRIC_ACCURACY = 'loss'

hp.hparams_config(hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_OUTPUT, HP_WINDOW], metrics=[hp.Metric(METRIC_ACCURACY, display_name='loss')],)

def run(run_dir, hparams):
    hp.hparams(hparams)  # record the values used in this trial
    train_model(run_dir,hparams)
    predicted = predict_model(hparams)
    return predicted

optimizers.RMSprop(lr=0.001,rho=0.9,decay=0.9)
optimizers.SGD(lr=0.001,decay=1e-6,momentum=0.9,nesterov=True)

def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(K.square(y_true - y_pred)*weights,axis=-1)
    return weighted_loss

def train_model(run_dir,hparams):

    hp.hparams(hparams)
    # Create data
    [X_train, y_train] = create_data(data_scaled=data, start_train=start_train, end_train=end_train, n_windows=hparams[HP_WINDOW], n_outputs=hparams[HP_OUTPUT])
    [X_test, y_test] = create_data(data_scaled=data, start_train=end_train, end_train=end_test, n_windows=hparams[HP_WINDOW], n_outputs=hparams[HP_OUTPUT])

    tf.compat.v1.keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(hparams[HP_NUM_UNITS], return_sequences=False ,input_shape=(hparams[HP_WINDOW], n_inputs) ,activation='tanh' ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1), dropout=hparams[HP_DROPOUT] ,recurrent_dropout=hparams[HP_DROPOUT]))
    #model.add(LSTM(hparams[HP_NUM_UNITS], activation='tanh' ,return_sequences=False ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1) ,dropout=hparams[HP_DROPOUT], recurrent_dropout=hparams[HP_DROPOUT]))
    model.add(Dense(units=1, activation='linear'))
    #weights = np.full([1,1,n_outputs],0)
    #weights[0,0,hparams[HP_OUTPUT]] = 1
    model.compile(optimizer=hparams[HP_OPTIMIZER], loss='mse') #get_weighted_loss(weights=weights)) # metrics=['mae'])
    model.summary()
    #model.load_weights('model.h5')

    # use this when using calling n_inputs = n_outputs
    #model.fit(X_train, y_train[:, 0, :], epochs=1, batch_size=batch_size, validation_data=(X_test, y_test[:, 0, :]))
    #model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=(X_test, y_test))
    #model.fit(X_train, y_train[:,0,hparams[HP_OUTPUT]:hparams[HP_OUTPUT]+1], epochs=1, batch_size=batch_size, validation_data=(X_test, y_test[:,0,hparams[HP_OUTPUT]:hparams[HP_OUTPUT]+1]))
    # model.fit(X_train, y_train[:,0,:], epochs=1, batch_size=batch_size, validation_data=(X_test, y_test[:,0,:]), callbacks=[
    #      TensorBoard(log_dir=run_dir, histogram_freq=10, write_graph=True, write_grads=True, update_freq='epoch'),
    #      hp.KerasCallback(writer=run_dir, hparams=hparams)])

    #use this if return_sequences = True
    #model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), batch_size=batch_size)
    #model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[
    #    TensorBoard(log_dir=run_dir, histogram_freq=10, write_graph=True, write_grads=True, update_freq='epoch'),
    #    hp.KerasCallback(writer=run_dir, hparams=hparams)])

    # use this if return_sequences = False
    model.fit(X_train, y_train[:, hparams[HP_WINDOW]-1:hparams[HP_WINDOW], 0], epochs=1, validation_data=(X_test, y_test[:, hparams[HP_WINDOW]-1:hparams[HP_WINDOW], 0]))
    #model.fit(X_train, y_train[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],0], epochs=1, batch_size=batch_size, validation_data=(X_test, y_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],0]), callbacks=[
    #    TensorBoard(log_dir=run_dir, histogram_freq=10, write_graph=True, write_grads=True, update_freq='epoch'),
    #    hp.KerasCallback(writer=run_dir, hparams=hparams)])

    model.save('model_' + str(hparams[HP_OUTPUT]) + '_.h5')
    #model.save_weights('model.h5')

    return 0

def predict_model(hparams):

    # use this when using calling n_inputs = n_outputs
    # X_test_extended = X_test
    # for iteration in range(hparams[HP_WINDOW]):
    #     predicted = model.predict(X_test_extended)
    #     X_test_extended = np.concatenate((X_test_extended[:, -hparams[HP_WINDOW]+1:, :], np.reshape(predicted, (X_test.shape[0], 1, X_test.shape[2]))), axis=1)
    #
    # predicted = X_test_extended

    # use this when n_inputs =! n_outputs
    # predicted = model.predict(X_test)



    # use this when n_inputs =! n_outputs but you want to interpolate rest of the X
    [X_test, y_test] = create_data(data_scaled=data, start_train=end_train, end_train=end_test, n_windows=hparams[HP_WINDOW], n_outputs=hparams[HP_OUTPUT])

    X_test_extended = X_test
    for iteration_w in range(hparams[HP_WINDOW]):
        predicted = []
        for iteration_i in range(n_inputs):
            model = load_model('model_' + str(iteration_i) + '_.h5')
            print(str(iteration_w) + ' ' + str(iteration_i))
            predicted.append(model.predict(X_test_extended))
        predicted = np.transpose(predicted,axes=(1,2,0))
        X_test_extended = np.concatenate((X_test_extended[:, -hparams[HP_WINDOW]+1:, :], predicted), axis=1)

    predicted = X_test_extended

    pd.DataFrame(X_sc.inverse_transform(np.reshape(predicted[::n_windows, :, :], (end_test - end_train, n_inputs)))).to_csv('pred.csv')

    #X_test_inversed = np.around(X_sc.inverse_transform(np.reshape(X_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:],(X_test.shape[0],X_test.shape[2]))))[:,0]
    #y_test_inversed = np.around(y_sc.inverse_transform(y_test[:, :, 0]))
    #pd.DataFrame(X_sc.inverse_transform(np.reshape(X_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:], (X_test.shape[0],X_test.shape[2])))).to_csv('X_'+ str(hparams[HP_WINDOW]) +'.csv')
    #pd.DataFrame(y_sc.inverse_transform(np.reshape(y_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:], (y_test.shape[0],y_test.shape[2])))).to_csv('y_'+ str(hparams[HP_WINDOW]) +'.csv')
    #pd.DataFrame(X_test_inversed).to_csv('X_'+ str(hparams[HP_WINDOW]) +'.csv')
    #pd.DataFrame(y_test_inversed).to_csv('y_'+ str(hparams[HP_WINDOW]) +'.csv')

    #use this if return_sequences = True
    #predicted_inversed = y_sc.inverse_transform(predicted[:, :, hparams[HP_OUTPUT]])
    #pd.DataFrame(predicted_inversed).to_csv('pred' + run_dir[19:34] + '_' + str(hparams[HP_WINDOW]) + '.csv')
    #pd.DataFrame(np.reshape(predicted_inversed[0::n_windows], (end_test - end_train))).to_csv('pred' + run_dir[19:34] + '.csv')

    #use this if return_sequences = False
    #predicted_inversed = np.around(y_sc.inverse_transform(predicted))
    #pd.DataFrame(predicted_inversed).to_csv('pred' + run_dir[19:34] + '_' + str(hparams[HP_WINDOW]) + '.csv')

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
                    # Create data after scaling
                    # [X_train, y_train] = create_data(data_scaled=data_scaled, start_train=start_train, end_train=end_train, n_windows=hparams[HP_WINDOW])
                    # [X_test, y_test] = create_data(data_scaled=data_scaled, start_train=end_train, end_train=end_test, n_windows=hparams[HP_WINDOW])

                    #train_model(run_dir, hparams)
                    predict_model(hparams)