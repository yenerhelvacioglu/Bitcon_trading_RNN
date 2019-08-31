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

# start_train = 50
# end_train = 4000
# end_test = 4500
# batch_size = 1000

# Importing the training set
# data = pd.read_csv('sp500.csv', index_col=1)
# data = data.values[:,2:3]#[:,::2]

n_company = 58
start_train = 12
end_train = 96
end_test = 126
batch_size = 12

# Importing the training set
data = pd.read_csv('rating_data.csv', index_col=0)

Rating = data.values[:,0:n_company]
Gross_Margin = data.values[:,1624:1624+n_company] #remove
Operating_Margin = data.values[:,2494:2494+n_company]
Net_Profit_Margin = data.values[:,2320:2320+n_company]
Return_on_Equity = data.values[:,2842:2842+n_company]
Return_on_Assets = data.values[:,2784:2784+n_company]
Current_Ratio = data.values[:,928:928+n_company] #remove
Liabilities_to_Equity_Ratio = data.values[:,1856:1856+n_company]
Debt_to_Assets_Ratio = data.values[:,986:986+n_company]
EV_EBITDA = data.values[:,1277:1277+n_company]
EV_Sales = data.values[:,1334:1334+n_company]
Book_to_Market = data.values[:,290:290+n_company]
Operating_Income_EV = data.values[:,2436:2436+n_company]
Share_Price = data.values[:,3074:3074+n_company]

# Rating[Rating == 3 ] = 2
# Rating[Rating == 4 ] = 2
# Rating[Rating == 5 ] = 3
# Rating[Rating == 6 ] = 3
# Rating[Rating == 7 ] = 3
# Rating[Rating == 8 ] = 4
# Rating[Rating == 9 ] = 4
# Rating[Rating == 10 ] = 4
# Rating[Rating == 11 ] = 5
# Rating[Rating == 12 ] = 5
# Rating[Rating == 13 ] = 5
# Rating[Rating == 14 ] = 6
# Rating[Rating == 15 ] = 6
# Rating[Rating == 16 ] = 6
# Rating[Rating >= 17 ] = 7

#data = np.concatenate((Rating,Gross_Margin, Operating_Margin, Net_Profit_Margin, Return_on_Equity, Return_on_Assets, Current_Ratio, Liabilities_to_Equity_Ratio, Debt_to_Assets_Ratio, EV_EBITDA, EV_Sales, Book_to_Market, Operating_Income_EV, Share_Price),axis=1)
data = [Rating,Gross_Margin, Operating_Margin, Net_Profit_Margin, Return_on_Equity, Return_on_Assets, Current_Ratio, Liabilities_to_Equity_Ratio, Debt_to_Assets_Ratio, EV_EBITDA, EV_Sales, Book_to_Market, Operating_Income_EV, Share_Price]

data = np.delete(data,[1,6],axis=0)
data = np.delete(data,[1,34,45,50,51,57],axis=2)

data = np.transpose(data)

# Feature Scaling
X_sc = MinMaxScaler(feature_range=(0, 1))
y_sc = MinMaxScaler(feature_range=(0, 1))

#This scales the data for x and y and creates overlapping data
def create_data(data_unscaled, start_train ,end_train ,n_windows,n_outputs):
    X = []
    y = []
    for i in range(start_train, end_train):
        for j in range(0,n_company-6):
            X.append(data_unscaled[j, i-n_windows:i, 0:n_inputs+1])
            y.append(data_unscaled[j, i-n_windows+1:i+1, n_outputs:n_outputs+1])
            #y.append(data_unscaled[i:i+n_windows, n_outputs:n_outputs+1])
    X, y = np.array(X), np.array(y)
    X_scaled = X_sc.fit_transform(np.reshape(X, (X.shape[0]* X.shape[1], X.shape[2])))
    y_scaled = y_sc.fit_transform(np.reshape(y, (y.shape[0]* y.shape[1], y.shape[2])))
    X_scaled = np.reshape(X_scaled, (X.shape[0], X.shape[1], X.shape[2]))
    y_scaled = np.reshape(y_scaled, (y.shape[0], y.shape[1], y.shape[2]))
    print(X_scaled.shape,y_scaled.shape)
    return X_scaled, y_scaled


# Part 2 - Building the RNN

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([100]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1,0.11))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['rmsprop']))
HP_OUTPUT = hp.HParam('output_number',hp.Discrete(list(range(0,12))))
HP_WINDOW = hp.HParam('window_size',hp.Discrete([12]))
HP_HORIZON = hp.HParam('horizon_size',hp.Discrete([12]))
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
    model.add(LSTM(hparams[HP_NUM_UNITS], return_sequences=True ,input_shape=(hparams[HP_WINDOW], n_inputs+1) ,activation='tanh' ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1), dropout=hparams[HP_DROPOUT] ,recurrent_dropout=hparams[HP_DROPOUT]))
    model.add(LSTM(hparams[HP_NUM_UNITS], activation='tanh' ,return_sequences=True ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1) ,dropout=hparams[HP_DROPOUT], recurrent_dropout=hparams[HP_DROPOUT]))
    model.add(LSTM(hparams[HP_NUM_UNITS], activation='tanh' ,return_sequences=True ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1) ,dropout=hparams[HP_DROPOUT], recurrent_dropout=hparams[HP_DROPOUT]))
    model.add(LSTM(hparams[HP_NUM_UNITS], activation='tanh' ,return_sequences=False ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1) ,dropout=hparams[HP_DROPOUT], recurrent_dropout=hparams[HP_DROPOUT]))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=hparams[HP_OPTIMIZER], loss='mse') #get_weighted_loss(weights=weights)) # metrics=['mae'])
    model.summary()
    model.fit(X_train, y_train[:, hparams[HP_WINDOW]-1:hparams[HP_WINDOW], 0], epochs=1, validation_data=(X_test, y_test[:, hparams[HP_WINDOW]-1:hparams[HP_WINDOW], 0]))
    #model.fit(X_train, y_train[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],0], epochs=1, batch_size=batch_size, validation_data=(X_test, y_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],0]), callbacks=[
    #    TensorBoard(log_dir=run_dir, histogram_freq=10, write_graph=True, write_grads=True, update_freq='epoch'),
    #    hp.KerasCallback(writer=run_dir, hparams=hparams)])
    model.save('model_' + str(hparams[HP_NUM_UNITS]) + '_' + str(hparams[HP_DROPOUT]) + '_' + str(hparams[HP_OPTIMIZER]) + '_' + str(hparams[HP_WINDOW]) + '_' + str(hparams[HP_OUTPUT]) + '.h5')

    return 0

def test_model(hparams):

    [X_test, y_test] = create_data(data_unscaled=data, start_train=end_train, end_train=end_test, n_windows=hparams[HP_WINDOW], n_outputs=0)
    pd.DataFrame(X_sc.inverse_transform(np.reshape(X_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:], (X_test.shape[0],X_test.shape[2])))).to_csv('X_'+ str(hparams[HP_WINDOW]) +'.csv')
    pd.DataFrame(y_sc.inverse_transform(np.reshape(y_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:], (y_test.shape[0],y_test.shape[2])))).to_csv('y_'+ str(hparams[HP_WINDOW]) +'.csv')

    X_test_extended = X_test
    for iteration_w in range(hparams[HP_WINDOW]):
        predicted = []
        for iteration_o in range(hparams[HP_OUTPUT]):
            tf.compat.v1.keras.backend.clear_session()
            model = load_model('model_' + str(hparams[HP_NUM_UNITS]) + '_' + str(hparams[HP_DROPOUT]) + '_' + str(
                hparams[HP_OPTIMIZER]) + '_' + str(hparams[HP_WINDOW]) + '_' + str(iteration_o) + '.h5')
            print(str(iteration_w) + ' ' + str(iteration_o))
            predicted.append(model.predict(X_test_extended))
        predicted = np.transpose(predicted,axes=(1,2,0))
        X_test_extended = np.concatenate((X_test_extended[:, -hparams[HP_WINDOW]+1:, :], predicted), axis=1)

    predicted = X_test_extended

    for iteration_h in hparams[HP_HORIZON]:
        pd.DataFrame(X_sc.inverse_transform(np.reshape(predicted[::iteration_h, :iteration_h, :], (predicted.shape[0], predicted.shape[2])))).to_csv('pred_' + str(iteration_h) + '.csv')

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

test_model({HP_NUM_UNITS: 100, HP_DROPOUT: round(0.1,1), HP_OPTIMIZER: 'rmsprop', HP_OUTPUT: 12, HP_WINDOW: 12, HP_HORIZON:[1,3,6,12]})