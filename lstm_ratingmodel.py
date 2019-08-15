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
from keras.utils import to_categorical

# Part 1 - Data Preprocessing

n_inputs = 58*14 # total number of inputs minus 1 as it starts from 0
n_outputs = 58
start_train = 0
end_train = 96
end_test = 126
batch_size = 12

# Importing the training set
data = pd.read_csv('rating_data.csv', index_col=0)

Rating = data.values[:,0:n_outputs]
Gross_Margin = data.values[:,1624:1624+n_outputs]
Operating_Margin = data.values[:,2494:2494+n_outputs]
Net_Profit_Margin = data.values[:,2320:2320+n_outputs]
Return_on_Equity = data.values[:,2842:2842+n_outputs]
Return_on_Assets = data.values[:,2784:2784+n_outputs]
Current_Ratio = data.values[:,928:928+n_outputs]
Liabilities_to_Equity_Ratio = data.values[:,1856:1856+n_outputs]
Debt_to_Assets_Ratio = data.values[:,986:986+n_outputs]
EV_EBITDA = data.values[:,1277:1277+n_outputs]
EV_Sales = data.values[:,1334:1334+n_outputs]
Book_to_Market = data.values[:,290:290+n_outputs]
Operating_Income_EV = data.values[:,2436:2436+n_outputs]
Share_Price = data.values[:,3074:3074+n_outputs]

data = np.concatenate((Rating,Gross_Margin, Operating_Margin, Net_Profit_Margin, Return_on_Equity, Return_on_Assets, Current_Ratio, Liabilities_to_Equity_Ratio, Debt_to_Assets_Ratio, EV_EBITDA, EV_Sales, Book_to_Market, Operating_Income_EV, Share_Price),axis=1)
# Feature Scaling
#sc = MinMaxScaler(feature_range = (0, 1))
#data_scaled = sc.fit_transform(data)
X_sc = MinMaxScaler(feature_range=(0, 1))
y_sc = MinMaxScaler(feature_range=(0, 1))
#pd.DataFrame(data_scaled).to_csv(('rating_data_scaled.csv'))
#data_scaled = np.concatenate([data.iloc[:,0:n_outputs].values, sc.fit_transform(data.iloc[:,n_outputs:n_inputs])],axis=1)

# Creating a data structure with n_windows timesteps
def create_data(data_scaled, start_train ,end_train ,n_windows):
    X_train = []
    y_train = []
    for i in range(start_train, end_train):
        X_train.append(data_scaled[i-n_windows:i, 0:n_inputs])
        y_train.append(data_scaled[i:i+n_windows, 0:n_outputs])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_sc.fit_transform(np.reshape(X_train, (X_train.shape[0]* X_train.shape[1], n_inputs)))
    y_train = y_sc.fit_transform(np.reshape(y_train, (y_train.shape[0]* y_train.shape[1], n_outputs)))
    X_train = np.reshape(X_train, (end_train-start_train, n_windows, n_inputs))
    y_train = np.reshape(y_train, (end_train-start_train, n_windows, n_outputs))
    print(X_train.shape,y_train.shape)
    return X_train, y_train

# Part 2 - Building the RNN

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([100]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1,0.11))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['rmsprop']))
HP_OUTPUT = hp.HParam('output_number',hp.Discrete(list(range(1))))
HP_WINDOW = hp.HParam('window_size',hp.Discrete([12]))
#HP_OUTPUT = hp.HParam('output_number',hp.Discrete(list(range(n_outputs-1))))
METRIC_ACCURACY = 'accuracy'

hp.hparams_config(hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER, HP_OUTPUT, HP_WINDOW], metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],)

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
    [X_train, y_train] = create_data(data_scaled=data, start_train=hparams[HP_WINDOW], end_train=end_train, n_windows=hparams[HP_WINDOW])
    [X_test, y_test] = create_data(data_scaled=data, start_train=end_train, end_train=end_test-hparams[HP_WINDOW], n_windows=hparams[HP_WINDOW])
    pd.DataFrame(X_sc.inverse_transform(np.reshape(X_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:], ((end_test - end_train-hparams[HP_WINDOW]), n_inputs)))).to_csv('X.csv')
    pd.DataFrame(y_sc.inverse_transform(np.reshape(y_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:], ((end_test - end_train-hparams[HP_WINDOW]), n_outputs)))).to_csv('y.csv')
    tf.compat.v1.keras.backend.clear_session()
    model = Sequential()
    model.add(LSTM(hparams[HP_NUM_UNITS], unit_forget_bias=True, return_sequences=True ,input_shape=(hparams[HP_WINDOW], n_inputs) ,activation='tanh' ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1), dropout=hparams[HP_DROPOUT] ,recurrent_dropout=hparams[HP_DROPOUT]))
    #model.add(LSTM(hparams[HP_NUM_UNITS], activation='tanh' ,return_sequences=False ,kernel_initializer='TruncatedNormal' ,bias_initializer=initializers.Constant(value=0.1) ,dropout=hparams[HP_DROPOUT], recurrent_dropout=hparams[HP_DROPOUT]))
    model.add(Dense(units=n_outputs, activation='linear'))
    weights = np.full([1,1,n_outputs],1)
    #weights[0,0,0:57] = 1
    model.compile(optimizer=hparams[HP_OPTIMIZER], loss=get_weighted_loss(weights=weights)) # metrics=['mae'])
    model.summary()
    #model.load_weights('model.h5')
    model.fit(X_train, y_train[:, hparams[HP_WINDOW]-1:hparams[HP_WINDOW], :], epochs=20, validation_data=(X_test, y_test[:, hparams[HP_WINDOW]-1:hparams[HP_WINDOW], :]))
    #model.fit(X_train, y_train[:,0,hparams[HP_OUTPUT]:hparams[HP_OUTPUT]+1], epochs=1, batch_size=batch_size, validation_data=(X_test, y_test[:,0,hparams[HP_OUTPUT]:hparams[HP_OUTPUT]+1]))
    #model.fit(X_train, y_train[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:], epochs=10, batch_size=batch_size, validation_data=(X_test, y_test[:,hparams[HP_WINDOW]-1:hparams[HP_WINDOW],:]), callbacks=[
    #    TensorBoard(log_dir=run_dir, histogram_freq=50, write_graph=True, write_grads=True, update_freq='epoch'),
    #    hp.KerasCallback(writer=run_dir, hparams=hparams)])
    #model.save_weights('model.h5')
    X_test_extended = X_test
    predicted = model.predict(X_test_extended)
    # for iteration in range(hparams[HP_WINDOW]):
    #     predicted = model.predict(X_test_extended)
    #     if hparams[HP_WINDOW] != 1:
    #         X_test_extended = np.concatenate((X_test_extended[:, -hparams[HP_WINDOW]+1:, :], np.reshape(predicted, (X_test.shape[0], 1, X_test.shape[2]))), axis=1)
    #predicted = X_test_extended

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
                    #pd.DataFrame(sc.inverse_transform(np.reshape(predicted, ((end_test-end_train-n_windows)*n_windows,n_outputs)))).to_csv('pred' +run_name+'.csv')
                    #pd.DataFrame(np.reshape(predicted, ((end_test-end_train-n_windows)*n_windows,n_outputs))).to_csv('pred' +run_name+'.csv')
                    pd.DataFrame(y_sc.inverse_transform(np.reshape(predicted[:,n_windows-1:n_windows,:],(predicted.shape[0],predicted.shape[2])))).to_csv('pred' + run_name + '.csv')