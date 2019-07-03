import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys

n_inputs = 99
n_neurons = 100
n_layers = 2
n_windows = 5
n_outputs = 1
start_train = 10000
end_train = 14001
end_test = 14402
size_test = n_windows

testing = (sys.argv[-1] == "testing")
data = pd.read_csv('data.csv', index_col=0) # header=None
data = preprocessing.scale(data, axis=0)
#data = pd.DataFrame(data).rolling(window=50).mean()
pd.DataFrame(data).to_csv('data_ma.csv')

data = np.array(data)
train = data[start_train:end_train]
test = data[end_train:end_test]

X = tf.compat.v1.placeholder(tf.float32,[None,n_windows,n_inputs])
y = tf.compat.v1.placeholder(tf.float32,[None,n_windows,n_outputs])

def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

keep_prob = 0.9 #dropout keep probability rate
learning_rate = 0.001 #learning rate

layer_rnn_cell = [tf.compat.v1.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.leaky_relu) for layer in range(n_layers)]
if not testing:
    layer_rnn_cell = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in layer_rnn_cell]
multi_layer_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(layer_rnn_cell)
rnn_outputs, state = tf.compat.v1.nn.dynamic_rnn(multi_layer_rnn_cell, X, dtype=tf.float32)
stacked_rnn_outputs = tf.reshape(rnn_outputs,[-1,n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_windows, n_outputs])

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.compat.v1.global_variables_initializer()
saver = tf.train.Saver()

def create_batches(df, windows, input, output):
   ## Create X
        x_data = df[:-output]
        X_batch = x_data.reshape(-1, windows, input)
    ## Create y
        y_data = df[output:,0]
        y_batch = y_data.reshape(-1, windows, output)
        return X_batch, y_batch

X_batch, y_batch = create_batches(df = train, windows = n_windows, input = n_inputs, output = n_outputs)
X_test, y_test = create_batches(df = test, windows = n_windows, input = n_inputs, output = n_outputs)
#print(X_batch)
#print(y_batch)
#print(X_test)
#print(y_test)
#print(X_batch.shape, y_batch.shape)
#print(X_test.shape, y_test.shape)

n_iterations = 100

with tf.Session() as sess:
    if not testing:
        init.run()
        saver.restore(sess, "./my_model_final.ckpt")
        for iteration in range(n_iterations):
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
            rnn_outputs_val, state_val = sess.run([rnn_outputs, state], feed_dict={X:X_batch, y:y_batch})
            if iteration %10 == 0:
                mse_train = loss.eval(feed_dict={X:X_batch, y:y_batch})
                mse_test = loss.eval(feed_dict={X:X_test, y:y_test})
                print(iteration, "\tMSE Train:", mse_train, "\tMSE Test:", mse_test)
        save_path = saver.save(sess, "./my_model_final.ckpt")
        y_pred = sess.run(outputs, feed_dict={X:X_test})
        y_fit = sess.run(outputs, feed_dict={X:X_batch})
    else:
        saver.restore(sess, "./my_model_final.ckpt")
        y_pred = sess.run(outputs, feed_dict={X:X_test})
    #pd.DataFrame(state_val[0][:]).to_csv('state_val.csv')
    #pd.DataFrame(rnn_outputs_val[0][:]).to_csv('rnn_outputs_val.csv')

plt.title("Forecast vs Actual", fontsize=14)
plt.plot(pd.Series(np.ravel(y_test)), markersize=8, label="Actual", color='green')
plt.plot(pd.Series(np.ravel(y_pred)), markersize=8, label="Forecast", color='red')
#plt.plot(pd.Series(np.ravel(y_batch)), markersize=8, label="Actual", color='green')
#plt.plot(pd.Series(np.ravel(y_fit)), markersize=8, label="Forecast", color='red')
plt.legend(loc="lower left")
#plt.scatter(pd.DataFrame(y_pred[0][:-1]).pct_change(axis=0),pd.DataFrame(y_test[0][:]).pct_change(axis=0))
#pd.DataFrame(y_pred[0][:]).pct_change(axis=0)

plt.show()