import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np
import os
from copy import copy

#Feature matrix init function:
def create_feature_matrix(x, nb_features):
  tmp_features = []
  for deg in range(1, nb_features+1):
    tmp_features.append(np.power(x, deg))
  return np.column_stack(tmp_features)

# Regression function:
def polynomial_regression(nb_features, data_x, data_y):
  data_x = create_feature_matrix(data_x, nb_features)
  
  # Model init:
  X = tf.placeholder(shape=(None, nb_features), dtype=tf.float32)
  Y = tf.placeholder(shape=(None), dtype=tf.float32)
  w = tf.Variable(tf.zeros(nb_features))
  bias = tf.Variable(0.0)

  w_col = tf.reshape(w, (nb_features, 1))
  hyp = tf.add(tf.matmul(X, w_col), bias)

  Y_col = tf.reshape(Y, (-1, 1))
  
  mse = tf.reduce_mean(tf.square(hyp - Y_col)) #loss
  opt_op = tf.train.AdamOptimizer().minimize(mse) #optimization

  # Training:
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
  
    nb_epochs = 100
    for epoch in range(nb_epochs):
      epoch_loss = 0
      for sample in range(nb_samples):
        _, curr_loss = sess.run([opt_op, mse], feed_dict={X: data_x[sample].reshape((1, nb_features)), 
                                                          Y: data_y[sample]})
        epoch_loss += curr_loss
 
    # Plotting and calc: 
    w_val = sess.run(w)
    bias_val = sess.run(bias)
    print('w = ', w_val, 'bias = ', bias_val)
    xs = create_feature_matrix(np.linspace(-4, 4, 100), nb_features)
    hyp_val = sess.run(hyp, feed_dict={X: xs})
    plt.plot(xs[:, 0].tolist(), hyp_val.tolist(), color='g')
    total_loss = sess.run(mse, feed_dict={X:data_x, Y:data_y})
    return total_loss

# *********************************************************************************
# Main calls:

# Load data:
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + '/data/corona.csv'
x, y = np.loadtxt(filename, delimiter=',', unpack=True)

# Shuffle data:
nb_samples = x.shape[0]
indices = np.random.permutation(nb_samples, )
x = x[indices]
y = y[indices]

# Normalization:
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

# First plot:
plt.figure(figsize=(11,4))
plt.subplot(121)

# Plot data:
plt.scatter(x , y, color='r')

# Plot curves:
nb_features_arr = [1,2,3,4,5,6]
loss_arr = []
for nb in nb_features_arr:
    loss_arr.append(polynomial_regression(nb, x, y))
plt.ylabel('y value')
plt.xlabel('x value')

# Second plot:
plt.subplot(122)
plt.plot(nb_features_arr, loss_arr)
plt.xticks(nb_features_arr)
plt.ylabel('loss')
plt.xlabel('polynomial degree')


plt.show()


# Diskusija:
# Sa drugog grafika se jasno vidi da imamo manji trosak za vece stepene polinoma, tj. za stepene >=3 i to da je jako blizu nuli (nije 0, vidi se kad se ispise)
# Sto je i logicno zbog vecih mogucnosti tih polinomijalnih funkcija. Ono sto je bitno kod ovih metoda je naci
# optimalnu velicinu za stepen i to: da bude sto manja a koliko toliko tacna, ne treba se preprilagoditi trening podacima tako da nije dobro kad smo bas mnogo blizu nule.
# (zato ne radimo interpolaciju vec aproksimaciju nekom funkcijom), jer onda gubimo sposobnost generalizacije, i novi podaci mogu imati manju tacnost.


