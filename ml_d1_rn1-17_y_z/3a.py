import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# napomena: kad koristimo prva 4 featurea (kao sto pise u zad) preciznost je 0.92, dok je sa 2 featurea 0.6
# tako da se jasno vidi da sa povecanjem featurea koje koristimo povecavamo preciznost


class KNN:
  
  def __init__(self, num_features, num_classes, train_x, train_y, k):
    self.num_features = num_features
    self.num_classes = num_classes
    self.x_data = train_x
    self.y_data = train_y
    self.k = k
    
    #Computation graph init:
    self.X = tf.placeholder(shape=(None, num_features), dtype=tf.float32)
    self.Y = tf.placeholder(shape=(None), dtype=tf.int32)
    self.Q = tf.placeholder(shape=(num_features), dtype=tf.float32)
    
    dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.X, self.Q)), axis=1))
    _, idxs = tf.nn.top_k(-dist, self.k)  
    
    self.classes = tf.gather(self.Y, idxs)
    self.distances = tf.gather(dist, idxs)
    
    self.classes_one_hot = tf.one_hot(self.classes, num_classes, on_value=1/k, off_value=0.0)
    self.scores = tf.reduce_sum(self.classes_one_hot, axis=0)
    
    self.sol = tf.argmax(self.scores)
  
  def predict(self, query_x, query_y):
    predicted = np.zeros((len(query_y),), dtype=int)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      qnum = len(query_y)
      matches = 0
      for i in range(qnum):
        solution = sess.run(self.sol, feed_dict = {self.X: self.x_data, self.Y: self.y_data, self.Q: query_x[i]})
        if query_y is not None:
          actual = query_y[i]
          if solution == actual:
            matches += 1
          predicted[i] = int(solution)
          #change = { 0:'B', 1:'M'}
          #print('Test example: {}/{}| Predicted: {}| Actual: {}' .format(i+1, qnum, change[solution], change[actual]))
      
      accuracy = matches / qnum
      return predicted, accuracy
    

# Dataset load:
dir_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(dir_path+'/data/Prostate_Cancer.csv')
y = data.diagnosis_result
x = data.drop('diagnosis_result', axis=1)
x = x.drop('id', axis=1)
x = x.iloc[:,0:2]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=36)

train_size = len(train_y)
test_size = len(test_y)


train_x = train_x.to_numpy()
train_y = train_y.to_numpy()
test_x = test_x.to_numpy()
test_y = test_y.to_numpy()
change = {'B':0, 'M':1}
train_y = np.array([change[string_class] for string_class in train_y])
test_y = np.array([change[string_class] for string_class in test_y])


num_features = 2
num_classes = 2
k = 3
knn = KNN(num_features, num_classes, train_x, train_y , k)
pred_true, accuracy = knn.predict(test_x, test_y)
print('\n\nTest set accuracy: ', accuracy)
 
size = 100
x1, x2 = np.meshgrid(np.linspace(min(train_x[:, 0]), max(train_x[:, 0]), num=size),
                      np.linspace(min(train_x[:, 1]), max(train_x[:, 1]), num=size))
x_fill = np.vstack((x1.flatten(), x2.flatten())).T
y_fill = np.zeros((x_fill.shape[0],), dtype=int)

pred_val,acc = knn.predict(x_fill, y_fill)
pred_plot = pred_val.reshape([x1.shape[0], x1.shape[1]])

from matplotlib.colors import LinearSegmentedColormap
classes_cmap = LinearSegmentedColormap.from_list('classes_cmap', 
                                                  ['lightgreen', 
                                                  'lightpink'])
plt.contourf(x1, x2, pred_plot, cmap=classes_cmap, alpha=0.7)

idxs_0 = train_y == 0.0
idxs_1 = train_y == 1.0
plt.scatter(train_x[idxs_0, 0], train_x[idxs_0, 1], c='g', 
            edgecolors='k', label='B')
plt.scatter(train_x[idxs_1, 0], train_x[idxs_1, 1], c='r', 
            edgecolors='k', label='M')
plt.show()


# Neki pokusaji redukcije dimenzija za prvobitni zadatak sa 4-dim:

#***************************************************************************************************************
#v3:
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
 
# scaler = StandardScaler().fit(train_x)
# x = scaler.transform(train_x)
 
# pca = PCA(2)
# pca_fit = pca.fit(x)
# x = pca.transform(x)

# size = 10
# x1, x2, x3, x4 = np.meshgrid(np.linspace(min(train_x[:, 0]), max(train_x[:, 0]), num=size),
#                       np.linspace(min(train_x[:, 1]), max(train_x[:, 1]), num=size),
#                       np.linspace(min(train_x[:, 2]), max(train_x[:, 2]), num=size),
#                       np.linspace(min(train_x[:, 3]), max(train_x[:, 3]), num=size))
# x_fill = np.vstack((x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten())).T
# y_fill = np.zeros((x_fill.shape[0],), dtype=int)

# pred_fill, _ = knn.predict(x_fill, y_fill)
# pred_fill_plot = pred_fill.reshape([10, 1000])

# x_fill_proj = pca_fit.transform(x_fill)
# x1_fill_plot = x_fill_proj[:,0].reshape([10, 1000])
# x2_fill_plot = x_fill_proj[:,1].reshape([10, 1000])
 
# from matplotlib.colors import LinearSegmentedColormap
# classes_cmap = LinearSegmentedColormap.from_list('classes_cmap',
#                                                 ['lightblue',
#                                                 'lightgreen'])
# plt.contourf(x1_fill_plot, x2_fill_plot, pred_fill_plot, cmap=classes_cmap, alpha=0.7)
# plt.show()




#******************************************************************************************************************
#v2:
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
 
# scaler = StandardScaler().fit(train_x)
# x = scaler.transform(train_x)
 
# pca = PCA(2)
# pca_fit = pca.fit(x)
# x = pca.transform(x)
# size = 100
# x1, x2 = np.meshgrid(np.linspace(min(x[:, 0]), max(x[:, 0]), num=size),
#                     np.linspace(min(x[:, 1]), max(x[:, 1]), num=size))
# x_fill = np.vstack((x1.flatten(), x2.flatten())).T
# assert x_fill.shape[-1] == 2
# x_fill_reproject = pca_fit.inverse_transform(x_fill)
# y_fill = np.zeros((x_fill.shape[0],), dtype=int)

# x_reproject = pca_fit.inverse_transform(x)

# pred_fill, _ = knn.predict(x_fill_reproject , y_fill)
# print(sum(pred_fill))
# pred_fill_plot = pred_fill.reshape([x1.shape[0], x1.shape[1]])
 
# from matplotlib.colors import LinearSegmentedColormap
# classes_cmap = LinearSegmentedColormap.from_list('classes_cmap',
#                                                 ['lightblue',
#                                                 'lightgreen'])
# plt.contourf(x1, x2, pred_fill_plot, cmap=classes_cmap, alpha=0.7)
# plt.show()

#************************************************************************************************************************

