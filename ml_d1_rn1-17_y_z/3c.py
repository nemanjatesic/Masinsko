import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
      
      accuracy = matches / qnum
      return predicted, accuracy
    

# Dataset load:
dir_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(dir_path+'/data/Prostate_Cancer.csv')
data = data.dropna()
data = data.reset_index(drop=True)
y = data.diagnosis_result
x = data.drop('diagnosis_result', axis=1)
x = x.drop('id', axis=1)


#plotting part:
seeds = [36]

#***********************************************************************************
for i in range(4):                         
    seeds.append(random.randint(0,1200))
#***********************************************************************************

acc = []
nums = []
for k in range(1,16):
    nums.append(k)
    acc.append(0)

for seed in seeds:
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, stratify=y, random_state=seed)

    train_size = len(train_y)
    test_size = len(test_y)


    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()
    change = {'B':0, 'M':1}
    train_y = np.array([change[string_class] for string_class in train_y])
    test_y = np.array([change[string_class] for string_class in test_y])

    #knn call
    num_features = 8
    num_classes = 2
    for k in range(1,16):
        knn = KNN(num_features, num_classes, train_x, train_y , k)
        pred_true, accuracy = knn.predict(test_x, test_y)
        acc[k-1] += accuracy


for i in range(15):
    acc[i]/=len(seeds)

#plot
k = np.array(nums)
accNP = np.array(acc)
plt.plot(k , accNP)
plt.xticks(k)
plt.ylabel('accuracy')
plt.xlabel('k values')
plt.title('Accuracy plot')
plt.show()

# Napomena: Poredjenje vrseno izmedju 4 i 8 featurea i tu nije velika razlika, izmedju 2 i 8 sigurno jeste posto se preciznost znacajno promeni.


# Diskusija: Analogno kao malopre, najverodostojniji podaci o tome koje k je najbolje se dobijaju racunanjem prosecnih accuracy vrednosti po
# razlicitim seedovima koji uticu na to kako ce biti rasporedjeni podaci u trening i test podatke. Pokretanja su vrsena u istim uslovima kao u 3b tj.
# sa 100 i 60 razlicitih seedova sa vrednostima do milion, i ti plotovi su sacuvani u 3_plots folderu posto njihovo generisanje traje dosta dugo.
# Poredjenjem sa prethodnim plotovima iz dela 3b, mozemo zakljuciti da se situcaija nije bas znacajno promenila, i da se znacajan porast
# preciznosti moze zabeleziti na vrednostima 5 i 14 s tim sto je sad tu i 15 kao vrednost koja ima preciznost medju najvecim.
# Kao i malopre, u kodu su ostavljene vrednosti sa malo seedova pa prikaz moze biti drugaciji, no ovo bi trebalo da je preciznije.
# Razumljivo, najgori rezultati su opet za k=2, gde dolazi do problema u bestezinskoj verziji knn-a. Ono sto se generalno moze zakljuciti jeste da ne 
# vazi da sa porastom k raste i preciznost,
# sto je i logicno, ali svakako treba birati k>=3 i ne preterivati sa velicinom. Svakako,
# ono sto se definitivno moze zakljuciti iz ovog eksperimenta je da su 5 i 14 najbolji izbori za k kada je broj featurea 4 ili 8 (sa 2 je vec drugacije)
# no svakako to nije uvek tacno jer kao sto je napomenuto postoji jos faktora koji uticu na ovu vrednost.
# Testovi se mogu videti u 3_plots folderu.

# Napomena: kad radimo sa svim featurima, u preostala 4 koja nisu koriscena u delovima 3a i 3b postoje podaci koje nemamo odnosno Na vrednosti.
# To mozemo regulisati na neki od sledecih nacina: izbrisati te redove ako ih nema mnogo (sto je i uradjeno ovde), naci prosek svih po toj koloni i
# upisati u polja koja ne postoje, ili primeniti kao knn nad ostalim featurima i nadji prosek najblizih k po koloni koja fali i upisati tu, 
# generalno to bi mogla da bude neka modifikacija ovog algoritma, koja bi potencijalno pokazivala bolje rezultate.


