import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#napomena: radjeno sa 4 feature-a posto je tako pisalo u zadatku 

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
y = data.diagnosis_result
x = data.drop('diagnosis_result', axis=1)
x = x.drop('id', axis=1)
x = x.iloc[:,0:4]


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
    num_features = 4
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


# Diskusija: Ne moze se lepo zakljuciti realna situacija i najbolja vrednost za k na osnovu samo jednog plota,
# tj. funkcija train_test_split ce uvek drugacije preurediti podatke tj dobicemo razlicite plotove svakim pokretanjem.
# osim ako se ne prosledi svaki put isti seed, tada se uvek dobija isti plot. Ali svakako, najbolja vrednost k zavisi od
# prosledjenog seeda u podeli podataka. U prethodnom delu je koriscen seed 36 jer se najbolje pokazao za k=3 koje se zahteva u zadatku
# Ovde, da bismo dobili validnu predikciju, koje k je najbolje u generalnom slucaju potrebno je da pokusamo sa sto vise razlicitih seedova
# i nadjemo njihov prosek za svako k. To je ono sto je uradjeno u ovom kodu, zakljucak je da se uglavnom za najbolje k moze uzeti 5 ili 14
# testovi su radjeni na 100 i 60 razlicitih seedova, dok je u kodu ostavljeno da radi sa 5 zbog duzine izvrsavanja (za 100 i 60 seedova izmedju 0 i milion - oko 20min), 
# ali su ostavljeni plotovi iz tih pokretanja u predatom folderu i mogu se videti rezultati,
# koji realno ne moze bas da prikaze koje je k najbolje jer postoje varijacije u svakom pokretanju.
# deo sa biranjem seedova je izdvojen zvezdicama i lako ga je pokrenuti po potrebi unosenjem drugacijih podataka.
# Znaci zakljucak je da su najbolje vrednosti za k uglavnom 5 i 14 (racunanjem prosecnog slucaja za izbor mnogo razlicitih seedova), 
# a najgora vrednost u apsolutno svakom pokretanju je za k=2 sto je nekako i logicno iz cinjenice da ako su dva najbliza razlicitih klasa, 
# klasifikacija bude vrlo random, jer imamo netezinsku verziju algoritma.
# Napomena: prilikom pokretanja, moguce je da dodje do drugacijih rezultata zbog malog broja seedova i malog broja za izbor,
# za verodostojniji prikaz mogu se povecati ti brojevi ali sa 100 seedova, izvrsavanje traje oko 20min pa se ne preporucuje.
# Slike testiranja se nalaze u folderu 3_plots.


