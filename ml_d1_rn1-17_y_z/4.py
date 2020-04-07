import os
import pandas as pd
import re
import enchant
import numpy as np
import math
import time
import random
from collections import Counter
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import words

# 29842830 71.2 - 10k - recnikON
# 29842830 76.1 - 10k - recnikOFF
# 29842830 76.8 - 5k - recnikOFF
# 29842830 75.0 - 5k - onjgenFixON
# 29842830 71.35 - 20k - onjgenFixON
# 29842830 73.35 - 20k - onjgenFixOFF
# 29842830 76.18 - 5k - izbacivanjePraznihBezRecnika
# 29842830 75.79 - 5k - izbacivanjePraznihSaRecnikom
# 29842830 73.02 - 20k - izbacivanjePraznihSaRecnikom
# 29842830 74.53 - 20k - izbacivanjePraznihBezRecnikom

random.seed(73979309)

class MultinomialNaiveBayes:
    def __init__(self, nb_classes, nb_words, pseudocount):
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.pseudocount = pseudocount
        self.like = np.zeros((self.nb_classes, self.nb_words))
        self.occurrences = np.zeros((self.nb_classes, self.nb_words))
        self.numberOfFeatures = 0
        self.numberOfPositive = 0
        self.numberOfNegative = 0

    def add_feature_vector(self, feature_vector, classPN):
        # Racunamo broj pojavljivanja svake reci u svakoj klasi
        self.numberOfFeatures += 1
        if classPN == 0:
            self.numberOfPositive += 1
        else:
            self.numberOfNegative += 1
        for w in range(len(feature_vector)):
            count = feature_vector[w]
            self.occurrences[classPN][w] += count

    def add_feature_vector_tmp(self, feature_vector, classPN):
        self.numberOfFeatures += 1
        if classPN == 0:
            self.numberOfPositive += 1
        else:
            self.numberOfNegative += 1
        for index, number_of_occs in feature_vector:
            if index != -1:
                self.occurrences[classPN][index] += number_of_occ

    def fit(self):
        # Racunamo P(Klasa) - priors
        # np.bincount nam za datu listu vraca broj pojavljivanja svakog celog
        # broja u intervalu [0, maksimalni broj u listi]
        self.priors = np.asarray([self.numberOfPositive/self.numberOfFeatures, self.numberOfNegative/self.numberOfFeatures])
        print('Priors:')
        print(self.priors)

        # Racunamo P(Rec_i|Klasa) - likelihoods
        for c in range(self.nb_classes):
            for w in range(self.nb_words):
                up = self.occurrences[c][w] + self.pseudocount
                down = np.sum(self.occurrences[c]) + self.nb_words * self.pseudocount
                self.like[c][w] = up / down
        print('Likelihoods:')
        print(self.like)

    def predict(self, bow):
        # Racunamo P(Klasa|bow) za svaku klasu
        probs = np.zeros(self.nb_classes)
        for c in range(self.nb_classes):
            prob = np.log(self.priors[c])
            for w in range(self.nb_words):
                cnt = bow[w]
                prob += cnt * np.log(self.like[c][w])
            probs[c] = prob
        # Trazimo klasu sa najvecom verovatnocom
        prediction = np.argmax(probs)
        return prediction

    def predict_tmp(self, bow):
        # Racunamo P(Klasa|bow) za svaku klasuu
        probs = np.zeros(self.nb_classes)
        for c in range(self.nb_classes):
            prob = np.log(self.priors[c])
            for index, value in bow:
                if index != -1:
                    prob += value * np.log(self.like[c][index])
            probs[c] = prob
        # Trazimo klasu sa najvecom verovatnocom
        prediction = np.argmax(probs)
        return prediction

    def predict_multiply(self, bow):
        # Racunamo P(Klasa|bow) za svaku klasu
        # Mnozimo i stepenujemo kako bismo uporedili rezultate sa slajdovima
        probs = np.zeros(self.nb_classes)
        for c in range(self.nb_classes):
            prob = self.priors[c]
            for w in range(self.nb_words):
                cnt = bow[w]
                prob *= self.like[c][w] ** cnt
            probs[c] = prob
        # Trazimo klasu sa najvecom verovatnocom
        print('\"Probabilities\" for a test BoW (without log):')
        print(probs)
        prediction = np.argmax(probs)
        return prediction


def remove_hashtag(obj):
    return re.sub(r'#([^\s]+)', r'\1', obj)


def too_many_chars(obj):
    return re.sub(r'(.)\1+', r'\1\1', obj)


def numocc_score(word, doc):
    return doc.count(word)


def create_random_indexes(maximum):
    random_index = []
    for i in range(maximum):
        random_index.append(i)
    random.shuffle(random_index)
    return random_index


enchantDict = enchant.Dict("en_US")

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")
dir_path = os.path.dirname(os.path.realpath(__file__))
fileName = dir_path + os.sep + 'data' + os.sep + 'twitter.csv'
data = dict()
print('Loading file...')
data['y'] = pd.read_csv(fileName, sep=',', usecols=[1]).T.values.tolist()[0]
data['x'] = pd.read_csv(fileName, usecols=[2], encoding="ISO-8859-1").T.values.tolist()[0]

corpus = data['x']

print('Cleaning the corpus...')
clean_corpus = []
stop_punc = set(stopwords.words('english')).union(set(punctuation))

cnt = 0
max = len(data['y'])
useFile = False

dictonary = dict()
f = open('output1.txt', 'w')
current_index = 0
for doc in corpus:
    cnt += 1
    current_index += 1
    words = wordpunct_tokenize(doc)
    words_lower = [w.lower() for w in words]
    words_filtered = [remove_hashtag(w) for w in words_lower]
    words_filtered = [w for w in words_filtered if w not in stop_punc]
    words_filtered = [w for w in words_filtered if w.isalpha()]
    words_filtered = [too_many_chars(w) for w in words_filtered]
    # words_filtered = [w for w in words_filtered if enchantDict.check(w)]
    words_filtered = [porter.stem(w) for w in words_filtered]

    for word in words_filtered:
        key = dictonary.get(word, 0)
        dictonary[word] = key + 1

    if len(words_filtered) > 0:
        clean_corpus.append(words_filtered)
    else:
        data['y'].pop(current_index)
        current_index -= 1

    if useFile: f.write(str(words_filtered))
    if useFile: f.write('\n')
    if cnt == max:
        break

max = len(clean_corpus)
f.close()
# Kreiramo vokabular
print('Creating the vocab...')
vocab_set = set()
for doc in clean_corpus:
    for word in doc:
        vocab_set.add(word)
vocab = list(vocab_set)

if len(vocab) > 10000:
    list_of_all_words = []
    for word in vocab:
        key = dictonary.get(word, 0)
        list_of_all_words.append((word, key))
    list_of_all_words.sort(key=lambda x: x[1], reverse=True)
    out_list = []
    for i in range(10000):
        out_list.append(list_of_all_words[i][0])
    vocab = out_list

# print('Vocab:', list(zip(vocab, range(len(vocab)))))
print('Feature vector size: ', len(vocab))
model = MultinomialNaiveBayes(nb_classes=2, nb_words=len(vocab), pseudocount=1)

# Bag of Words model
# print('Creating BOW features...')
# X = np.zeros((len(clean_corpus), len(vocab)), dtype=np.float32)
# for doc_idx in range(len(clean_corpus)):
#     doc = clean_corpus[doc_idx]
#     for word_idx in range(len(vocab)):
#         word = vocab[word_idx]
#         cnt = numocc_score(word, doc)
#         X[doc_idx][word_idx] = cnt


######################################################################

random_index = create_random_indexes(max)

# Bag of Words model
print('Creating BOW features...')

feature_vector_size = len(vocab)

# 5% done
broj = int(int(max*0.8)*0.05)
tmp = 0

vocab_dic = dict()
for i in range(len(vocab)):
    vocab_dic[vocab[i]] = i

for i in range(int(max*0.8)):
    doc_idx = random_index[i]
    doc = clean_corpus[doc_idx]
    doc_set = set()
    for word in doc:
        doc_set.add(word)

    new_feature_vector = []

    for word in doc_set:
        number_of_occ = numocc_score(word, doc)
        new_feature_vector.append((vocab_dic.get(word,-1), number_of_occ))

    model.add_feature_vector_tmp(new_feature_vector, data['y'][doc_idx])


    # new_feature_vector = [0 for i in range(feature_vector_size)]
    # for word_idx in range(feature_vector_size):
    #     word = vocab[word_idx]
    #     cnt = numocc_score(word, doc)
    #     new_feature_vector[word_idx] = cnt
    if i % broj == 0:
        print(tmp, '%')
        tmp += 5
    # model.add_feature_vector(new_feature_vector, data['y'][doc_idx])

model.fit()

print('Checking for test set...')
brojTacnih = 0
class_names = ['Positive', 'Negative']


broj = int(max*0.2*0.05)
tmp = 0
for i in range(int(max*0.8), max):
    doc_idx = random_index[i]
    doc = clean_corpus[doc_idx]

    doc_set = set()
    for word in doc:
        doc_set.add(word)

    new_feature_vector = []

    for word in doc_set:
        number_of_occ = numocc_score(word, doc)
        new_feature_vector.append((vocab_dic.get(word,-1), number_of_occ))

    # new_feature_vector = [0 for i in range(feature_vector_size)]
    # for word_idx in range(feature_vector_size):
    #     word = vocab[word_idx]
    #     cnt = numocc_score(word, doc)
    #     new_feature_vector[word_idx] = cnt
    if i % broj == 0:
        print(tmp, '%')
        tmp += 5
    # prediction = model.predict(np.asarray(new_feature_vector))
    prediction = model.predict_tmp(new_feature_vector)
    if class_names[prediction] == 'Positive' and data['y'][doc_idx] == 0:
        brojTacnih += 1
    if class_names[prediction] == 'Negative' and data['y'][doc_idx] == 1:
        brojTacnih += 1

tmpBr = brojTacnih / int(max * 0.2) * 100

print('Procent pogodjenih : ', tmpBr)

######################################################################

# f.close()


# # creating sets for learning and testing
# print('Creating sets for learning and testing...')
#
# YLearning = []
# XLearning = []
# for i in range(int(max * 0.8)):
#     index = random_index[i]
#     YLearning.append(data['y'][index])
#     XLearning.append(X[index])
#
# YTest = []
# XTest = []
# for i in range(int(max * 0.8), max):
#     index = random_index[i]
#     YTest.append(data['y'][index])
#     XTest.append(X[index])
#
# class_names = ['Positive', 'Negative']
# YLearningNP = np.asarray(YLearning)
# XLearningNP = np.asarray(XLearning)
#
# print('Starting to fit...')
#
# model.fit(XLearningNP, YLearningNP)
#
# print('Checking for test set...')
# brojTacnih = 0
# for i in range(len(YTest)):
#     prediction = model.predict(np.asarray(XTest[i]))
#     if class_names[prediction] == 'Positive' and YTest[i] == 0:
#         brojTacnih += 1
#     if class_names[prediction] == 'Negative' and YTest[i] == 1:
#         brojTacnih += 1
#
# tmpBr = brojTacnih / int(max * 0.2) * 100
#
# print('Procent pogodjenih : ', tmpBr)

print('done')
