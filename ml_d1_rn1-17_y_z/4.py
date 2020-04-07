import os
import pandas as pd
import re
import enchant
import numpy as np
import math
import time
import random
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

random.seed(a=None)


class MultinomialNaiveBayes:
    def __init__(self, nb_classes, nb_words, pseudocount):
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.pseudocount = pseudocount

    def fit(self, X, Y):
        nb_examples = X.shape[0]

        # Racunamo P(Klasa) - priors
        # np.bincount nam za datu listu vraca broj pojavljivanja svakog celog
        # broja u intervalu [0, maksimalni broj u listi]
        self.priors = np.bincount(Y) / nb_examples
        print('Priors:')
        print(self.priors)

        # Racunamo broj pojavljivanja svake reci u svakoj klasi
        occs = np.zeros((self.nb_classes, self.nb_words))
        for i in range(nb_examples):
            c = Y[i]
            for w in range(self.nb_words):
                cnt = X[i][w]
                occs[c][w] += cnt
        print('Occurences:')
        print(occs)

        # Racunamo P(Rec_i|Klasa) - likelihoods
        self.like = np.zeros((self.nb_classes, self.nb_words))
        for c in range(self.nb_classes):
            for w in range(self.nb_words):
                up = occs[c][w] + self.pseudocount
                down = np.sum(occs[c]) + self.nb_words * self.pseudocount
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

# corpus = ['shouldn\'t', 'cao', '#hello']

# f = open('output.txt', 'a')

cnt = 0
max = 10000

for doc in corpus:
    cnt += 1
    words = wordpunct_tokenize(doc)
    words_lower = [w.lower() for w in words]
    words_filtered = [remove_hashtag(w) for w in words_lower]
    words_filtered = [w for w in words_filtered if w not in stop_punc]
    words_filtered = [w for w in words_filtered if w.isalpha()]
    words_filtered = [too_many_chars(w) for w in words_filtered]
    words_stemmed = [porter.stem(w) for w in words_filtered]
    words_final = [w for w in words_stemmed if enchantDict.check(w)]
    clean_corpus.append(words_final)

    # f.write(str(words_final))
    # f.write('\n')

    if cnt == max:
        break

# Kreiramo vokabular
print('Creating the vocab...')
vocab_set = set()
for doc in clean_corpus:
    for word in doc:
        vocab_set.add(word)
vocab = list(vocab_set)

# print('Vocab:', list(zip(vocab, range(len(vocab)))))
print('Feature vector size: ', len(vocab))

# Bag of Words model
print('Creating BOW features...')
X = np.zeros((len(clean_corpus), len(vocab)), dtype=np.float32)
for doc_idx in range(len(clean_corpus)):
    doc = clean_corpus[doc_idx]
    for word_idx in range(len(vocab)):
        word = vocab[word_idx]
        cnt = numocc_score(word, doc)
        X[doc_idx][word_idx] = cnt

# f.close()

# creating sets for learning and testing
random_index = create_random_indexes(max)

YLearning = []
XLearning = []
for i in range(int(max * 0.8)):
    index = random_index[i]
    YLearning.append(data['y'][index])
    XLearning.append(X[index])

YTest = []
XTest = []
for i in range(int(max * 0.8), max):
    index = random_index[i]
    YTest.append(data['y'][index])
    XTest.append(X[index])

class_names = ['Positive', 'Negative']
YLearningNP = np.asarray(YLearning)
XLearningNP = np.asarray(XLearning)

model = MultinomialNaiveBayes(nb_classes=2, nb_words=len(vocab), pseudocount=1)
model.fit(XLearningNP, YLearningNP)

brojTacnih = 0
for i in range(len(YTest)):
    prediction = model.predict(np.asarray(XTest[i]))
    if class_names[prediction] == 'Positive' and YTest[i] == 0:
        brojTacnih += 1
    if class_names[prediction] == 'Negative' and YTest[i] == 1:
        brojTacnih += 1

tmpBr = brojTacnih / int(max * 0.2) * 100

print('Procent pogodjenih : ', tmpBr)

print('done')
