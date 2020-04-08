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
# 984563275 74.8058 - 100k
# 42069101 74.55 - 100k
# 11769077 74.905 -100k

# 9621128  75.1273387462807 - 100k !!!
# 53410 75.33410661152857 - 100k !!!
# 3971057 75.47027081547229 - 100k !!!
# 187372311 75.50557264612436
# 1234567 75.55600383277017
# 7465633 75.69721115537848 - 100k !!!
# 7465633 75.80311664733472 - 100k !!! tm-1
# 7465633 75.86867718997428
# 7465633 76.59350042694258
# 7465633 76.64875182078457
# 7465633 76.94241890872517 - trenutno
#35630563,24887067,98411893,1345785,6155814,22612812,21228945,7465634 - ovi imaju izmedju 75 i 75.5
seed_rand = time.time_ns()//100000000000
print(seed_rand)
random.seed(7465633)

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
        if classPN == 1:
            self.numberOfPositive += 1
        else:
            self.numberOfNegative += 1
        for w in range(len(feature_vector)):
            count = feature_vector[w]
            self.occurrences[classPN][w] += count

    def add_feature_vector_tmp(self, feature_vector, classPN):
        self.numberOfFeatures += 1
        if classPN == 1:
            self.numberOfPositive += 1
        else:
            self.numberOfNegative += 1
        for index, value in feature_vector:
            if index != -1:
                self.occurrences[classPN][index] += value

    def fit(self):
        # Racunamo P(Klasa) - priors
        # np.bincount nam za datu listu vraca broj pojavljivanja svakog celog
        # broja u intervalu [0, maksimalni broj u listi]-
        print(self.occurrences[0])
        print(self.occurrences[1])
        self.priors = np.asarray([self.numberOfNegative/self.numberOfFeatures, self.numberOfPositive/self.numberOfFeatures])
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

    def best_tweets(self, vocabulary, amount=5):
        list_negatives = []
        for i in range(len(self.occurrences[0])):
            list_negatives.append((vocabulary[i], self.occurrences[0][i]))
        list_positives = []
        for i in range(len(self.occurrences[1])):
            list_positives.append((vocabulary[i], self.occurrences[1][i]))
        list_negatives.sort(key=lambda x: x[1], reverse=True)
        list_positives.sort(key=lambda x: x[1], reverse=True)
        list_out = []
        tmp_neg = []
        tmp_pos = []
        for i in range(amount):
            # Da vrati samo string dodati [0] na kraj
            tmp_neg.append(list_negatives[i])
            tmp_pos.append(list_positives[i])
        list_out.append(tmp_neg)
        list_out.append(tmp_pos)
        return list_out

    def best_lr_tweets(self, vocabulary, amount=5):
        list_all = []
        for i in range(len(self.occurrences[0])):
            if self.occurrences[0][i] >= 10 and self.occurrences[1][i] >= 10:
                list_all.append((vocabulary[i], self.occurrences[1][i]/self.occurrences[0][i]))
        list_all.sort(key=lambda x: x[1], reverse=True)
        list_out = []
        best_tmp = []
        for i in range(amount):
            best_tmp.append(list_all[i])
        list_out.append(best_tmp)

        list_all.reverse()
        worst_tmp = []
        for i in range(amount):
            worst_tmp.append(list_all[i])
        list_out.append(worst_tmp)
        return list_out


def remove_mentions(obj):
    return re.sub(r'@\w+', '', obj)


def remove_links(obj):
    return re.sub(r'http.?://[^\s]+[\s]?', '', obj)


def remove_hashtag(obj):
     return re.sub(r'#([^\s]+)', r'\1', obj)


def remove_symbols(obj):
    return re.sub(r'[^a-zA-Z\s]', '', obj)


def too_many_chars(obj):
    #return re.sub(r'(\w)\1{2,}', r'\1', obj)
    return re.sub(r'(.)\1+', r'\1', obj)


def numocc_score(word, doc):
    return 1 if word in doc else 0
    #return doc.count(word)


def create_random_indexes(maximum):
    random_index = []
    for i in range(maximum):
        random_index.append(i)
    random.shuffle(random_index)
    return random_index


enchantDict = enchant.Dict("en_US")
max = 100000
num_of_rows = max
if num_of_rows > 90000:
    num_of_rows = None

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")
dir_path = os.path.dirname(os.path.realpath(__file__))
fileName = dir_path + os.sep + 'data' + os.sep + 'twitter.csv'
data = dict()
print('Loading file...')
data['y'] = pd.read_csv(fileName, sep=',', usecols=[1], nrows=num_of_rows).T.values.tolist()[0]
data['x'] = pd.read_csv(fileName, usecols=[2], nrows=num_of_rows, encoding="ISO-8859-1").T.values.tolist()[0]

corpus = data['x']

print('Cleaning the corpus...')
clean_corpus = []
#['to', 'it', 'that', 'is', 'my', 'in', 'have', 'me', 'im', 'so', 'be', 'out', 'wa']
forbidden = ['and','to', 'have', 'get', 'now', 'thi', 'oh', 'got', 'am', 'he', 'back', 'ned', 'so', 'at', 'today', 'it', 'that', 'is', 'my', 'in', 'have', 'me', 'im', 'so', 'be', 'out', 'wa', '']
stop_punc = set(forbidden) # set(forbidden) #set(stopwords.words('english')).union(set(punctuation))
stop_punc.add('lt')
stop_punc.add('gt')
stop_punc.add('quot')
stop_punc.add('amp')
# stop_punc.add('im')
# stop_punc.remove('no')
#stop_punc.remove('not')

#stop_punc = [remove_symbols(w) for w in stop_punc]
print(stop_punc)

cnt = 0
useFile = True

dictonary = dict()
f = open('output2.txt', 'w')
current_index = -1

for doc in corpus:
    cnt += 1
    current_index += 1
    doc = remove_mentions(doc)
    doc = remove_links(doc)
    doc = remove_hashtag(doc)
    # doc = re.sub(r'\basap\b', 'as soon as possible', doc)
    # doc = re.sub(r'\bidk\b', 'i don\'t know', doc)
    # doc = re.sub(r'\bppl\b', 'people', doc)
    # doc = re.sub(r'\bomg\b', 'oh my god', doc)
    # doc = re.sub(r'\bwtf\b', 'what the fuck', doc)
    # doc = re.sub(r'\blmao\b', 'laughing my ass off', doc)
    # doc = re.sub(r'\blol\b', 'laugh out loud', doc)
    # doc = re.sub(r'\blol\b', 'laugh out loud', doc)
    doc = remove_symbols(doc)
    words = wordpunct_tokenize(doc)
    words_lower = [w.lower() for w in words]
    words_filtered = [remove_hashtag(w) for w in words_lower]
    words_filtered = [w for w in words_filtered if w not in stop_punc]
    words_filtered = [w for w in words_filtered if w.isalpha()]
    words_filtered = [too_many_chars(w) for w in words_filtered]
    # words_filtered = [w for w in words_filtered if enchantDict.check(w)]
    words_filtered = [porter.stem(w) for w in words_filtered]
    words_filtered = [w for w in words_filtered if w not in stop_punc]
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
# MORA DA STOJI !!!
vocab.sort()


if len(vocab) > 10000:
    list_of_all_words = []
    for word in vocab:
        key = dictonary.get(word, 0)
        list_of_all_words.append((word, key))
    list_of_all_words.sort(key=lambda x: x[1], reverse=True)
    out_list = []
    print(list_of_all_words[9999][1])
    for i in range(10000):
        if list_of_all_words[i][1] >= 10:
            out_list.append(list_of_all_words[i][0])
    vocab = out_list

# print('Vocab:', list(zip(vocab, range(len(vocab)))))
print('Feature vector size: ', len(vocab))
model = MultinomialNaiveBayes(nb_classes=2, nb_words=len(vocab), pseudocount=1)

random_index = create_random_indexes(max)

# Bag of Words model
print('Creating BOW features...')

feature_vector_size = len(vocab)

vocab_dic = dict()
for i in range(feature_vector_size):
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
        new_feature_vector.append((vocab_dic.get(word, -1), number_of_occ))

    model.add_feature_vector_tmp(new_feature_vector, data['y'][doc_idx])

model.fit()

print('Checking for test set...')
brojTacnih = 0
class_names = ['Negative', 'Positive']

confusion_matrix = []
true_negatives = 0
true_positives = 0
false_negative = 0
false_positives = 0
for i in range(int(max*0.8), max):
    doc_idx = random_index[i]
    doc = clean_corpus[doc_idx]

    doc_set = set()
    for word in doc:
        doc_set.add(word)

    new_feature_vector = []

    for word in doc_set:
        number_of_occ = numocc_score(word, doc)
        new_feature_vector.append((vocab_dic.get(word, -1), number_of_occ))

    prediction = model.predict_tmp(new_feature_vector)
    if class_names[prediction] == 'Positive':
        if data['y'][doc_idx] == 1:
            true_positives += 1
            brojTacnih += 1
        else:
            false_positives += 1
    if class_names[prediction] == 'Negative':
        if data['y'][doc_idx] == 0:
            true_negatives += 1
            brojTacnih += 1
        else:
            false_negative += 1

confusion_matrix.append([true_negatives, false_positives])
confusion_matrix.append([false_negative, true_positives])

print('Confusion matrix\n', confusion_matrix)
print('Proecenat novi', (true_positives+true_negatives) / (true_positives+true_negatives+false_negative+false_positives) * 100)

# negative = [('not', 3463.0), ('no', 2935.0), ('go', 2767.0), ('dont', 2253.0), ('get', 2224.0)]
# positive = [('thank', 4022.0), ('god', 3635.0), ('love', 3295.0), ('like', 2777.0), ('u', 2324.0)]
print(model.best_tweets(vocab, amount=5))
# za generisanje forbidden-a
pos = model.best_tweets(vocab,amount=10)[1]
neg = model.best_tweets(vocab,amount=10)[0]
out = []
for str1, br1 in pos:
    for str2, br2 in neg:
        if str1 == str2 and abs(br1-br2)/(br1+br2)<=0.15:
            out.append(str1)
            break
print(model.best_lr_tweets(vocab, amount=5))
print('Done.')
print(out)