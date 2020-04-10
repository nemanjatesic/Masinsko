import os
import pandas as pd
import re
import numpy as np
import random
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
import sys


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
        self.priors = np.asarray([self.numberOfNegative/self.numberOfFeatures, self.numberOfPositive/self.numberOfFeatures])
        print('Priors:')
        print(self.priors)

        # Racunamo P(Rec_i|Klasa) - likelihoods
        for c in range(self.nb_classes):
            for w in range(self.nb_words):
                up = self.occurrences[c][w] + self.pseudocount
                down = np.sum(self.occurrences[c]) + self.nb_words * self.pseudocount
                self.like[c][w] = up / down
        print('Finished fitting')

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
                # Moze da se desi da se neka rec pojavila manje od 10 puta i da smo je
                # izbacili za takve reci saljemo -1 i ne ubacujemo ih
                if index != -1:
                    prob += value * np.log(self.like[c][index])
            probs[c] = prob
        # Trazimo klasu sa najvecom verovatnocom
        prediction = np.argmax(probs)
        return prediction

    def best_tweets(self, vocabulary, amount=5):
        # Napravi dve liste pozitivnih i negativnih reci i njihovih ponavljanja u occurrences matrici
        # sortiramo po broju ponavljanja i uzmemo 5 najboljih
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
        # Slicno kao best_tweets samo sto umesto da vidimo koliko puta se sta desilo i stavimo to
        # kao drugi element samo stavimo LR(rec) = countPos(rec)/countNeg(rec) i sortiramo po tome
        # i uzimamo 5 najboljih i 5 najgorih
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


def print_progress_bar(i, max, postText):
    n_bar = 10
    j = i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()


def remove_mentions(obj):
    return re.sub(r'@\w+', '', obj)


def remove_links(obj):
    return re.sub(r'http.?://[^\s]+[\s]?', '', obj)


def remove_hashtag(obj):
    return re.sub(r'#([^\s]+)', r'\1', obj)


def remove_symbols(obj):
    return re.sub(r'[^a-zA-Z\s]', '', obj)


def too_many_chars(obj):
    # return re.sub(r'(\w)\1{2,}', r'\1', obj)
    return re.sub(r'(.)\1+', r'\1', obj)


def numocc_score(word, doc):
    return 1 if word in doc else 0
    # return doc.count(word)


def create_random_indexes(maximum):
    random_index = []
    for i in range(maximum):
        random_index.append(i)
    random.shuffle(random_index)
    return random_index


def load_data(num_of_rows):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fileName = dir_path + os.sep + 'data' + os.sep + 'twitter.csv'
    data = dict()
    print('Loading file...')
    data['y'] = pd.read_csv(fileName, sep=',', usecols=[1], nrows=num_of_rows).T.values.tolist()[0]
    data['x'] = pd.read_csv(fileName, usecols=[2], nrows=num_of_rows, encoding="ISO-8859-1").T.values.tolist()[0]
    return data


random.seed(7465633)
# Broj twitova koji zelimo da ucitamo
max = 100000
# Broj redova koji zelimo da ucitamo iz fajla
num_of_rows = max
if num_of_rows > 90000:
    # None ucitava sve automatski
    num_of_rows = None


data = load_data(num_of_rows)
corpus = data['x']


forbidden = ['and', 'to', 'have', 'get', 'now', 'thi', 'oh', 'got', 'am', 'he', 'back', 'lt', 'gt', 'quot', 'amp',
             'ned', 'so', 'at', 'it', 'my', 'that', 'is', 'in', 'have', 'me', 'im', 'so', 'be', 'out', 'wa', '']
stop_punc = set(forbidden)

# Mapirati svaku rec na njen broj pojavljivanja
dictonary = dict()

clean_corpus = []
porter = PorterStemmer()
cnt = 0
current_index = -1

# Filtriranje tvitova
print('Cleaning the corpus...')
for doc in corpus:
    if cnt % (max//100) == 0:
        print_progress_bar(cnt, max, 'cleaned')
    cnt += 1
    current_index += 1

    doc = remove_mentions(doc)
    doc = remove_links(doc)
    doc = remove_hashtag(doc)
    doc = remove_symbols(doc)

    words = wordpunct_tokenize(doc)

    words_filtered = [w.lower() for w in words]
    words_filtered = [remove_hashtag(w) for w in words_filtered]
    words_filtered = [w for w in words_filtered if w not in stop_punc]
    words_filtered = [w for w in words_filtered if w.isalpha()]
    words_filtered = [too_many_chars(w) for w in words_filtered]
    words_filtered = [porter.stem(w) for w in words_filtered]
    words_filtered = [w for w in words_filtered if w not in stop_punc]

    # Ubacivanje u mapu koliko se svaka rec pojavila
    for word in words_filtered:
        key = dictonary.get(word, 0)
        dictonary[word] = key + 1

    # Proverite da li postoji najmanje jedna reč u filtriranim recima
    if len(words_filtered) > 0:
        clean_corpus.append(words_filtered)
    else:
        # Ako smo sve filtrirali, ne zelimo da ga dodamo i te podatke moramo popovati iz data['y']
        data['y'].pop(current_index)
        current_index -= 1

    if cnt == max:
        break

print_progress_bar(max, max, 'cleaned')

# Moramo da promenimo max u slucaju da smo izbacili neke twitove
max = len(clean_corpus)
print('\nCreating the vocab...')
vocab_set = set()
for doc in clean_corpus:
    for word in doc:
        vocab_set.add(word)
vocab = list(vocab_set)
# MORA DA STOJI !!! JER SE UVEK DRUGACIJE UBACUJE U SET
vocab.sort()

# Ako je broj razlicitih reci vec od 10000 zelimo da ih svedemo na najboljih 10000
if len(vocab) > 10000:
    list_of_all_words = []
    # Ubacujemo u listu tuplova (rec, broj_pojavljivanja)
    for word in vocab:
        key = dictonary.get(word, 0)
        list_of_all_words.append((word, key))
    # Sortiramo po broju pojavljivanja
    list_of_all_words.sort(key=lambda x: x[1], reverse=True)
    out_list = []
    # Zelimo da uzmemo samo one reci koje se pojavljuju makar 10 puta u svim twitovima
    for i in range(10000):
        if list_of_all_words[i][1] >= 10:
            out_list.append(list_of_all_words[i][0])
    vocab = out_list


print('Feature vector size: ', len(vocab))
model = MultinomialNaiveBayes(nb_classes=2, nb_words=len(vocab), pseudocount=1)

# Random indeksi da ne bi morali data da mesamo
random_index = create_random_indexes(max)

# Bag of Words model
print('Creating BOW features...')
feature_vector_size = len(vocab)

# Mapiranje svake reci na njen indeks u vokabularu
vocab_dic = dict()
for i in range(feature_vector_size):
    vocab_dic[vocab[i]] = i

# Uzimamo i pravo feature vektore od 80% podataka za treniranje
for i in range(int(max*0.8)):
    # Uzimamo jedan twit iz corpusa
    doc_idx = random_index[i]
    doc = clean_corpus[doc_idx]

    doc_set = set()
    for word in doc:
        doc_set.add(word)

    # Umesto da pravimo feature vector velicine vokabulara pravimo samo niz tuplova
    # gde je prva stvar indeks na kom se nalazi u vokabularu a druga stvar je koliko
    # puta se nalazi u tom twitu
    # prakticno kompresija niza [0,0,0,1,0,0,5,0,0,0,2] na [(3,1),(6,5),(10,2)]
    # posto tvitovi generalno nece imati previse reci ovo je ogromna optimizacija jer
    # ne moramo vise da imamo za svaki feature vector po 10000 elemenata vec mozemo da
    # imamo po 10-ak po twitu
    new_feature_vector = []
    for word in doc_set:
        number_of_occ = numocc_score(word, doc)
        new_feature_vector.append((vocab_dic.get(word, -1), number_of_occ))

    # Modelu saljemo feature vector i da li je twit bio pozitivan ili ne to je deo data['y'][doc_idx]
    model.add_feature_vector_tmp(new_feature_vector, data['y'][doc_idx])

# Kada zavrsimo sa ubacivanjem svih feature vectora fitujemo
model.fit()


brojTacnih = 0
class_names = ['Negative', 'Positive']
confusion_matrix = []
true_negatives = 0
true_positives = 0
false_negative = 0
false_positives = 0
print('Checking for test set...')
# Uzimamo i pravo feature vektore od 20% podataka za testiranje
for i in range(int(max*0.8), max):
    # Ista stvar kao i gore
    doc_idx = random_index[i]
    doc = clean_corpus[doc_idx]

    doc_set = set()
    for word in doc:
        doc_set.add(word)

    new_feature_vector = []

    for word in doc_set:
        number_of_occ = numocc_score(word, doc)
        new_feature_vector.append((vocab_dic.get(word, -1), number_of_occ))

    # Predictujemo da li je postivan ili negativan twit
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

print('Confusion matrix [[TN,FP],[FN,TP]]\n', confusion_matrix)
print('Proecenat tacnosti : ', (true_positives+true_negatives) / (true_positives+true_negatives+false_negative+false_positives) * 100)

# negative = [('i', 15900.0), ('the', 8377.0), ('a', 6364.0), ('you', 6064.0), ('but', 3971.0)]
# positive = [('i', 13712.0), ('you', 11874.0), ('the', 11238.0), ('a', 9189.0), ('for', 5839.0)]
print(model.best_tweets(vocab, amount=5))
# negative = [('sad', 0.067), ('sadli', 0.075), ('por', 0.125), ('upset', 0.137), ('depres', 0.14)]
# positive = [('folowfriday', 20.43), ('vip', 13.083), ('welcom', 13.0), ('recomend', 10.54), ('congrat', 8.64)]
print(model.best_lr_tweets(vocab, amount=5))

# LR metrika kaze da ako je dobijeni broj preko 1 to znaci da se ta rec vise pojavljuje u pozitivnim tvitovima
# a ako je broj manji od 1 to znaci da se vise pojavljuje u negativnim sto je broj veci ili manji to je drasticnija razlika
# za LR == 20, to znaci da se rec pojavlju 20 puta vise u pozitivnim tvitovima nego u negativnim
# za LR == 0.067 (1/0.067) to znaci da se rec pojavljuje skoro 15 puta vise u negativnim nego pozitivnim

# Ovih 10 dobijenih reci iz LR-a ne moraju nuzno da budu i u prvom skupu, one samo oznacavaju koliko puta se vise pojavljuju
# u nekoj klasi, ali i dalje postoje reci koje se pojavljuju u obe klase mnogo puta i koje imaju LR metriku obicno izmedju 0.5 - 1.5

print('Done.')
