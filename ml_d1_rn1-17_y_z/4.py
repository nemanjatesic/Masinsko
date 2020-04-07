import os
import pandas as pd
import re
import enchant
import numpy as np
import math
import time
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

enchantDict = enchant.Dict("en_US")

porter = PorterStemmer()
dir_path = os.path.dirname(os.path.realpath(__file__))
fileName = dir_path + os.sep + 'data' + os.sep + 'twitter.csv'
data = dict()
data['y'] = pd.read_csv(fileName, sep=',', usecols=[1]).T.values.tolist()[0]
data['x'] = pd.read_csv(fileName, usecols=[2], encoding="ISO-8859-1").T.values.tolist()[0]

corpus = data['x']

print('Cleaning the corpus...')
clean_corpus = []
stop_punc = set(stopwords.words('english')).union(set(punctuation))

# corpus = ['shouldn\'t', 'cao', '#hello']

count = 0


def remove_hashtag(obj):
    return re.sub(r'#([^\s]+)', r'\1', obj)


def too_many_chars(obj):
    return re.sub(r'(.)\1+', r'\1\1', obj)


#f = open('output.txt', 'a')

cnt = 0

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

    #f.write(str(words_final))
    #f.write('\n')

    if cnt == 50: break


#f.close()

print('done')
