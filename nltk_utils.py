import nltk
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stemming(word):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(word)


def bag_of_words(tokenize_sentence, all_words):
    tokenize_sentence = [stemming(w) for w in tokenize_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenize_sentence:
            bag[idx] = 1.0

    return bag
