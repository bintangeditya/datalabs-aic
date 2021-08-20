import random
import json
import pickle
import numpy as np
# import nltk
# from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize

stopwords_indonesia = stopwords.words('indonesian')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stopword tambahan
more_stopword = set(["nya", "ya", "kali"])

# kamus bahasa tidak baku
slang_word_df = pd.read_csv(r"colloquial-indonesian-lexicon.csv")
slang_word = {}
for i, row in slang_word_df.iterrows():
    slang_word[row["slang"]] = row["formal"]

# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])

# all emoticons (happy + sad)
emoticons = emoticons_happy.union(emoticons_sad)

# lemmatizer = WordNetLemmatizer()

def preprocessing(content):
    print("sebelum ", word_tokenize(content))

    content = content.strip(" ")  # menghapus karakter spasi pada awal dan akhir kalimat

    symbols_regrex_pattern = re.compile(pattern="["
                                                u"\U0001F600-\U0001F64F"  # emoticons
                                                u"\U0001F300-\U0001F5FF"  # simbol & piktograf
                                                u"\U0001F680-\U0001F6FF"  # simbol transport & map
                                                "]+", flags=re.UNICODE)

    content = symbols_regrex_pattern.sub(r' ', content)  # menghapus emoticon dalam bentuk unicode

    content = content.lower()  # ubah menjadi huruf kecil

    content = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', content, flags=re.MULTILINE)  # menghapus hyperlink

    content = re.sub(r'\n', ' ', content)  # menghapus baris baru

    content = re.sub('[0-9]+', ' ', content)  # menghapus angka

    content = re.sub(r'[?|$.!_:")(+,*&%#@]', ' ', content)  # menghapus beberapa karakter khusus

    content = re.sub(' +', ' ', content)  # lebih dari satu spasi yang berdekatan, menjadi satu spasi

    tokens = word_tokenize(content)  # tokenize kata-kata

    content_clean = []
    for word in tokens:
        if word in slang_word:  # mengganti kata baku dengan yang formal
            word = slang_word[word]

        if (word not in stopwords_indonesia and  # menghilangkan stopwords. proses filtering
                word not in emoticons and  # menghilangkan emoticons. proses filtering
                word not in more_stopword and
                word not in string.punctuation):  # menghilangkan tanda baca. proses filtering
            stem_word = stemmer.stem(word)  # mengubah ke kata dasar. proses stemming
            content_clean.append(stem_word)

    print("sesudah ", content_clean)
    print("-----------------------------------------------------------------")
    return content_clean


intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []

#proses pengisian kamus kata "words", daftar kategori kelas "classes", dataset "documents"
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # word_list = nltk.word_tokenize(pattern)
        word_list = preprocessing(pattern)
        words.extend(word_list)
        documents.append((intent['tag'],word_list))
        classes.append(intent['tag'])

# words = [lemmatizer.lemmatize(w) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

#simpan kamus kata
pickle.dump(words, open('words.pkl','wb'))
#simpan kategori kelas
pickle.dump(classes, open('classes.pkl','wb'))

training_data = []
output_empty = [0] * len(classes)

#vectorize, mengubah kata" menjadi array 1 dimensi
for d in documents:
    input_row = []
    word_pattern = d[1]
    for w in words:
        input_row.append(1) if w in word_pattern else input_row.append(0)

    output_row = list(output_empty)
    output_row[classes.index(d[0])] = 1
    # input_row = np.asarray(input_row).astype(np.float32)
    # output_row = np.asarray(output_row).astype(np.float32)
    training_data.append([input_row,output_row])

random.shuffle(training_data)
training_data = np.array(training_data)

#dataset pada documents -> dijadikan vector -> dataset disesuaikan dengan format masukan Tensorflow
train_X = list(training_data[:, 0])
train_y = list(training_data[:, 1])
train_X = tf.stack(train_X)
train_y = tf.stack(train_y)

#training
model = Sequential([
    Dense(128, input_shape=(len(train_X[0]),), activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(train_y[0]),activation="softmax")
])

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(np.array(train_X), np.array(train_y), epochs=100, batch_size=5, verbose=1)

#simpan model
model.save("chatbot_model2.h5", history)
print("finished training :) yay")



#snippet codingan, sayang dibuang bro-----------------------------------
# def preprocessing_input(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(w) for w in sentence_words]
#     input_row = []
#     for w in words:
#         input_row.append(1) if w in sentence_words else input_row.append(0)
#     return np.array(input_row)
#
# results = model.predict(np.array([preprocessing_input("See you later, thanks for visiting us")]))
# temp = []
# for i, v in enumerate(results[0]):
#     temp.append([v, classes[i]])
# temp = sorted(temp, key=lambda x: x[0], reverse=True)
# print(temp)
# print(classes)


