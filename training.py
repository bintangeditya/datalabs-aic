import random
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import text_utils as utils

stopwords_indonesia = stopwords.words('indonesian')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

intents = json.loads(open("intents.json", encoding="utf8").read())

words = []
classes = []
documents = []

#proses pengisian kamus kata "words", daftar kategori kelas "classes", dataset "documents"
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # word_list = nltk.word_tokenize(pattern)
        word_list = utils.preprocessing(pattern)
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
model.save("chatbot_model.h5", history)
print("finished training :) yay")



#snippet codingan-----------------------------------
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


