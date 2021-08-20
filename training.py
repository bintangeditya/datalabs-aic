import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignore_letters = ['?']

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((intent['tag'],word_list))
        classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training_data = []
output_empty = [0] * len(classes)

for d in documents:
    input_row = []
    word_pattern = d[1]
    word_pattern = [lemmatizer.lemmatize(w) for w in word_pattern if w]
    for w in words:
        input_row.append(1) if w in word_pattern else input_row.append(0)

    output_row = list(output_empty)
    output_row[classes.index(d[0])] = 1
    # input_row = np.asarray(input_row).astype(np.float32)
    # output_row = np.asarray(output_row).astype(np.float32)
    training_data.append([input_row,output_row])

random.shuffle(training_data)
training_data = np.array(training_data)

train_X = list(training_data[:, 0])
train_y = list(training_data[:, 1])
train_X = tf.stack(train_X)
train_y = tf.stack(train_y)

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

model.save("chatbot_model.h5", history)
print("finished training :) yay")

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


