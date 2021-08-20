import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

model = load_model('chatbot_model.h5')

def preprocessing_input(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w) for w in sentence_words]
    input_row = []
    for w in words:
        input_row.append(1) if w in sentence_words else input_row.append(0)
    return np.array(input_row)

def predict_answer(sentence):
    input_row = preprocessing_input(sentence)
    output = model.predict(np.array([input_row]))[0]
    ET = 0.2
    results = [[i, o] for i, o in enumerate(output) if o > ET]
    #[1 0 0 0 0 0 0 0]
    results.sort(key=lambda x: x[1], reverse=True)

    intent_list = []

    for r in results:
        intent_list.append({'intent': classes[r[0]], 'prob': str(r[1])})

    # print(results)
    # print(intent_list)
    temp = []
    for i, v in enumerate(results):
        temp.append([v, classes[i]])
    temp = sorted(temp, key=lambda x: x[0], reverse=True)
    # print(temp)

    return intent_list

def get_chat_response(sentence):
    intent_list = predict_answer(sentence)
    tag = intent_list[0]['intent']
    chat_response = ""
    for i in intents['intents']:
        if tag == i['tag']:
            chat_response = random.choice(i['responses'])
            break
    return chat_response



# predict_answer("Saya mau belajar tentang Web Developer")

while True:
    message = input("")
    print(get_chat_response("message"))