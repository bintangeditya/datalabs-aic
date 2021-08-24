import json
import random
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import text_utils as utils

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')
fallback_intent = "Hmmm... tidak pernah dengar itu sebelumnya."


def _predict_answer(sentence):
    input_row = utils.vectorizer(sentence, words)
    output = model.predict(np.array([input_row]))[0]
    ET = 0.6
    results = [[i, o] for i, o in enumerate(output) if o > ET]

    results.sort(key=lambda x: x[1], reverse=True)

    intent_list = []

    for r in results:
        intent_list.append({'intent': classes[r[0]], 'prob': str(r[1])})
    return intent_list


def get_chat_response(sentence):
    '''
    :rtype: string
    :type sentence : string
    :param sentence: type string, kalimat yang jawabannya akan diprediksi oleh chatbot
    :return: type string, jawaban dari chatbot
    '''
    intent_list = _predict_answer(sentence)
    if len(intent_list) == 0:
        return fallback_intent
    print(intent_list)
    tag = intent_list[0]['intent']
    chat_response = ""
    for i in intents['intents']:
        if tag == i['tag']:
            chat_response = random.choice(i['responses'])
            break
    return chat_response


# untuk testing chatbot
while True:
    message = input("")
    print(get_chat_response(message))
