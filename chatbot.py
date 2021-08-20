import json
import random
import pickle
import numpy as np
# import nltk
# from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
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

# lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

model = load_model('chatbot_model2.h5')

def vectorizer(sentence):
    # sentence_words = nltk.word_tokenize(sentence)
    sentence_words = preprocessing(sentence)

    input_row = []
    for w in words:
        input_row.append(1) if w in sentence_words else input_row.append(0)
    return np.array(input_row)

def predict_answer(sentence):
    input_row = vectorizer(sentence)
    output = model.predict(np.array([input_row]))[0]
    ET = 0.2
    results = [[i, o] for i, o in enumerate(output) if o > ET]

    results.sort(key=lambda x: x[1], reverse=True)

    intent_list = []

    for r in results:
        intent_list.append({'intent': classes[r[0]], 'prob': str(r[1])})
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

while True:
    message = input("")
    print(get_chat_response(message))