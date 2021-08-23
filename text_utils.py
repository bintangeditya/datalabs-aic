import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
import numpy as np

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
    print("-----text_utils.py:preprocessing---------------------------------")
    print("sebelum : ", word_tokenize(content))

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

    print("sesudah : ", content_clean)
    print("-----------------------------------------------------------------")
    return content_clean


def vectorizer(sentence,words):
    # sentence_words = nltk.word_tokenize(sentence)
    sentence_words = preprocessing(sentence)

    input_row = []
    for w in words:
        input_row.append(1) if w in sentence_words else input_row.append(0)
    return np.array(input_row)



#snippet codingan-----------------------------------
# def tokenize(sentence):
#     return nltk.word_tokenize(sentence)
#
#
# def stemming(word):
#     factory = StemmerFactory()
#     stemmer = factory.create_stemmer()
#     return stemmer.stem(word)
#
#
# def bag_of_words(tokenize_sentence, all_words):
#     tokenize_sentence = [stemming(w) for w in tokenize_sentence]
#     bag = np.zeros(len(all_words), dtype=np.float32)
#     for idx, w in enumerate(all_words):
#         if w in tokenize_sentence:
#             bag[idx] = 1.0
#
#     return bag
