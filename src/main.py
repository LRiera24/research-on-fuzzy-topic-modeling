import os
from gensim.models import KeyedVectors
from tools.lexical_preprocessing import LexicalPreprocessing
from semantic_preprocessing.topic_number_estimation import TopicNumberEstimation
from semantic_preprocessing.topic_discovery import TopicDiscovery
from semantic_preprocessing.topic_naming import TopicNaming
from nltk.corpus import wordnet_ic
from sklearn.datasets import fetch_20newsgroups
from semantic_classsification import semantic_classification

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Preprocess the text data
corpus = newsgroups.data

# corpus = ['dog, cat, rabbit',
#             'egg, sugar, butter, flour, recipe, cake, dessert',
#             'birthday, party, gift, music, candles, wish',
#             'birthday, cake, gift, dance',
#             'dance, cake, music, people, candles, sugar',
#             'computer, program, development, web, application, data',
#             'school, class, homework, student, book, knowledge, learn, teach']

corpus_name = '20newsgroups'

real_k = 20

real_tags = []

s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
c = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for sim in s:
    for coh in c:
        semantic_classification(corpus, corpus_name, real_k, real_tags, min_sim=sim, min_coh=coh)