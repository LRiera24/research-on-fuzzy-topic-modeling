import os
from gensim.models import KeyedVectors
from tools.lexical_preprocessing import LexicalPreprocessing
from semantic_preprocessing.topic_number_estimation import TopicNumberEstimation
from semantic_preprocessing.topic_discovery import TopicDiscovery
from semantic_preprocessing.topic_naming import TopicNaming
from nltk.corpus import wordnet_ic
from sklearn.datasets import fetch_20newsgroups
from semantic_classsification import semantic_classification
import nltk
from nltk.corpus import brown

# Obtener una lista de identificadores de documentos en el Brown Corpus
document_ids = brown.fileids()

# Crear una lista de documentos, donde cada documento es un string
corpus = [' '.join(brown.words(fileid)) for fileid in document_ids]

corpus_name = 'Brown'

real_k = 15

real_tags = []

s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
c = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

description = None
min_words_per_topic = 1000

test_folder = os.path.abspath('tests')
test_folder += f'/{corpus_name}'

if not os.path.exists(test_folder):
    os.makedirs(test_folder)

print(test_folder)
# Get the count of files in the folder
file_count = len([name for name in os.listdir(test_folder)])
print(file_count)
test_folder += f'/run{file_count+1}_{min_words_per_topic}'

if description:
    description = description.lower()
    description = description.split(' ')
    description = '_'.join(description)
    test_folder += f'_{description}'

os.makedirs(test_folder)

for sim in s:
    for coh in c:
        semantic_classification(corpus, corpus_name, real_k, real_tags, test_folder, min_sim=sim, min_coh=coh, min_words_per_topic=min_words_per_topic)