import pprint       # why not use normal print?

from collections import defaultdict         # why not a normal dictionary?
from gensim import corpora      # gensim module for corpus management
from gensim import models
from gensim import similarities


text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

stoplist = set('for a of the and to in'.split(' '))         # list of stopwords

texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]           # removing stopwords

# count frequency of each word in corpus
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# keep words that appear more than once
processed_corpus = [
    [token for token in text if frequency[token] > 1] for text in texts]
print("PROCESSED CORPUS")
pprint.pprint(processed_corpus)
print()

# assign unique id to each word in processed corpus
# this dictionary defines the vocabulary of all words that our processing knows about
dictionary = corpora.Dictionary(processed_corpus)
print("CREATE CORPORA DICTIONARY")
print(dictionary)
print()

# see corresponding IDs and words
print("ACCESS CORRESPONDING IDs")
pprint.pprint(dictionary.token2id)
print()

# gets the bag of words representation of a given doc according to the words in the model's dictionary
# [Tuple(ID, frequency in doc)]
new_doc = "Human computer interaction minors minors minors"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print("ADD NEW VECTOR REPRESENTATION")
print(new_vec)
print()

# convert corpus to a list of vectors
print("CONVERT CORPUS TO BoW")
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
pprint.pprint(bow_corpus)
print()

# train the model
print("CONVERT CORPUS FROM BoW TO TF-IDF")
tfidf = models.TfidfModel(bow_corpus)
print(tfidf.idfs)
print()

# gets the tf-idf representation of a given doc according to the words in the model's dictionary
# [Tuple(ID, tf-idf weight in doc)]
print("TF-IDF REPR OF DOC ACCORDING TO CORPUS")
words = "system minors".lower().split()
print(tfidf[dictionary.doc2bow(words)])
print()


index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)

# gets a list of the similarities of the query to each document
# Tuple[(Doc index, similarity)]
print("SIMILARITIES OF A QUERY TO EACH DOC IN CORPUS")
query_document = 'system engineering'.split()
query_bow = dictionary.doc2bow(query_document)
sims = index[tfidf[query_bow]]
print(list(enumerate(sims)))
print()

print("SORTED SIMILARITIES")
for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)
print()
