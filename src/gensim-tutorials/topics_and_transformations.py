import tempfile
import os
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import models
from gensim import corpora
from collections import defaultdict

documents = [
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

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
print()

# transformation
tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model
print()

# transforming a vector
doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow])  # step 2 -- use the model to transform vectors
print()

# transforming a corpus
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)
print()

# Transformations can also be serialized, one on top of another, in a sort of chain:
# initialize an LSI transformation
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
# create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
corpus_lsi = lsi_model[corpus_tfidf]
print()

for doc in corpus_lsi:
    print(doc)
print()

#
lsi_model.print_topics(2)
print()

# valor de pertenencia a cada uno de los topicos por doc (menor valor -> mayor probabilidad de pertenencia)
for doc, as_text in zip(corpus_lsi, documents):
    print(doc, as_text)

# model persistency
with tempfile.NamedTemporaryFile(prefix='model-', suffix='.lsi', delete=False) as tmp:
    lsi_model.save(tmp.name)  # same for tfidf, lda, ...

loaded_lsi_model = models.LsiModel.load(tmp.name)

os.unlink(tmp.name)
