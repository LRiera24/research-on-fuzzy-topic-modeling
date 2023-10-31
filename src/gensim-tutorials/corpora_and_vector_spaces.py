from smart_open import open  # for transparently opening remote files
from gensim import corpora
from collections import defaultdict
from pprint import pprint  # pretty-printer
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# stopwords list
stoplist = set('for a of the and to in'.split())

# construct the dictionary without loading all texts into memory
# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open(
    'https://radimrehurek.com/mycorpus.txt'))
# remove stop words and words that appear only once
stop_ids = [
    dictionary.token2id[stopword]
    for stopword in stoplist
    if stopword in dictionary.token2id
]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items()
            if docfreq == 1]
# remove stop words and words that appear only once
dictionary.filter_tokens(stop_ids + once_ids)
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)

# accesses corpus saved in disk and allows us to iterate througout its documents one at a time


class MyCorpus:
    def __iter__(self):
        for line in open('https://radimrehurek.com/mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())


# initialize MyCorpus object
corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
print(corpus_memory_friendly)

# get docs in corpus efficiently
for vector in corpus_memory_friendly:  # load one vector into memory at a time
    print(vector)
print()

# save corpus in different formats
corpus = [[(1, 0.5)], []]  # make one document empty, for the heck of it

corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)
corpora.SvmLightCorpus.serialize('/tmp/corpus.svmlight', corpus)
corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)
corpora.LowCorpus.serialize('/tmp/corpus.low', corpus)
print()


# load a corpus iterator from a Matrix Market file:
corpus = corpora.MmCorpus('/tmp/corpus.mm')

# does not work
print(corpus)
print()

# one way of printing a corpus: load it entirely into memory
# calling list() will convert any sequence to a plain Python list
print(list(corpus))
print()

# another way of doing it: print one document at a time, making use of the streaming interface
for doc in corpus:
    print(doc)
print()

# save the same Matrix Market document stream in Blei's LDA-C format
corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)
