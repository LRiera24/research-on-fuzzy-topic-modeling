import os
from gensim.models import KeyedVectors
from tools.lexical_preprocessing import LexicalPreprocessing
from semantic_preprocessing.topic_number_estimation import TopicNumberEstimation
from semantic_preprocessing.topic_discovery import TopicDiscovery
from semantic_preprocessing.topic_naming import TopicNaming
from nltk.corpus import wordnet_ic


def semantic_classification(corpus, corpus_name, real_k, real_tags):
    # Load pre-trained word embeddings (Word2Vec model)
    model_path = os.path.abspath('src')
    model_path += '/word2vec/GoogleNews-vectors-negative300.bin'
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # Load pre-trained information content corpus
    information_content_corpus = wordnet_ic.ic('ic-brown.dat')

    preprocesser = LexicalPreprocessing()
    preprocesser.preprocess_text(corpus, corpus_name)

    estimator = TopicNumberEstimation(preprocesser.vocabulary, word2vec_model)
    k = estimator.estimate_topic_number(preprocesser.co_occurrence_dict)

    print("Estimated number of topics:", k)

    topic_finder = TopicDiscovery(preprocesser.vector_repr, preprocesser.dictionary, k)
    topic_model = topic_finder.train_lda()
    topics = topic_finder.get_topics(topic_model)
    print(topics)

    tagger = TopicNaming(topic_model, information_content_corpus, word2vec_model)
    tagger.tag_topics()
    print("Tags:", tagger.domains)

