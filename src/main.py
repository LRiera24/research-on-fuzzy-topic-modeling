import os
from tools.lexical_preprocessing import LexicalPreprocessing
from semantic_preprocessing.topic_number_estimation import TopicNumberEstimation
from semantic_preprocessing.topic_discovery import TopicDiscovery
from semantic_preprocessing.topic_naming import TopicNaming


corpus = ['']

model_path = os.path.abspath('src')
model_path += '/word2vec/GoogleNews-vectors-negative300.bin'

preprocesser = LexicalPreprocessing()
preprocesser.preprocess_text(corpus)

estimator = TopicNumberEstimation(preprocesser.vocabulary, model_path)
k = estimator.estimate_topic_number()

topic_model = TopicDiscovery(preprocesser.vector_repr, preprocesser.dictionary, k)
model = topic_model.train_lda()
topics = topic_model.get_topics()

tagger = TopicNaming(model)
# TODO: TAG FUNCTION




