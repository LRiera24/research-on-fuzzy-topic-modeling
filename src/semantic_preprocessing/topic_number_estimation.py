from gensim.models import KeyedVectors
import numpy as np
from auto_incremental_clustering import AutoIncrementalClustering
import os

# Define path to embeddings model
model_path = os.path.abspath('src')
model_path += '/word2vec/GoogleNews-vectors-negative300.bin'

# Define constants
ELEMENTS = 1  # Used as an index

# Class for estimating the number of topics using word embeddings
class TopicNumberEstimation:
    def __init__(self, vocabulary, model_path):
        self.vocabulary = vocabulary
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.embedding_dim = self.model.vector_size
        self.word_embeddings = []

    # Estimate the number of topics using a clustering algorithm
    def estimate_topic_number(self, min_words_per_topic=20):
        clustering = AutoIncrementalClustering(self.word_embeddings, self.model)
        clustering.clustering()
        
        k = 0
        # Count clusters with a minimum number of words
        k += sum(1 for cluster_num in len(clustering.clusters) 
                if len(clustering.clusters[cluster_num][ELEMENTS]) >= min_words_per_topic)

        return k
        
    # Retrieve word embeddings for the given vocabulary
    def _get_word_embeddings(self):
        for word in self.vocabulary:
            if word in self.model:
                vector = self.model[word]
                print(f"Word embedding for '{word}':\n{vector}")
            else:
                vector = np.zeros(self.embedding_dim)
            self.word_embeddings.append(vector)


estimator = TopicNumberEstimation(["hello", "hola"], model_path)
estimator._get_word_embeddings()