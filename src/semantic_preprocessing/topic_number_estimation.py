from gensim.models import KeyedVectors
import numpy as np
from semantic_preprocessing.auto_incremental_clustering import AutoIncrementalClustering

class TopicNumberEstimation:
    """
    Class for estimating the number of topics in a dataset using word embeddings.

    Attributes:
        vocabulary (list): A list of words in the dataset.
        embeddings_model (KeyedVectors): A pre-trained word embeddings model.
    """

    ELEMENTS = 1  # Used as an index for cluster elements

    def __init__(self, vocabulary, embeddings_model):
        """
        Initializes the TopicNumberEstimation class with the given vocabulary and embeddings model.

        Args:
            vocabulary (list): A list of words in the dataset.
            embeddings_model (KeyedVectors): A pre-trained word embeddings model.
        """
        self.vocabulary = vocabulary
        self.model = embeddings_model
        self.embedding_dim = self.model.vector_size
        self.word_embeddings = []

    def estimate_topic_number(self, co_occurrence_matrix, min_words_per_topic=20):
        """
        Estimates the number of topics using a clustering algorithm.

        Args:
            co_occurrence_matrix (np.array): A matrix representing word co-occurrences.
            min_words_per_topic (int): Minimum number of words to consider a valid topic.

        Returns:
            int: Estimated number of topics.
        """
        self._get_word_embeddings()

        cluster = AutoIncrementalClustering(self.vocabulary, co_occurrence_matrix, self.word_embeddings, self.model)
        cluster.clustering()

        print('Clusters')
        for t in cluster.clusters.values():
            print(t[2])
        print()

        k = sum(1 for cluster_num in range(len(cluster.clusters)) 
                if len(cluster.clusters[cluster_num][self.ELEMENTS]) >= min_words_per_topic)
        return k

    def _get_word_embeddings(self):
        """
        Retrieves word embeddings for the given vocabulary from the embeddings model.
        """
        for word in self.vocabulary:
            if word in self.model:
                vector = self.model[word]
            else:
                vector = np.zeros(self.embedding_dim)
            self.word_embeddings.append(vector)
