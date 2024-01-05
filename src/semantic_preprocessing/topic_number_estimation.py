from gensim.models import KeyedVectors
import numpy as np
from nltk import pos_tag
from semantic_preprocessing.auto_incremental_clustering import AutoIncrementalClustering

class TopicNumberEstimation:
    """
    Class for estimating the number of topics in a dataset using word embeddings.

    Attributes:
        vocabulary (list): A list of words in the dataset.
        embeddings_model (KeyedVectors): A pre-trained word embeddings model.
    """

    ELEMENTS = 1  # Used as an index for cluster elements

    def __init__(self, embeddings_model):
        """
        Initializes the TopicNumberEstimation class with the given vocabulary and embeddings model.

        Args:
            vocabulary (list): A list of words in the dataset.
            embeddings_model (KeyedVectors): A pre-trained word embeddings model.
        """
        self.model = embeddings_model
        self.embedding_dim = self.model.vector_size
        self.word_embeddings = []

    def estimate_topic_number(self, vocabulary, co_occurrence_matrix, min_sim, min_coh, min_words_per_topic):
        """
        Estimates the number of topics using a clustering algorithm.

        Args:
            co_occurrence_matrix (np.array): A matrix representing word co-occurrences.
            min_words_per_topic (int): Minimum number of words to consider a valid topic.

        Returns:
            int: Estimated number of topics.
        """
        tagged = pos_tag(vocabulary)
        nouns_verbs = [word for word, tag in tagged if tag.startswith('NN') and tag != 'NNP' and tag != 'NNPS']

        self._get_word_embeddings(nouns_verbs)

        cluster = AutoIncrementalClustering(nouns_verbs, co_occurrence_matrix, self.word_embeddings, self.model, min_sim, min_coh)
        cluster.clustering()

        clusters = []
        print('Clusters')
        for t in cluster.clusters.values():
            if len(t[2]) >= min_words_per_topic:
                print(t[2])
                clusters.append(t[2])
        print()

        return len(clusters), clusters

    def _get_word_embeddings(self, vocabulary):
        """
        Retrieves word embeddings for the given vocabulary from the embeddings model.
        """
        for word in vocabulary:
            if word in self.model:
                vector = self.model[word]
            else:
                vector = np.zeros(self.embedding_dim)
            self.word_embeddings.append(vector)
