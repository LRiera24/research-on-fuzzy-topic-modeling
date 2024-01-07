import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import hmean
from statistics import mean
import itertools
from gensim.models import KeyedVectors
import os
import numpy as np

# Constants for dictionary
CENTROID = 0
ELEMENTS = 1
WORDS = 2

class AutoIncrementalClustering:
    """
    Perform auto-incremental clustering on word embeddings.

    This class uses word embeddings to cluster words based on their semantic similarity. 
    Clusters are formed incrementally, and each new word embedding is assigned to an existing 
    or a new cluster based on coherence and similarity measures.

    Attributes:
        words (list): List of words corresponding to the embeddings.
        occurrences (dict): Dictionary containing word co-occurrence information.
        word_embeddings (list): List of word embeddings.
        model (KeyedVectors): Pre-trained word embeddings model.
        min_similarity (float): Minimum similarity threshold for considering cluster assignment.
        clusters (dict): Dictionary to store the clusters.
        pairwise_similarities (dict): Stores pairwise similarities (unused in current implementation).
    """
    def __init__(self, words, occurrences, word_embeddings, model, min_similarity, min_coherence):
        """
        Initializes the AutoIncrementalClustering class.

        Args:
            words (list): List of words corresponding to the embeddings.
            occurrences (dict): Dictionary containing word co-occurrence information.
            word_embeddings (list): List of word embeddings.
            model (KeyedVectors): Pre-trained word embeddings model.
            min_coherence (float): Minimum coherence threshold for cluster formation.
        """
        self.words = words
        self.occurrences = occurrences
        self.word_embeddings = word_embeddings
        self.min_similarity = min_similarity
        self.min_coherence = min_coherence
        self.model = model
        self.clusters = {}

    def clustering(self):
        """
        Perform clustering on the provided word embeddings.

        This method iteratively processes each word embedding, calculates its similarity
        to existing cluster centroids, and assigns it to a cluster based on these similarities
        and coherence values.
        """
        for index, w_embedding in enumerate(self.word_embeddings):
            # print("!!!!!!!!!!!!!", all(w_embedding) != None)
            if len(self.clusters) == 0:
                self._assignment(index, w_embedding, None)
                # print("FIRST ELEMENT ADDED")
                continue
            similarities = self._calculate_similarities(w_embedding)
            # print("!!!!!!!!!!!!!!! SIMS DONE")
            try:
                coherences = self._evaluate_coherence(self.words[index], similarities)
                # print("!!!!!!!!!!!!!!! QUALITY DONE")
                self._assignment(index, w_embedding, coherences)
                # print("!!!!!!!!!!!!!!! ASSIGNMENT DONE")
                # for a in self.clusters.values():
                #     print(a[2])
            except:
                continue

    def _calculate_similarities(self, word_embedding):
        """
        Calculate similarities between the word embedding and existing cluster centroids.

        Args:
            word_embedding (np.array): The word embedding vector to compare against cluster centroids.

        Returns:
            list of tuples: Contains cluster index and similarity score for each cluster.
        """

        similarities = []

        for index, cluster_data in enumerate(self.clusters.values()):
            cluster_centroid = cluster_data[CENTROID]
            # print("c", cluster_centroid)
            similarity = np.dot(word_embedding, cluster_centroid) / (
                np.linalg.norm(word_embedding) * np.linalg.norm(cluster_centroid))
            similarities.append((index, similarity))

        # Calculate harmonic mean of similarities and filter clusters based on a threshold.
        similarities = [(cluster_num, similarity) for cluster_num,
                        similarity in similarities if similarity >= self.min_similarity]

        # print("similarities",similarities)

        return similarities

    def _evaluate_coherence(self, word, sims):
        """
        Evaluate the quality of clusters based on word embedding coherences within the cluster.

        Args:
            word (str): The word whose coherence is being evaluated.
            sims (list of tuples): Contains cluster index and similarity score.

        Returns:
            list of tuples: Contains cluster index and coherence score for each cluster.
        """

        coherences = []
        print(word)

        word_coocurrences = self.occurrences[word]
        
        for index, _ in sims:
            cluster_elements = self.clusters[index][WORDS]
            coherence = 0
            for w in cluster_elements:
                if w in word_coocurrences.keys():
                    coherence += 1

            # Calculate coherence for the cluster by averaging the pairwise similarities.
            coherence = coherence / len(cluster_elements)
            coherences.append(coherence)

        coherences = [(cluster_num, coherence) for cluster_num, coherence in zip(
            self.clusters.keys(), coherences) if coherence >= self.min_coherence]

        # print("coherences", coherences)
        return coherences

    def _assignment(self, index, word_embedding, coherences):
        """
        Assign the word embedding to an existing or new cluster.

        Args:
            index (int): Index of the word in the original list.
            word_embedding (np.array): The word embedding vector to be assigned.
            similarities (list of tuples): Cluster similarities.
            coherences (list of tuples): Cluster coherences.
        """

        indexes = []
        if coherences:
            indexes = [index for index, _ in coherences]

        if len(self.clusters) == 0 or len(indexes) == 0:
            # print('assignment')
            # If no clusters exist or there are no common clusters, create a new cluster.
            self.clusters[len(self.clusters)] = [
                word_embedding, [word_embedding], [self.words[index]]]
        else:
            for cluster_num in indexes:
                # Update an existing cluster with the new word embedding.
                self.clusters[cluster_num][ELEMENTS].append(word_embedding)
                updated_elements = self.clusters[cluster_num][ELEMENTS]
                print(self.clusters[cluster_num][CENTROID])
                print(word_embedding)
                new_centroid = np.mean(updated_elements, axis=0)
                print(new_centroid)
                self.clusters[cluster_num][CENTROID] = new_centroid
                self.clusters[cluster_num][ELEMENTS] = updated_elements
                self.clusters[cluster_num][WORDS].append(self.words[index])
