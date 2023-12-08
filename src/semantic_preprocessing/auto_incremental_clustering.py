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

class AutoIncrementalClustering:
    def __init__(self, word_embeddings, model, min_coherence=0.5):
        # Initialize the class with provided word embeddings, a model, and a minimum coherence threshold.
        self.word_embeddings = word_embeddings
        self.min_coherence = min_coherence
        self.model = model
        self.clusters = {}
        self.pairwise_similarities = {}

    def clustering(self):
        # Perform clustering on the provided word embeddings.
        for w_embedding in self.word_embeddings:
            if len(self.clusters) == 0:
                self._assignment(w_embedding, None, None)
                print("FIRST ELEMENT ADDED")
                continue
            similarities = self._calculate_similarities(w_embedding)
            print("!!!!!!!!!!!!!!! SIMS DONE")
            coherences = self._evaluate_quality(w_embedding, similarities)
            print("!!!!!!!!!!!!!!! QUALITY DONE")
            self._assignment(w_embedding, similarities, coherences)
            print("!!!!!!!!!!!!!!! ASSIGNMENT DONE")

    def _calculate_similarities(self, word_embedding):
        # Calculate similarities between the word embedding and existing cluster centroids.
        similarities = []

        for index, cluster_data in enumerate(self.clusters.values()):
            cluster_centroid = cluster_data[CENTROID]
            similarity = np.dot(word_embedding, cluster_centroid) / (
                np.linalg.norm(word_embedding) * np.linalg.norm(cluster_centroid))
            similarities.append((index, similarity))

        print(similarities)
        # Calculate harmonic mean of similarities and filter clusters based on a threshold.
        harmonic_mean = hmean([round(sim, 6) for _, sim in similarities])
        print(harmonic_mean)
        similarities = [(cluster_num, similarity) for cluster_num,
                        similarity in similarities if similarity >= harmonic_mean]

        return similarities

    def _evaluate_quality(self, word_embedding, sims):
        # Evaluate the quality of clusters based on word embedding coherences within the cluster.
        coherences = []

        pairwise_sims = {}
        
        for index, _ in sims:
            cluster_elements = self.clusters[index][ELEMENTS]

            # Calculate cosine similarities for pairwise combinations of words in the cluster.
            for pair in itertools.combinations(cluster_elements + [word_embedding], 2):
                pair = sorted(pair, key=lambda x: x[0])
                print(pair)
                key = hashlib.md5(f"{pair[0]}-{pair[1]}".encode()).hexdigest()
                if key in pairwise_sims.keys():
                    print('here1')
                    similarity = pairwise_sims[key]
                else:
                    similarity = np.dot(pair[0], pair[1]) / (
                        np.linalg.norm(pair[0]) * np.linalg.norm(pair[1]))
                    print('here2')
                    pairwise_sims[key] = similarity

            # Calculate coherence for the cluster by averaging the pairwise similarities.
            coherence = mean(pairwise_sims.values())
            coherences.append(coherence)

        # Normalize coherences between 0 and 1, and filter clusters based on a coherence threshold.
        min_coherence = max(coherences)
        if min_coherence > 0:
            coherences = [coherence /
                          min_coherence for coherence in coherences]
        else:
            coherences = [0.0] * len(coherences)

        coherences = [(cluster_num, coherence) for cluster_num, coherence in zip(
            self.clusters.keys(), coherences) if coherence >= self.min_coherence]

        return coherences

    def _assignment(self, word_embedding, similarities, coherences):
        # Assign the word embedding to an existing or new cluster based on similarities and coherences.
        common_clusters = set(index1 for index1, _ in similarities) & set(
            index2 for index2, _ in coherences) if similarities and coherences else []

        if len(self.clusters) == 0 or len(common_clusters) == 0:
            print('assignment')
            # If no clusters exist or there are no common clusters, create a new cluster.
            self.clusters[len(self.clusters)] = [
                word_embedding, [word_embedding]]
            print(self.clusters)
        else:
            for cluster_num in common_clusters:
                # Update an existing cluster with the new word embedding.
                updated_elements = self.clusters[cluster_num][ELEMENTS].append(
                    word_embedding)
                new_centroid = mean(updated_elements)
                self.clusters[cluster_num][CENTROID] = new_centroid
                self.clusters[cluster_num][ELEMENTS] = updated_elements


# Load pre-trained word embeddings (Word2Vec model)
model_path = os.path.abspath('src')
model_path += '/word2vec/GoogleNews-vectors-negative300.bin'
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# context = ['dog', 'cat', 'rabbit']
# context = ['egg', 'sugar', 'butter', 'flour', 'recipe', 'cake', 'dessert']
# context = ['birthday', 'party', 'gift', 'music', 'candles', 'wish']
# context = ['computer', 'program', 'development', 'web', 'application', 'data']
context = ['school', 'class', 'homework', 'student', 'book', 'knowledge', 'learn', 'teach', 'dog', 'cat', 'rabbit',
            'egg', 'sugar', 'butter', 'flour', 'recipe', 'cake', 'dessert', 'birthday', 'party', 'gift', 'music', 
            'candles', 'wish', 'computer', 'program', 'development', 'web', 'application', 'data']

embeddings = [word2vec_model[word] for word in context]

cluster = AutoIncrementalClustering(embeddings, word2vec_model)
cluster.clustering()