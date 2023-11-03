import hashlib
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import hmean
from statistics import mean
import itertools

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
            similarities = self.calculate_similarities(w_embedding)
            coherences = self.evaluate_quality(w_embedding)
            self.assignment(w_embedding, similarities, coherences)

    def calculate_similarities(self, word_embedding):
        # Calculate similarities between the word embedding and existing cluster centroids.
        similarities = []

        for cluster_data in self.clusters.values():
            cluster_centroid = cluster_data[CENTROID]
            similarity = cosine_similarity(word_embedding, cluster_centroid)
            similarities.append((cluster_data, similarity))

        # Calculate harmonic mean of similarities and filter clusters based on a threshold.
        harmonic_mean = hmean([sim for _, sim in similarities])
        similarities = [(cluster_num, similarity) for cluster_num,
                        similarity in similarities if similarity >= harmonic_mean]

        return similarities

    def evaluate_quality(self, word_embedding):
        # Evaluate the quality of clusters based on word embedding coherences within the cluster.
        coherences = []

        for cluster_data in self.clusters.values():
            cluster_elements = cluster_data[ELEMENTS]

            # Calculate cosine similarities for pairwise combinations of words in the cluster.
            pairwise_sims = []
            for pair in itertools.combinations(cluster_elements + [word_embedding], 2):
                pair = sorted(pair)
                key = hashlib.md5(f"{pair[0]}-{pair[1]}".encode()).hexdigest()
                if key in self.pairwise_similarities:
                    similarity = self.pairwise_similarities[key]
                else:
                    similarity = self.model.similarity(pair[0], pair[1])
                    self.pairwise_similarities[key] = similarity
                pairwise_sims.append(similarity)

            # Calculate coherence for the cluster by averaging the pairwise similarities.
            coherence = mean(pairwise_sims)
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

    def assignment(self, word_embedding, similarities, coherences):
        # Assign the word embedding to an existing or new cluster based on similarities and coherences.
        common_clusters = set(index1 for index1, _ in similarities) & set(
            index2 for index2, _ in coherences)

        if len(self.clusters) == 0 or len(common_clusters) == 0:
            # If no clusters exist or there are no common clusters, create a new cluster.
            self.clusters[len(self.clusters)] = [
                word_embedding, [word_embedding]]
        else:
            for cluster_num in common_clusters:
                # Update an existing cluster with the new word embedding.
                updated_elements = self.clusters[cluster_num][ELEMENTS].append(
                    word_embedding)
                new_centroid = mean(updated_elements)
                self.clusters[cluster_num][CENTROID] = new_centroid
                self.clusters[cluster_num][ELEMENTS] = updated_elements
