from math import inf
import numpy as np
from scipy.stats import entropy
from scipy.stats import hmean
from scipy.stats import mean

centroid = 0
elements = 1


class auto_incremental_clustering:
    def __init__(self, word_embeddings, max_entropy=0.5):
        self.word_embeddings = word_embeddings
        self.max_entropy = max_entropy
        self.clusters = {}

    def clustering(self):
        for w_embedding in self.word_embeddings:
            distances = self.calculate_distance(w_embedding)
            entropies = self.evaluate_quality(w_embedding)

            self.assignment(w_embedding, distances, entropies)

    def calculate_distance(self, word_embedding):
        centroids = [c for c, cluster in self.clusters.values()]
        euclidean_distances = [np.linalg.norm(
            word_embedding - centroid) for centroid in centroids]

        harmonic_mean = hmean(euclidean_distances)
        euclidean_distances = [(index, distance) for index, distance in enumerate(
            euclidean_distances) if distance <= harmonic_mean]

        return euclidean_distances

    def evaluate_quality(self, word_embeddig):
        entropies = []
        for i in range(self.clusters):
            test = self.clusters[i][elements].append(word_embeddig)
            entropies.append(entropy(test))

        entropies = [(index, entropy)
                     for index, entropy in entropies if entropy <= self.max_entropy]

        return entropies

    def assignment(self, word_embedding, distances, entropies):
        common_indices = set(index1 for index1, _ in distances) & set(
            index2 for index2, _ in entropies)
        if len(self.clusters) == 0 or len(common_indices) == 0:
            self.clusters[len(self.clusters)] = [
                word_embedding, [word_embedding]]
        else:
            for index in common_indices:
                updated_elements = self.clusters[index][elements].append(
                    word_embedding)
                new_centroid = mean(updated_elements)
                self.clusters[index][centroid] = new_centroid
                self.clusters[index][elements] = updated_elements
