from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Load a pre-trained Word2Vec model
model = Word2Vec.load("your_word2vec_model.model")

# Choose a set of words for visualization
word_set = ["apple", "banana", "cherry", "dog", "cat", "elephant"]

# Get the Word2Vec vectors for the selected words
vectors = [model.wv[word] for word in word_set]

# Perform PCA to reduce dimensionality to 2D
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

# Create a scatter plot for the word vectors in 2D space
plt.figure(figsize=(8, 8))
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(word_set):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()
