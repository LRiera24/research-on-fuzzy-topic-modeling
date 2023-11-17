import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import numpy as np
import os
from gensim.models import KeyedVectors

nltk.download('punkt')
nltk.download('wordnet')

# Define path to embeddings model
model_path = os.path.abspath('src')
model_path += '/word2vec/GoogleNews-vectors-negative300.bin'
print(model_path)

# Load pre-trained word embeddings (Word2Vec model)
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

def lesk_embedding(word, context, model, synsets=None):
    if synsets is None:
        synsets = wordnet.synsets(word)

    if not synsets:
        return None

    context_embedding = np.mean([model[word] for word in context if word in model], axis=0)

    best_synset = None
    max_similarity = float('-inf')

    for synset in synsets:
        definition_embedding = np.mean([model[word] for word in word_tokenize(synset.definition()) if word in model], axis=0)
        similarity = np.dot(context_embedding, definition_embedding) / (np.linalg.norm(context_embedding) * np.linalg.norm(definition_embedding))

        if similarity > max_similarity:
            max_similarity = similarity
            best_synset = synset
            print(best_synset.definition(), similarity)

    return best_synset

def hyper(s): return s.hypernyms()

# Example Usage
context = ['egg', 'sugar', 'butter', 'flour', 'recipe', 'cake', 'dessert']

for word in context:
    print(word)
    synsets = wordnet.synsets(word)
    best_synset = lesk_embedding(word, context, word2vec_model, synsets=synsets)

    if best_synset:
        print("Selected Synset:", best_synset.name())
        print("Definition:", best_synset.definition())
        print([w.lemma_names() for w in list(best_synset.closure(hyper))])
    else:
        print("No suitable synset found.")

    print()


