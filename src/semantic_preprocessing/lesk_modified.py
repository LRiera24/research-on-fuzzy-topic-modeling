import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
from gensim.models import KeyedVectors
from math import inf 

# nltk.download('punkt')
# nltk.download('wordnet')

# # Define path to embeddings model
# model_path = os.path.abspath('src')
# model_path += '/word2vec/GoogleNews-vectors-negative300.bin'
# print(model_path)

# Load pre-trained word embeddings (Word2Vec model)
# word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

def lesk_embedding(word, context, model, synsets=None):
    if synsets is None:
        synsets = wordnet.synsets(word)

    if not synsets:
        return None

    # Extract important terms using TF-IDF
    important_terms = extract_important_terms([synset.definition() for synset in synsets], top_n=5)
    print(important_terms)

    context_embedding = np.mean([model[word] for word in context if word in model], axis=0)

    best_synset = None
    max_similarity = float('-inf')

    for synset in synsets:
        if synset.pos() != 'v':
            definition_embedding = np.mean([model[word] for word in word_tokenize(synset.definition()) if word in model and word in important_terms], axis=0)
            
            if definition_embedding is not None:
                similarity = np.dot(context_embedding, definition_embedding) / (np.linalg.norm(context_embedding) * np.linalg.norm(definition_embedding))

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_synset = synset
                    print(best_synset.definition(), similarity)

    return best_synset

def extract_important_terms(definitions, top_n=5):
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the definitions
    tfidf_matrix = vectorizer.fit_transform(definitions)

    # Get feature names (terms)
    terms = vectorizer.get_feature_names_out()

    # Get TF-IDF scores for each term
    tfidf_scores = tfidf_matrix.sum(axis=0).A1

    # Create a list of (term, TF-IDF score) tuples
    term_scores = list(zip(terms, tfidf_scores))

    # Sort terms by TF-IDF score in descending order
    sorted_terms = sorted(term_scores, key=lambda x: x[1], reverse=True)

    # Select the top N terms
    important_terms = set([term for term, _ in sorted_terms])

    return important_terms

def hyper(s): return s.hypernyms()

# Example Usage
# context = ['dog', 'cat', 'rabbit']
# context = ['egg', 'sugar', 'butter', 'flour', 'recipe', 'cake', 'dessert']
# context = ['birthday', 'party', 'gift', 'music', 'candles', 'wish']
# context = ['computer', 'program', 'development', 'application', 'web', 'data']
context = ['school', 'student', 'book', 'teacher']
chosen_defs = []

# lemmas = {}
# lemmas['MAX_OCCURRENCE'] = 0 

# for word in context:
#     print(word)
#     synsets = wordnet.synsets(word)
#     best_synset = lesk_embedding(word, context, word2vec_model, synsets=synsets)

#     if best_synset:
#         print("Selected Synset:", best_synset.name())
#         print("Definition:", best_synset.definition())
#         print("Attributes:", best_synset.entailments())
#         print('Lemmas', [w.lemma_names() for w in list(best_synset.closure(hyper)) if w.min_depth() > 5])
#         # sum = 0
#         # c = 0
#         # for w in list(best_synset.closure(hyper)): 
#         #     d = w.min_depth()
#         #     if d > 3:
#         #         sum += w.min_depth()
#         #         c += 1
#         # mean = round(sum / c)
#         # print('mean', mean)
#         # print(sum/c)
        
#         chosen_defs.append(best_synset)
#     else:
#         print("No suitable synset found.")

#     print()

# print(chosen_defs)
# # min = inf
# # for cd in chosen_defs:
# #     d = cd.min_depth()
# #     if d < min:
# #         min = d
# #     print(d)
# # print('min', min)

# sum = 0
# c = 0
# # for cd in chosen_defs:
# #     d = cd.min_depth()
# #     sum += d
# #     c += 1
# #     print(d)
# # mean = sum / c
# # print('mean', mean)

# for s in chosen_defs:
#     for w in list(s.closure(hyper)): 
#                     # print(w)
#                     d = w.min_depth()
#                     # print(d, mean)
#                     if 3 < d:
#                         lemmas[w] = lemmas.get(w, 0) + 1 
#                         if lemmas['MAX_OCCURRENCE'] < lemmas[w]:
#                             lemmas['MAX_OCCURRENCE'] = lemmas[w]

# print(lemmas)
# print('CHOSEN TAGS')
# for l in lemmas.keys():
#     try:    
#         if lemmas[l] == lemmas['MAX_OCCURRENCE']:
#             d = l.min_depth()
#             sum += d
#             c += 1
#             print(l, d)
#     except:
#         pass

# mean = round(sum / c)
# print('mean', mean)