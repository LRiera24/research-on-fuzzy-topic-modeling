from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet
import numpy as np
import os
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from word_sense_desambiguation.lesk_modified import lesk_embedding


def calculate_weighted_score(context_synsets, information_content_corpus, embeddings_model):
    candidate_tags = {}
    MAX_OCCURRENCE = 0

    def hyper(s): return s.hypernyms()

    print(context_synsets)
    for s in context_synsets:
        for w in list(s.closure(hyper)):
            if w.min_depth() >= 4:
                candidate_tags[w] = candidate_tags.get(w, 0) + 1
                if MAX_OCCURRENCE < candidate_tags[w]:
                    MAX_OCCURRENCE = candidate_tags[w]

    print(candidate_tags)
    combined_weights = []

    for synset in candidate_tags.keys():
        # Information content
        information_content = information_content_corpus.get(
            synset.name(), 0.0)

        # Specificity or generality
        specificity_weight = 1 / 1 + synset.min_depth()

        # Semantic relevance to other synsets in the set
        semantic_relevance_weight = calculate_semantic_relevance(
            embeddings_model, synset, context_synsets)

        # Frequency of occurrence in the closures of the original synsets
        frequency_weight = candidate_tags[synset]

        # Combine weights (adjust weights and formula as needed)
        combined_weights.append(
            np.mean([information_content, semantic_relevance_weight, frequency_weight]))

    max_value = max(combined_weights)
    max_indices = [i for i, x in enumerate(combined_weights) if x == max_value]

    return [s for index, s in enumerate(candidate_tags.keys()) if index in max_indices], max_value


def calculate_semantic_relevance(embeddings_model, synset, context_synsets):
    semantic_relevance = 0
    for s in context_synsets:
        print(s, synset)
        s1 = s.lemma_names()[0]
        s2 = synset.lemma_names()[0]
        if '_' in s1:
            s1 = s1.split('_')[0]
        if '_' in s2:
            s2 = s2.split('_')[0]
        print(s1, s2)
        try:
            semantic_relevance += np.dot(embeddings_model[s1], embeddings_model[s2]) / (
                np.linalg.norm(embeddings_model[s1]) * np.linalg.norm(embeddings_model[s2]))
        except:
            pass
    semantic_relevance = semantic_relevance / len(context_synsets)
    return semantic_relevance


# # Use an appropriate information content corpus
# information_content_corpus = wordnet_ic.ic('ic-brown.dat')

# model_path = os.path.abspath('src')
# model_path += '/word2vec/GoogleNews-vectors-negative300.bin'
# embeddings_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# # context = ['dog', 'cat', 'rabbit', 'pig', 'bird']
# # context = ['egg', 'sugar', 'butter', 'flour', 'recipe', 'cake', 'dessert']
# # context = ['birthday', 'party', 'gift', 'music', 'people', 'candles', 'wish']
# # context = ['computer', 'web', 'internet', 'network', 'communication']
# context = ['school', 'class', 'homework', 'student', 'book', 'knowledge', 'learn', 'teach']

# context_synsets = []
# for word in context:
#     print(word)
#     synsets = wordnet.synsets(word)
#     best_synset = lesk_embedding(
#         word, context, embeddings_model, synsets=synsets)
#     if best_synset:
#         context_synsets.append(best_synset)

# tags = calculate_weighted_score(
#     context_synsets, information_content_corpus, embeddings_model)
# print('CHOSEN TAGS', tags)
