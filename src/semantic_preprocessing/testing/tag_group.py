from nltk.corpus import wordnet_ic
from nltk.corpus import wordnet
import numpy as np
import os
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from lesk_modified import lesk_embedding


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
        print(synset)
        # Information content
        information_content = information_content_corpus.get(
            synset.name(), 0.0)

        # Specificity or generality
        specificity_weight = synset.min_depth()

        # Semantic relevance to other synsets in the set
        semantic_relevance_weight = calculate_semantic_relevance(
            embeddings_model, synset, context_synsets)

        # Frequency of occurrence in the closures of the original synsets
        frequency_weight = candidate_tags[synset]

        # Combine weights (adjust weights and formula as needed)
        combined_weights.append(
            np.mean([0.2 * information_content, 0.2 * specificity_weight, 0.5 * semantic_relevance_weight, 0.1 * frequency_weight]))

    max_value = max(combined_weights)
    max_indices = [i for i, x in enumerate(combined_weights) if x == max_value]

    return [s for index, s in enumerate(candidate_tags.keys()) if index in max_indices], max_value


def calculate_semantic_relevance(embeddings_model, synset, context_synsets):
    semantic_relevance = 0
    for s in context_synsets:
        s1 = s.lemma_names()[0]
        s2 = synset.lemma_names()[0]
        if '_' in s1:
            s1 = s1.split('_')[0]
        if '_' in s2:
            s2 = s2.split('_')[0]
        try:
            semantic_relevance += np.dot(embeddings_model[s1], embeddings_model[s2]) / (
                np.linalg.norm(embeddings_model[s1]) * np.linalg.norm(embeddings_model[s2]))
        except:
            pass
    semantic_relevance = semantic_relevance / len(context_synsets)
    return semantic_relevance


# Use an appropriate information content corpus
information_content_corpus = wordnet_ic.ic('ic-brown.dat')

model_path = os.path.abspath('src')
model_path += '/word2vec/GoogleNews-vectors-negative300.bin'
embeddings_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# # context = ['dog', 'cat', 'rabbit', 'pig', 'bird']
# # context = ['egg', 'sugar', 'butter', 'flour', 'recipe', 'cake', 'dessert']
# # context = ['birthday', 'party', 'gift', 'music', 'people', 'candles', 'wish']
# # context = ['computer', 'web', 'internet', 'network', 'communication']
# context = ['school', 'class', 'homework', 'student', 'book', 'knowledge', 'learn', 'teach']
context = [
            "game",
            "team",
            "player",
            "playing_period",
            "year",
            "season",
            "win",
            "field_hockey",
            "league",
            "last",
            "goal",
            "first_base",
            "period",
            "sports_fan",
            "shot",
            "second_base",
            "leaf",
            "series",
            "blue",
            "boston",
            "confused",
            "score",
            "deuce",
            "texas_ranger",
            "point",
            "class",
            "third",
            "red",
            "microphone"
        ]

context = [
            "file",
            "window",
            "program",
            "server",
            "application",
            "user",
            "system",
            "translation",
            "font",
            "available",
            "stage_set",
            "directory",
            "message",
            "information",
            "list",
            "entry",
            "name",
            "sun",
            "resource",
            "web_site",
            "node",
            "coach",
            "track",
            "motif",
            "anonymous",
            "instruction"
        ]

# context = ['window', 'program', 'application', 'use', 'available', 'file', 'system', 'user', 'version', 'server', 'code', 'also', 'set', 'graphic', 'get', 'run', 'list', 'mail', 'font', 'support', 'information', 'source', 'send', 'library', 'sun', 'display', 'package', 'computer', 'message', 'include']

context_synsets = []
for word in context:
    print(word)
    synsets = wordnet.synsets(word)
    best_synset = lesk_embedding(
        word, context, embeddings_model, synsets=synsets)
    if best_synset:
        context_synsets.append(best_synset)

tags = calculate_weighted_score(
    context_synsets, information_content_corpus, embeddings_model)
print('CHOSEN TAGS', tags)
