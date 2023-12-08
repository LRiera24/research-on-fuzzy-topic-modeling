from nltk.corpus import wordnet
import numpy as np
from backtrack import backtrack_solution
from genetic_algorithm import genetic_algorithm
from simplex import simplex_sol
from lesk_modified import lesk_embedding

class TopicNaming:
    def __init__(self, topic_model, information_content_corpus, embeddings_model):
        self.topic_model = topic_model
        self.information_content_corpus = information_content_corpus
        self.embeddings_model = embeddings_model
        self.domains = []

    def tag_topics(self):
        # Assign a name to the topic based on the lowest common domain
        for topic_num in range(self.topic_model.num_topics):
            top_words = self.get_top_words(topic_num, 10)
            synsets = self.get_definitions_for_context(top_words)
            self.domains.append(self.calculate_max_weighted_score(synsets, self.information_content_corpus, self.embeddings_model))

    def get_top_words(self, topic_num, k):
        # Get the most probable words for the specified topic
        topic_words = self.topic_model.show_topic(topic_num, topn=k)

        # Sort the words by their probabilities in descending order
        topic_words = sorted(topic_words, key=lambda x: x[1], reverse=True)

        # Get the top K words for the topic
        top_words = [word for word, prob in topic_words[:k]]

        print(f"Top {k} words for topic {topic_num}: {top_words}")

        return top_words

    def get_definitions_for_context(self, context, algorithm='simplex', sim_measure='path'):
        synsets = [wordnet.synsets(word) for word in context]
        chosen_synsets = []
        if algorithm == 'simplex':
            chosen_synsets = simplex_sol(synsets)
        if algorithm == 'genetic':
            chosen_synsets = genetic_algorithm(synsets)
        if algorithm == 'lesk':
            chosen_synsets = lesk_embedding(synsets)
        else:
            chosen_synsets = backtrack_solution(synsets)
        return chosen_synsets

    def calculate_max_weighted_score(self, context_synsets, information_content_corpus, embeddings_model):
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
            semantic_relevance_weight = self.calculate_semantic_relevance(
                embeddings_model, synset, context_synsets)

            # Frequency of occurrence in the closures of the original synsets
            frequency_weight = candidate_tags[synset]

            # Combine weights (adjust weights and formula as needed)
            combined_weights.append(
                np.mean([information_content, semantic_relevance_weight, frequency_weight]))

        max_value = max(combined_weights)
        max_indices = [i for i, x in enumerate(combined_weights) if x == max_value]

        return [s for index, s in enumerate(candidate_tags.keys()) if index in max_indices], max_value


    def calculate_semantic_relevance(self, embeddings_model, synset, context_synsets):
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