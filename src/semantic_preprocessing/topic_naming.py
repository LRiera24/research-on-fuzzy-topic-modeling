from nltk.corpus import wordnet
import numpy as np
from semantic_preprocessing.word_sense_desambiguation.genetic_algorithm import genetic_algorithm
from semantic_preprocessing.word_sense_desambiguation.simplex import simplex_sol
from semantic_preprocessing.word_sense_desambiguation.lesk_modified import lesk_embedding

class TopicNaming:
    """
    A class for naming topics identified in a topic model.

    This class uses word embeddings, wordnet synsets, and algorithms like simplex, genetic algorithm,
    and a modified Lesk algorithm to assign meaningful names to the topics based on the most relevant
    words in each topic.

    Attributes:
        topic_model: A trained topic model object.
        information_content_corpus (dict): A dictionary mapping synsets to their information content.
        embeddings_model: A word embeddings model.
        domains (list): List to store domain names for each topic.
    """

    def __init__(self, topic_model, information_content_corpus, embeddings_model):
        """
        Initializes the TopicNaming class with a topic model, information content corpus, and embeddings model.

        Args:
            topic_model: A trained topic model object.
            information_content_corpus (dict): A dictionary mapping synsets to their information content.
            embeddings_model: A word embeddings model.
        """
        self.topic_model = topic_model
        self.information_content_corpus = information_content_corpus
        self.embeddings_model = embeddings_model
        self.domains = []

    def tag_topics(self):
        """
        Assigns a name to each topic based on the lowest common domain among its top words.
        """
        for topic_num in range(self.topic_model.num_topics):
            top_words = self.get_top_words(topic_num, 3)
            synsets = self.get_definitions_for_context(top_words)
            self.domains.append((topic_num, self.calculate_max_weighted_score(synsets)))

    def get_top_words(self, topic_num, k):
        """
        Retrieves the most probable words for a specified topic.

        Args:
            topic_num (int): The topic number.
            k (int): The number of top words to retrieve.

        Returns:
            list: A list of the top k words for the specified topic.
        """
        topic_words = self.topic_model.show_topic(topic_num, topn=k)
        topic_words = sorted(topic_words, key=lambda x: x[1], reverse=True)
        top_words = [word for word, prob in topic_words[:k]]
        print(f"Top words for topic {topic_num}: {top_words}")
        return top_words
    # def get_top_words(self, topic_num, k):
    #     """
    #     Retrieves the most probable words for a specified topic, filtered by a significant drop in probabilities.

    #     Args:
    #         topic_num (int): The topic number.
    #         k (int): The initial number of top words to retrieve.

    #     Returns:
    #         list: A list of the most relevant words for the specified topic.
    #     """
    #     # Retrieve the top k words with their probabilities
    #     topic_words = self.topic_model.show_topic(topic_num, topn=k)
    #     topic_words = sorted(topic_words, key=lambda x: x[1], reverse=True)

    #     # Calculate the rate of change in probability
    #     probability_changes = [topic_words[i][1] - topic_words[i + 1][1] 
    #                            for i in range(len(topic_words) - 1)]

    #     # Calculate the average and standard deviation of these changes
    #     average_change = sum(probability_changes) / len(probability_changes)
    #     std_deviation = (sum([(x - average_change) ** 2 for x in probability_changes]) / len(probability_changes)) ** 0.5

    #     # Detect the point where the change is significantly higher than the average
    #     significant_drop_index = next((i for i, change in enumerate(probability_changes)
    #                                   if change > average_change + std_deviation), len(topic_words) - 1)

    #     # Filter words up to the point of significant drop
    #     filtered_words = [word for word, _ in topic_words[:significant_drop_index + 1]]

    #     print(f"Top words for topic {topic_num}: {filtered_words}")
    #     return filtered_words

    def get_definitions_for_context(self, context, algorithm='lesk', sim_measure='path'):
        """
        Retrieves synsets for given words using the specified algorithm.

        Args:
            context (list): A list of words.
            algorithm (str): The algorithm to use ('simplex', 'genetic', or 'lesk').
            sim_measure (str): The similarity measure to use (currently unused).

        Returns:
            list: A list of chosen synsets for the given context.
        """
        synsets = [wordnet.synsets(word) for word in context]
        chosen_synsets = []
        if algorithm == 'simplex':
            chosen_synsets = simplex_sol(synsets)
        elif algorithm == 'genetic':
            chosen_synsets = genetic_algorithm(synsets)
        elif algorithm == 'lesk':
            for i, word in enumerate(context):
                best_synset = lesk_embedding(word, context, self.embeddings_model, synsets[i])
                if best_synset:
                    chosen_synsets.append(best_synset)
        return chosen_synsets

    def calculate_max_weighted_score(self, context_synsets):
        """
        Calculates the maximum weighted score for given synsets based on various criteria.

        Args:
            context_synsets (list): A list of synsets.

        Returns:
            tuple: A tuple containing the list of candidate tags and the maximum value.
        """
        candidate_tags = {}
        MAX_OCCURRENCE = 0
        hyper = lambda s: s.hypernyms()
        for s in context_synsets:
            for w in list(s.closure(hyper)):
                if w.min_depth() >= 4:
                    candidate_tags[w] = candidate_tags.get(w, 0) + 1
                    MAX_OCCURRENCE = max(MAX_OCCURRENCE, candidate_tags[w])

        combined_weights = []
        for synset in candidate_tags.keys():
            information_content = self.information_content_corpus.get(synset.name(), 0.0)
            specificity_weight = 1 / (1 + synset.min_depth())
            semantic_relevance_weight = self.calculate_semantic_relevance(synset, context_synsets)
            frequency_weight = candidate_tags[synset]
            combined_weights.append(
                np.mean([information_content, semantic_relevance_weight, frequency_weight]))

        max_value = max(combined_weights)
        max_indices = [i for i, x in enumerate(combined_weights) if max_value - x <= 0.01]

        return [s for index, s in enumerate(candidate_tags.keys()) if index in max_indices], round(max_value, 2)

    def calculate_semantic_relevance(self, synset, context_synsets):
        """
        Calculates the semantic relevance of a synset to the context synsets.

        Args:
            synset: The synset to evaluate.
            context_synsets (list): A list of context synsets.

        Returns:
            float: The calculated semantic relevance.
        """
        semantic_relevance = 0
        for s in context_synsets:
            s1, s2 = s.lemma_names()[0].split('_')[0], synset.lemma_names()[0].split('_')[0]
            try:
                relevance = np.dot(self.embeddings_model[s1], self.embeddings_model[s2]) / (
                    np.linalg.norm(self.embeddings_model[s1]) * np.linalg.norm(self.embeddings_model[s2]))
                semantic_relevance += relevance
            except KeyError:
                pass
        return semantic_relevance / len(context_synsets) if context_synsets else 0
