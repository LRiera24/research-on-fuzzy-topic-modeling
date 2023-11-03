from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic 

class TopicNaming:
    def __init__(self, topic_keywords, domain):
        self.topic_keywords = topic_keywords
        self.domain = domain

    def get_topic_keywords(self, word_topic_distro, min_prob = 0.7):
        # Return the most significant words for the topic
        self.topic_keywords = [word for word in word_topic_distro if word.probability >= min_prob] #TODO: ver como funciona esto

    def map_keywords_to_domains(self):
        # Map keywords to relevant domains
        pass

    def get_lowest_common_domain(self, keyword_domains):
        # Find the lowest common domain for the topic
        pass

    def assign_name_to_topic(self):
        # Assign a name to the topic based on the lowest common domain
        pass
