from gensim import corpora


class QueryTopicExtractor:
    def __init__(self, query, model, dictionary):
        self.query = query
        self.topic_model = model
        self.dictionary = dictionary
        self.topics = []

    def extract_topics(self, prob=0.5):
        # Get the topic distribution for the query
        topic_distribution = self.topic_model[self.query]

        # Filter topics with a probability higher than the threshold
        filtered_topics = [
            topic for topic in topic_distribution if topic[1] >= prob]

        # Sort filtered topics by their contribution to the query
        sorted_topics = sorted(filtered_topics, key=lambda x: -x[1])

        self.topics = sorted_topics

        return sorted_topics

    def preprocess_query(self):
        pass
