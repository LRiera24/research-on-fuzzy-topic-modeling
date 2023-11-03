import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

class TopicNaming:
    def __init__(self, model):
        self.model = model
        self.domains = []

    def assign_name_to_topic(self):
        # Assign a name to the topic based on the lowest common domain
        for topic_num in range(self.model.num_topics):
            top_words = self.get_top_words(topic_num, 10)
            synsets = self.get_top_words_synsets(top_words)
            self.domains.append(self.get_lowest_common_hypernym(synsets))
    
    def get_top_words(self, topic_num, k):
        # Get the most probable words for the specified topic
        topic_words = self.model.show_topic(topic_num, topn=k)

        # Sort the words by their probabilities in descending order
        topic_words = sorted(topic_words, key=lambda x: x[1], reverse=True)

        # Get the top K words for the topic
        top_words = [word for word, prob in topic_words[:K]]

        print(f"Top {k} words for topic {topic_num}: {top_words}")

        return top_words

    def get_top_words_synsets(self, top_words):
        synsets = [wordnet.synsets(word) for word in top_words]
        return synsets

    def get_lowest_common_hypernym(self, synsets):
        # Find the lowest common domain for the topic

        # Initialize the common hypernym with the first synset
        common_hypernym = synsets[0]

        # Find the common hypernym with the rest of the synsets
        for synset in synsets[1:]:
            common_hypernym = common_hypernym.lowest_common_hypernym(synset)


# words = ['cat', 'dog', 'bird']
# common_hypernym = lowest_common_hypernym(words)
# print("Lowest Common Hypernym:", common_hypernym)
