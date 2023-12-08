from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic

class TopicNaming:
    def __init__(self, model):
        self.model = model
        self.domains = []

    def assign_name_to_topic(self):
        # Assign a name to the topic based on the lowest common domain
        for topic_num in range(self.model.num_topics):
            top_words = self.get_top_words(topic_num, 10)
            synsets = self.get_definitions_for_context(top_words)
            self.domains.append(self.get_common_hypernym(synsets))

    def get_top_words(self, topic_num, k):
        # Get the most probable words for the specified topic
        topic_words = self.model.show_topic(topic_num, topn=k)

        # Sort the words by their probabilities in descending order
        topic_words = sorted(topic_words, key=lambda x: x[1], reverse=True)

        # Get the top K words for the topic
        top_words = [word for word, prob in topic_words[:k]]

        print(f"Top {k} words for topic {topic_num}: {top_words}")

        return top_words

    def get_definitions_for_context(self, context, algorithm='simplex'):
        synsets = [wordnet.synsets(word) for word in context]
        chosen_synsets = []
        if algorithm == 'simplex':
            pass
        if algorithm == 'genetic':
            pass
        else:
            pass
        return chosen_synsets

    def get_common_hypernym(self, synsets, thredshole):
        # Find the lowest common domain for the topic
        
        tags = {}
        tags['MAX_OCCURRENCE'] = 0

        # Find the minimum depth
        min_depth = min(s.min_depth() for s in synsets)

        def hyper(s): return s.hypernyms()

        # Count lemmas at the specified depth
        for s in synsets:
            for w in list(s.closure(hyper)):
                if w.min_depth() == min_depth:
                    tags[w] = tags.get(w, 0) + 1 
                    tags['MAX_OCCURRENCE'] = max(tags['MAX_OCCURRENCE'], tags[w])

        print(tags)
        print('CHOSEN TAGS')
        for l in tags.keys():
            try:    
                if tags[l] == tags['MAX_OCCURRENCE']:
                    print(l, l.min_depth())
            except:
                pass

