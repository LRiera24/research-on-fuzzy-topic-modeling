
from gensim import models, corpora

class TopicDiscovery:
    def __init__(self, vector_representation, dictionary, num_topics):
        self.vector_representation = vector_representation  # List of document vectors
        self.dictionary = dictionary  # Gensim dictionary
        self.num_topics = num_topics

    def train_lda(self, passes=10):
        # Train an LDA model
        lda_model = models.LdaModel(
            self.vector_representation, num_topics=self.num_topics, id2word=self.dictionary, passes=passes)
        return lda_model

    def train_lsa(self):
        # Train an LSA model
        lsa_model = models.LsiModel(
            self.vector_representation, num_topics=self.num_topics, id2word=self.dictionary)
        return lsa_model

    def train_nmf(self):
        # Train an NMF model
        nmf_model = models.Nmf(
            self.vector_representation, num_topics=self.num_topics, id2word=self.dictionary)
        return nmf_model

    def train_hdp(self):
        # Train an HDP model
        hdp_model = models.HdpModel(
            self.vector_representation, id2word=self.dictionary)
        return hdp_model

    def train_ctm(self, dictionary):
        # Train a Correlated Topic Model (CTM)
        ctm_model = models.CtmModel(orpus=self.vector_representation, id2word=dictionary, num_topics=self.num_topics)
        return ctm_model

    def train_dtm(self, num_topics=10, passes=10):
        # Train a Dynamic Topic Model (DTM)
        dtm_model = models.DtmModel(
            self.vector_representation, num_topics=num_topics, id2word=self.dictionary, passes=passes)
        return dtm_model

    def get_topics(self, model, num_words=10):
        # Get topics from a trained model
        return model.print_topics(num_topics=-1, num_words=num_words)

    def get_document_topics(self, model):
        # Get topic distribution for each document
        return [model[doc] for doc in self.vector_representation]

