from gensim import models

class TopicDiscovery:
    """
    A class for topic discovery using Latent Dirichlet Allocation (LDA).

    This class provides functionality to train an LDA model on a given set of document vectors,
    and retrieve topics and document-topic distributions from the trained model.

    Attributes:
        vector_representation (list): A list of document vectors.
        dictionary (gensim.corpora.dictionary.Dictionary): A Gensim dictionary mapping of ids to words.
        num_topics (int): The number of topics to be extracted by the LDA model.
    """

    def __init__(self, vector_representation, dictionary, num_topics):
        """
        Initializes the TopicDiscovery class with document vectors, a dictionary, and the number of topics.

        Args:
            vector_representation (list): A list of document vectors.
            dictionary (gensim.corpora.dictionary.Dictionary): A Gensim dictionary.
            num_topics (int): The number of topics to be extracted by the LDA model.
        """
        self.vector_representation = vector_representation
        self.dictionary = dictionary
        self.num_topics = num_topics

    def train_lda(self, passes=100):
        """
        Train an LDA model on the document vectors.

        Args:
            passes (int): The number of passes through the corpus during training.

        Returns:
            gensim.models.LdaModel: A trained LDA model.
        """
        lda_model = models.LdaModel(
            self.vector_representation, num_topics=self.num_topics, id2word=self.dictionary,
                                           chunksize=2000,
                                           passes=passes, iterations=50)
        return lda_model

    def get_topics(self, model, num_words=10):
        """
        Get the topics from a trained LDA model.

        Args:
            model (gensim.models.LdaModel): A trained LDA model.
            num_words (int): The number of words to include for each topic.

        Returns:
            list of tuples: Each tuple contains the topic number and the words associated with the topic.
        """
        return model.print_topics(num_topics=-1, num_words=num_words)

    def get_document_topics(self, model):
        """
        Get the topic distribution for each document in the vector representation.

        Args:
            model (gensim.models.LdaModel): A trained LDA model.

        Returns:
            list of lists: Each inner list contains tuples of topic number and its proportion in the document.
        """
        return [model[doc] for doc in self.vector_representation]

    def get_topic_category_correspondence(self, model, real_categories):
        """
        Establish a correspondence between LDA topics and real document categories, without repetitions.

        Args:
            model (gensim.models.LdaModel): A trained LDA model.
            real_categories (list): A list of real categories for each document.

        Returns:
            dict: A dictionary with LDA topics as keys and sets of real categories as values.
        """
        topic_category_correspondence = {i: set() for i in range(self.num_topics)}

        for doc_vector, real_category in zip(self.vector_representation, real_categories):
            # Obtener la distribución de tópicos para el documento
            doc_topics = model[doc_vector]

            if len(doc_topics) > 0:
                # Identificar el tópico dominante para el documento
                doc_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)
                print(doc_topics)
                dominant_topic = doc_topics[0][0]
                print(dominant_topic)

                # Agregar todas las categorías reales del documento al conjunto del tópico dominante
                topic_category_correspondence[dominant_topic].update(real_category)

        return topic_category_correspondence
