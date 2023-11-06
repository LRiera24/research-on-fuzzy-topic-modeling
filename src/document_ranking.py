import gensim
from gensim import corpora
import numpy as np
from lexical_preprocessing import LexicalPreprocessing

class TopicDocumentMatcher:
    def __init__(self, query, topic_model, dictionary, document_topic_matrix):
        self.query = query
        
        self.topic_model = topic_model
        self.dictionary = dictionary
        self.document_topic_matrix = document_topic_matrix
        
        self.query_top_topics = []

    def match_documents_to_query(self, query_topic_distribution):
        # Preprocess the query using LexicalPreprocessing
        lp = LexicalPreprocessing()
        lp.preprocess_text([self.query])
        
        self.query = lp.vector_repr

        self.query_top_topics = self.extract_topics()

        # Filter documents that belong to the top topics
        relevant_documents = self.get_documents_in_top_topics(self.query_top_topics)

        # Rank documents within the relevant set
        ranked_documents = self.rank_documents_by_topic_similarity(
            query_topic_distribution, relevant_documents)

        return ranked_documents

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

    def get_documents_in_top_topics(self, top_topics):
        # Initialize an empty list to store relevant documents and their topics
        relevant_documents = []

        # Iterate over the top topics obtained from the query
        for topic_index, _ in top_topics:
            # Get the indices of documents that belong to the current topic
            document_indices = np.where(self.document_topic_matrix[:, topic_index] > 0)[0]
            
            # Iterate over the document indices and associate them with the topic
            for doc_index in document_indices:
                relevant_documents.append((doc_index, topic_index))

        # Return the list of relevant documents with their associated topic numbers
        return relevant_documents


    def rank_documents_by_topic_similarity(self, query_topic_distribution, relevant_documents):
        # Calculate the cosine similarity between the query topics and document topics
        similarities = np.dot(
            self.document_topic_matrix[:, [topic for _, topic in relevant_documents]], query_topic_distribution)

        # Calculate the total similarity for each document
        total_similarities = np.sum(similarities, axis=1)

        # Sort documents by total similarity in descending order
        ranked_document_indices = np.argsort(total_similarities)[::-1]

        # Get the top N documents using the ranked indices
        top_documents = [relevant_documents[i] for i in ranked_document_indices]

        return top_documents

