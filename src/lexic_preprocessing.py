from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim import corpora, models
from collections import defaultdict


class LexicalPreprocessing:
    def __init__(self, documents):
        self.documents = documents
        self.tokens = []
        self.clean_tokens = []
        self.stemmed_tokens = []
        self.lemmatized_tokens = []
        self.vocabulary = []
        self.vector_representation_bow = None
        self.vector_representation_tfidf = None

    def tokenization(self):
        self.tokens = [word_tokenize(doc) for doc in self.documents]

    def remove_noise(self):
        self.clean_tokens = [
            [word.lower() for word in doc if word.isalpha()] for doc in self.tokens]

    def remove_stopwords(self):
        stop_words = set(stopwords.words('english'))
        self.clean_tokens = [
            [word for word in doc if word not in stop_words] for doc in self.clean_tokens]

    def stemming_or_lemmatization(self, use_lemmatization=False):
        if use_lemmatization:
            lemmatizer = WordNetLemmatizer()
            self.lemmatized_tokens = [[lemmatizer.lemmatize(
                word) for word in doc] for doc in self.clean_tokens]
        else:
            stemmer = PorterStemmer()
            self.stemmed_tokens = [
                [stemmer.stem(word) for word in doc] for doc in self.clean_tokens]

    def build_vocabulary(self):
        all_tokens = self.stemmed_tokens if self.stemmed_tokens else self.lemmatized_tokens
        frequency = defaultdict(int)

        for doc in all_tokens:
            for token in doc:
                frequency[token] += 1

        self.vocabulary = list(frequency.keys())

    def vector_representation(self, use_tfidf=False):
        if use_tfidf:
            dictionary = corpora.Dictionary(self.stemmed_tokens)
            corpus = [dictionary.doc2bow(doc) for doc in self.stemmed_tokens]
            tfidf = models.TfidfModel(corpus)
            self.vector_representation_tfidf = [tfidf[doc] for doc in corpus]
        else:
            dictionary = corpora.Dictionary(self.stemmed_tokens)
            corpus = [dictionary.doc2bow(doc) for doc in self.stemmed_tokens]
            self.vector_representation_bow = corpus

    def __repr__(self) -> str:
        # Get all attributes of the class as a dictionary
        attributes = vars(self)
        repr_str = "\n".join(f"{key}: {value}" for key,
                             value in attributes.items())
        return repr_str


# Create a list of documents (each document is a list of words)
documents = ["This is creation the first document.", "This document is the second document.",
             "And this is the third one.", "Is this the first document?"]

# Initialize the LexicalPreprocessing class with your documents
preprocessor = LexicalPreprocessing(documents)

# Preprocess the documents
preprocessor.tokenization()
preprocessor.remove_noise()
preprocessor.remove_stopwords()
preprocessor.stemming_or_lemmatization()
preprocessor.build_vocabulary()

# Calculate Bag of Words (BoW) vector representation
preprocessor.vector_representation(use_tfidf=False)

# Calculate TF-IDF vector representation
preprocessor.vector_representation(use_tfidf=True)

print(preprocessor)
