from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim import corpora, models
from collections import defaultdict

class LexicalPreprocessing:
    def __init__(self):
        self.tokens = []  # List to store tokenized documents
        self.morphed_tokens = []  # List to store stemmed or lemmatized tokens
        self.vocabulary = []  # List to store the processed vocabulary
        self.vector_repr = None  # Vector representation of the documents
        self.dictionary = None  # Represents a Gensim Dictionary

    def tokenization(self, documents):
        # Tokenizes the input documents and stores them in 'self.tokens'
        self.tokens = [word_tokenize(doc) for doc in documents]

    def remove_noise(self):
        # Removes non-alphabetic words and converts to lowercase
        self.tokens = [
            [word.lower() for word in doc if word.isalpha()] for doc in self.tokens]

    def remove_stopwords(self):
        # Removes common English stopwords
        stop_words = set(stopwords.words('english'))
        self.tokens = [
            [word for word in doc if word not in stop_words] for doc in self.tokens]

    def morphological_reduction(self, use_lemmatization=True):
        if use_lemmatization:
            # Lemmatize the tokens 
            lemmatizer = WordNetLemmatizer()
            self.morphed_tokens = [[lemmatizer.lemmatize(
                word) for word in doc] for doc in self.tokens]
        else:
            # Stem the tokens 
            stemmer = PorterStemmer()
            self.morphed_tokens = [
                [stemmer.stem(word) for word in doc] for doc in self.tokens]

    def build_vocabulary(self):
        # Build the vocabulary with optional stemming or lemmatization
        all_tokens = self.morphed_tokens
        frequency = defaultdict(int)

        for doc in all_tokens:
            for token in doc:
                frequency[token] += 1

        self.vocabulary = list(frequency.keys())

    def vector_representation(self, use_bow=False):
        # Generate the vector representation of the documents using Bag of Words (BoW) or TF-IDF
        self.dictionary = corpora.Dictionary(self.morphed_tokens)
        corpus = [self.dictionary.doc2bow(doc) for doc in self.morphed_tokens]
        if use_bow:
            # Using BoW representation
            self.vector_repr = corpus
        else:
            # Using TF-IDF representation
            tfidf = models.TfidfModel(corpus)
            self.vector_repr = [tfidf[doc] for doc in corpus]

    def __repr__(self) -> str:
        # Get all attributes of the class as a dictionary and format them for printing
        attributes = vars(self)
        repr_str = "\n".join(f"{key}: {value}" for key,
                             value in attributes.items())
        return repr_str


# Create a list of documents (each document is a list of words)
documents = ["This is creation the first document.", "This document is the second document.",
             "And this is the third one.", "Is this the first document?"]

# Initialize the LexicalPreprocessing class with your documents
preprocessor = LexicalPreprocessing()

# Preprocess the documents
preprocessor.tokenization(documents)
preprocessor.remove_noise()
preprocessor.remove_stopwords()
preprocessor.morphological_reduction()
preprocessor.build_vocabulary()
preprocessor.vector_representation(True)

print(preprocessor)
