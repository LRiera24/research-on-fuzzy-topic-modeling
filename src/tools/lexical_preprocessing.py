from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim import corpora, models
from collections import defaultdict
import json
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

class LexicalPreprocessing:
    """
    Class for performing lexical preprocessing on a collection of text documents.
    """

    def __init__(self):
        """
        Initialize the LexicalPreprocessing class.
        """
        self.tokens = []  # List to store tokenized documents
        self.morphed_tokens = []  # List to store stemmed or lemmatized tokens
        self.vocabulary = []  # List to store the processed vocabulary
        self.vector_repr = None  # Vector representation of the documents
        self.dictionary = None  # Represents a Gensim Dictionary
        self.co_occurrence_dict = None  # Co-ocurrence matrix of the corpus

    def preprocess_text(self, documents, name):
        """
        Perform lexical preprocessing on the given documents.

        Parameters:
        - documents (list): List of input documents.
        - name (str): Name of the corpus (used for saving co-occurrence matrix).

        Returns:
        - None
        """
        self._tokenization(documents)
        self._remove_noise()
        self._remove_stopwords()
        self._morphological_reduction()
        self._filter_tokens_by_occurrence()
        self._build_vocabulary()
        self._vector_representation()
        self._calculate_co_occurrence_matrix(name)

    def _tokenization(self, documents):
        """
        Tokenize the input documents and store them in 'self.tokens'.

        Parameters:
        - documents (list): List of input documents.

        Returns:
        - None
        """
        self.tokens = [word_tokenize(doc) for doc in documents]

    def _remove_noise(self):
        """
        Remove non-alphabetic words and convert to lowercase.

        Returns:
        - None
        """
        self.tokens = [
            [word.lower() for word in doc if word.isalpha() and len(word)>1] for doc in self.tokens]

    def _remove_stopwords(self):
        """
        Remove common English stopwords.

        Returns:
        - None
        """
        stop_words = set(stopwords.words('english'))
        self.tokens = [
            [word for word in doc if word not in stop_words] for doc in self.tokens]

    def _filter_tokens_by_occurrence(self, no_below=0.1, no_above=0.5):
        """
        Filter out infrequent and highly frequent words.

        Parameters:
        - no_below (float): Threshold for filtering infrequent words.
        - no_above (float): Threshold for filtering highly frequent words.

        Returns:
        - None
        """
        self.dictionary = corpora.Dictionary(self.morphed_tokens)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    def _morphological_reduction(self, use_lemmatization=True):
        """
        Apply lemmatization or stemming to the tokens.

        Parameters:
        - use_lemmatization (bool): Use lemmatization if True, else use stemming.

        Returns:
        - None
        """
        real_words = set(words.words())
        if use_lemmatization:
            lemmatizer = WordNetLemmatizer()
            self.morphed_tokens = [[lemmatizer.lemmatize(
                word) for word in doc if word in real_words] for doc in self.tokens]
        else:
            stemmer = PorterStemmer()
            self.morphed_tokens = [
                [stemmer.stem(word) for word in doc] for doc in self.tokens]

    def _build_vocabulary(self):
        """
        Build the vocabulary

        Returns:
        - None
        """
        all_tokens = self.morphed_tokens
        frequency = defaultdict(int)

        for doc in all_tokens:
            for token in doc:
                frequency[token] += 1

        self.vocabulary = list(frequency.keys())

    def _vector_representation(self, use_bow=True):
        """
        Generate vector representation of the documents using Bag of Words (BoW) or TF-IDF.

        Parameters:
        - use_bow (bool): Use Bag of Words if True, else use TF-IDF.

        Returns:
        - None
        """
        corpus = [self.dictionary.doc2bow(doc) for doc in self.morphed_tokens]
        if use_bow:
            self.vector_repr = corpus
        else:
            tfidf = models.TfidfModel(corpus)
            self.vector_repr = [tfidf[doc] for doc in corpus]

    def _calculate_co_occurrence_matrix(self, corpus_name):
        """
        Calculate the co-occurrence matrix and save it to a file.

        Parameters:
        - corpus_name (str): Name used for saving co-occurrence matrix.

        Returns:
        - dict: Co-occurrence matrix as a dictionary.
        """
        path = os.path.abspath('src') + '/co_occurrence_dicts/' + f'{corpus_name}.json'
        if os.path.exists(path):
            with open(path, 'r') as file:
                self.co_occurrence_dict = json.load(file)
        else:
            corpus = [" ".join(doc) for doc in self.morphed_tokens]

            cv = CountVectorizer(ngram_range=(1, 1), stop_words='english')
            X = cv.fit_transform(corpus)
            Xc = (X.T * X)
            Xc.setdiag(0)
            Xc_normalized = normalize(Xc, norm='l2', axis=1)
            names = cv.get_feature_names_out()

            cooccurrence_dict = {}
            for i, word in enumerate(names):
                word_dict = {}
                for j in range(len(names)):
                    if Xc_normalized[i, j] > 0:
                        word_dict[names[j]] = Xc_normalized[i, j]
                cooccurrence_dict[word] = word_dict

            with open(path, 'w') as file:
                json.dump(cooccurrence_dict, file)

            self.co_occurrence_dict = cooccurrence_dict
            return cooccurrence_dict

    def __repr__(self) -> str:
        """
        Get a string representation of the class attributes.

        Returns:
        - str: String representation of the class.
        """
        attributes = vars(self)
        repr_str = "\n".join(f"{key}: {value}" for key,
                             value in attributes.items())
        return repr_str



# # Create a list of documents (each document is a list of words)
# documents = ["This is first creation the first document.", "This document is the second document.",
#              "And this is the third one.", "Is hola document laura victoria riera perez holita holiwis fuera this the first document?"]

# documents = ['kfhwluighwlr, dog, cat, rabbit', 
#             'egg, sugar, butter, flour, recipe, cake, dessert', 
#             'birthday, party, gift, music, candles, wish', 
#             'computer, program, development, web, application, data', 
#             'school, class, homework, student, book, knowledge, learn, teach']

# # Initialize the LexicalPreprocessing class with your documents
# preprocessor = LexicalPreprocessing()
# preprocessor.preprocess_text(documents, 'test2')

# print(preprocessor)