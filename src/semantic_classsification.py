import os
from gensim.models import KeyedVectors
from tools.lexical_preprocessing import LexicalPreprocessing
from semantic_preprocessing.topic_number_estimation import TopicNumberEstimation
from semantic_preprocessing.topic_discovery import TopicDiscovery
from semantic_preprocessing.topic_naming import TopicNaming
from nltk.corpus import wordnet_ic
import resource
import json
import time

def semantic_classification(corpus, corpus_name, real_k, real_tags, min_words_per_topic=20, min_sim=0.01, min_coh=0.5, wsd_algorithm='lesk'):
    if wsd_algorithm not in ["lesk", "simplex", "genetic"]:
        raise KeyError("WSD algorithm not found. Try: lesk, simplex, genetic")

    # Load pre-trained word embeddings (Word2Vec model)
    model_path = os.path.abspath('src')
    model_path += '/word2vec/GoogleNews-vectors-negative300.bin'
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # Load pre-trained information content corpus
    information_content_corpus = wordnet_ic.ic('ic-brown.dat')

    start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start_time = time.time()

    preprocesser = LexicalPreprocessing()
    preprocesser.preprocess_text(corpus, corpus_name)

    estimator = TopicNumberEstimation(word2vec_model)
    k, clusters = estimator.estimate_topic_number(preprocesser.vocabulary, preprocesser.co_occurrence_dict, min_sim, min_coh, min_words_per_topic)

    print("Estimated number of topics:", k)

    # topic_finder = TopicDiscovery(preprocesser.vector_repr, preprocesser.dictionary, k)
    # topic_model = topic_finder.train_lda()
    # topics = topic_finder.get_topics(topic_model)
    # print(topics)

    # tagger = TopicNaming(topic_model, information_content_corpus, word2vec_model)
    # tagger.tag_topics()
    # top_words = tagger.top_words
    # chosen_defs = tagger.chosen_defs
    # tags = tagger.domains
    # print("Tags:", tags)

    end_time = time.time()
    end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    execution_time = end_time - start_time
    memory_usage = end_memory - start_memory   

    test_data = {"corpus_name" : corpus_name, 
                "real_k" : real_k, 
                "real_tags" : real_tags, 
                "memory_usage" : memory_usage, 
                "execution_time" : execution_time, 
                "estimated_k" : k, 
                "clusters" : clusters, 
                # "topics" : topics, 
                # "top_words" : top_words,
                # "chosen_defs" : chosen_defs,
                # "estimated_tags" : tags, 
                "parameters" : {"min_words_per_topic" : min_words_per_topic, 
                                "min_sim" : min_sim, 
                                "min_coh" : min_coh, 
                                "wsd_algorithm" : wsd_algorithm}} 
    
    test_folder = os.path.abspath('tests')
    test_folder += f'/{corpus_name}'

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Get the count of files in the folder
    file_count = len([name for name in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, name))])

    # Filename for the test to be saved
    file_name = f"test_{file_count + 1}.json"

    # Full file path
    file_path = os.path.join(test_folder, file_name)

    # Save the test to a JSON file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(test_data, file, ensure_ascii=False, indent=4)

    print(f"Test saved in {file_path}")