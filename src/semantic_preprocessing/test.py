import numpy as np
from collections import Counter

def calculate_co_occurrence_dict(vocabulary, documents, window_size):
    # Flatten the list of tokenized documents
    flat_documents = [token for doc in documents for token in doc]

    # Compute the frequency distribution of words
    fdist = Counter(flat_documents)

    # Initialize the co-occurrence dictionary
    co_occurrence_dict = {word: {} for word in vocabulary}

    # Iterate over each document
    for doc in documents:
        # Iterate over each word in the document
        for i, word in enumerate(doc):
            # Get the context window around the current word
            start = max(0, i - window_size)
            end = min(len(doc), i + window_size + 1)
            context = doc[start:end]

            # Update co-occurrence counts
            for context_word in context:
                if word != context_word and word in vocabulary and context_word in vocabulary:
                    co_occurrence_dict[word][context_word] = co_occurrence_dict[word].get(context_word, 0) + 1

    # Normalize the co-occurrence dictionary
    for word, context_dict in co_occurrence_dict.items():
        total_word_frequency = fdist[word]
        for context_word, count in context_dict.items():
            co_occurrence_dict[word][context_word] = count / (total_word_frequency * fdist[context_word])

    return co_occurrence_dict

# Example usage
vocabulary = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
documents = [
    ["the", "quick", "brown", "fox"],
    ["brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["the", "quick", "fox", "jumps"],
    # Add more documents as needed
]
window_size = 2

result_dict = calculate_co_occurrence_dict(vocabulary, documents, window_size)
print(result_dict)
