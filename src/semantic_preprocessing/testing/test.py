import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('averaged_perceptron_tagger')

# Sample text
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the text
tokens = word_tokenize(text)

# Perform POS tagging
tagged = pos_tag(tokens)

# Filter out nouns and verbs
nouns_verbs = [word for word, tag in tagged if tag.startswith('NN') or tag.startswith('VB')]

print(nouns_verbs)
