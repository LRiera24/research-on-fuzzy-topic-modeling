from nltk.corpus import wordnet as wn

# Let's take a synset for the word "dog"
dog_synset = wn.synsets('dog')[0]  # taking the first synset for demonstration

# Using str() to get string representation
print(str(dog_synset))

# Using the name() method
print(dog_synset.name())
