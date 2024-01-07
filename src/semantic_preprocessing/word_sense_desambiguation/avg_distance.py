from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic

def calculate_mean_similarity(synsets):
    calculated_distances = []
    sim = 0
    c = 0
    for synset1 in synsets:
        for synset2 in synsets:
            if synset1 == synset2 or (synset1, synset2) in calculated_distances or (synset2, synset1) in calculated_distances or synset1.pos() == 'v' or synset2.pos() == 'v':
                continue
            calculated_distances.append((synset1, synset2))
            sim += synset1.path_similarity(synset2)
            c += 1
    if c > 0:
        sim = sim / c
    else: 
        sim = -1
    return sim
