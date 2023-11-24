from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic

def calculate_similarity(synset1, synset2, measure='wup'):
    ic = None
    
    if measure in ['resnik', 'jcn', 'lin']:
        # Load information content (IC) file
        ic = wordnet_ic.ic('ic-brown.dat')
    
    if measure == 'path':
        return synset1.path_similarity(synset2)
    elif measure == 'wup':
        return synset1.wup_similarity(synset2)
    elif measure == 'resnik':
        return synset1.res_similarity(synset2, ic)
    elif measure == 'jcn':
        return synset1.jcn_similarity(synset2, ic)
    elif measure == 'lin':
        return synset1.lin_similarity(synset2, ic)
    else:
        raise ValueError(f"Unsupported similarity measure: {measure}")

def calculate_mean_similarity(synsets, measure='path'):
    num_synsets = len(synsets)
    total_similarity = 0.0

    for i in range(num_synsets - 1):
        for j in range(i + 1, num_synsets):
            similarity = calculate_similarity(synsets[i], synsets[j], measure)
            total_similarity += similarity

    # Calculate mean similarity
    mean_similarity = total_similarity / (num_synsets * (num_synsets - 1) / 2)
    return mean_similarity

# Example Usage
context = ['egg', 'sugar', 'butter', 'flour', 'recipe', 'cake', 'dessert', 'bake', 'sweet', 'birthday']

word_synsets = [wordnet.synsets(word)[0] for word in context]  # Assuming you want only the first synset for each word

avg_distance = calculate_mean_similarity(word_synsets)

print("Average Distance Between Synsets:", avg_distance)
