from nltk.corpus import wordnet

def average_distance_between_synsets(synsets):
    if len(synsets) < 2:
        return 0.0  # No pairs to compare

    total_distance = 0.0
    pair_count = 0

    for i in range(len(synsets)):
        for j in range(i + 1, len(synsets)):
            distance = synsets[i].path_similarity(synsets[j])
            if distance is not None:
                total_distance += distance
                pair_count += 1

    if pair_count == 0:
        return 0.0  # No valid pairs

    return total_distance / pair_count

# Example Usage
context = ['egg', 'sugar', 'butter', 'flour', 'recipe', 'cake', 'dessert']

word_synsets = [wordnet.synsets(word)[0] for word in context]  # Assuming you want only the first synset for each word

avg_distance = average_distance_between_synsets(word_synsets)

print("Average Distance Between Synsets:", avg_distance)
