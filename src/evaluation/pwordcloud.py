from wordcloud import WordCloud, get_single_color_func
import matplotlib.pyplot as plt
import random

def split_title(title, max_length=30):
    """ Split a title into multiple lines if it's too long. """
    if len(title) <= max_length:
        return title
    else:
        words = title.split()
        split_title = words[0]
        current_length = len(words[0])

        for word in words[1:]:
            if current_length + len(word) + 1 > max_length:
                split_title += '\n' + word
                current_length = len(word)
            else:
                split_title += ' ' + word
                current_length += len(word) + 1

        return split_title

def rgb_to_hex(rgb):
    """ Convert RGB tuple to hexadecimal string """
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def plot_wordclouds(list_of_word_sets, tfidf_scores, img_name, tags=None, use_frequency=True):
    for i, word_set in enumerate(list_of_word_sets, start=1):
        filtered_dict = {}
        for word in word_set:
            filtered_dict[word] = tfidf_scores[word]

        text = ' '.join(word_set)

        base_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        hex_color = rgb_to_hex(base_color)
        color_func = get_single_color_func(hex_color)

        wordcloud = None
        if use_frequency:
            wordcloud = WordCloud(width=800, height=600, background_color='white', color_func=color_func).generate_from_frequencies(filtered_dict)
        else:
            wordcloud = WordCloud(width=800, height=600, background_color='white', color_func=color_func).generate(text)

        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        if not tags:
            plt.title(f'Topic {i}', fontsize=20)
        else:
            plt.title(split_title(", ".join(tags[i-1])), fontsize=20)

        plt.tight_layout()
        plt.savefig(f'individual_wordcloud_{img_name}_topic_{i}.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()

# example_sets_of_words = [
#     ["python", "programming", "code", "scripting", "automation"],
#     ["machine", "learning", "AI", "neural", "networks"],
#     ["data", "analytics", "big data", "visualization", "statistics"],
#     ["environment", "sustainability", "conservation", "ecology"],
#     ["education", "teaching", "learning", "school", "university"],
#     ["health", "medicine", "wellness", "fitness", "nutrition"]
# ]

# tags = [["python", "programming"],
#         ["AI" * 20, "hola"],
#         ["data", "analytics"],
#         ["environment", "sustainability", "conservation"],
#         ["school"],
#         ["health", "medicine", "wellness", "fitness", "nutrition"]]
# # Generate and save the plot with less title spacing
# less_spacing_file_path = plot_wordclouds(example_sets_of_words, tags)
# less_spacing_file_path

