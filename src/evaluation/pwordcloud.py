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
    num_word_sets = len(list_of_word_sets)
    num_columns = min(4, int(num_word_sets / 2))
    num_rows = -(-num_word_sets // num_columns)  # Ceiling division

    fig = plt.figure(figsize=(20 + num_columns * 5, num_rows * 15))
    plt.suptitle(f'Identified number of topics in the corpus - {num_word_sets}', fontsize=20, y=0.96, weight='bold')

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
            wordcloud = WordCloud(width=300, height=300, background_color='white', color_func=color_func).generate_from_frequencies(filtered_dict)
        else:
            wordcloud = WordCloud(width=300, height=300, background_color='white', color_func=color_func).generate(text)

        plt.subplot(num_rows, num_columns, i)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        if not tags:
            plt.title(f'Topic {i}', y=-0.1)
        else:
            plt.title(split_title(", ".join(tags[i-1])), fontsize=40, y=-0.1)

    plt.tight_layout(pad=15.0, h_pad=15.0, w_pad=15.0, rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'wordcloud_{img_name}.png')

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

