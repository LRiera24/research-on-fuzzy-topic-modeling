from wordcloud import WordCloud, get_single_color_func
import matplotlib.pyplot as plt
import random

def rgb_to_hex(rgb):
    """ Convert RGB tuple to hexadecimal string """
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def wordclouds_matrix(list_of_word_sets):
    num_word_sets = len(list_of_word_sets)
    num_columns = int(num_word_sets / 2)
    num_rows = -(-num_word_sets // num_columns)  # Ceiling division to determine the number of rows

    # Adjust figure size for additional horizontal spacing between plots
    fig = plt.figure(figsize=(20 + num_columns, num_rows * 6))

    # Slightly adjust main title position for less spacing
    plt.suptitle(f'Identified number of topics in the corpus - {num_word_sets}', fontsize=20, y=0.96, weight='bold')

    for i, word_set in enumerate(list_of_word_sets, start=1):
        # Combine the set of words into a single string
        text = ' '.join(word_set)

        # Randomly generate a base color in RGB format
        base_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        hex_color = rgb_to_hex(base_color)  # Convert RGB to hex

        # Create a word cloud object with a color function based on the base color
        color_func = get_single_color_func(hex_color)
        wordcloud = WordCloud(width=300, height=300, background_color='white', color_func=color_func).generate(text)

        # Determine the position in the grid and adjust title position
        plt.subplot(num_rows, num_columns, i)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Topic {i}', y=-0.1)

    # Adjust layout for more horizontal space between plots
    plt.tight_layout(pad=4.0, h_pad=4.0, w_pad=4.0)
    plt.show()

example_sets_of_words = [
    ["python", "programming", "code", "scripting", "automation"],
    ["machine", "learning", "AI", "neural", "networks"],
    ["data", "analytics", "big data", "visualization", "statistics"],
    ["environment", "sustainability", "conservation", "ecology"],
    ["education", "teaching", "learning", "school", "university"],
    ["health", "medicine", "wellness", "fitness", "nutrition"]
]

# Generate and save the plot with less title spacing
less_spacing_file_path = wordclouds_matrix(example_sets_of_words)
less_spacing_file_path

