def filter_keywords_sliding_window(tuple_list, window_size=5):
    """
    Filters keywords based on a significant drop in probabilities using a sliding window approach.

    Args:
        tuple_list (list of tuples): A list of tuples (word, probability).
        window_size (int): The size of the sliding window to consider for change in probability.

    Returns:
        list: A list of words before a significant drop in probability.
    """

    # Ensure the tuples are sorted by probability in descending order
    tuple_list = sorted(tuple_list, key=lambda x: x[1], reverse=True)

    # Calculate the average drop in probability for each window
    avg_drops = []
    for i in range(len(tuple_list) - window_size + 1):
        window = tuple_list[i:i + window_size]
        avg_drop = (window[0][1] - window[-1][1]) / window_size
        avg_drops.append(avg_drop)

    # Find the window with the maximum average drop
    max_avg_drop = max(avg_drops)
    max_drop_index = avg_drops.index(max_avg_drop)

    # Return words up to the start of the window with the maximum average drop
    return [word for word, _ in tuple_list[:max_drop_index + 1]]

# Example usage
tuples = [('apple', 0.9), ('orange', 0.85), ('banana', 0.83), ('kiwi', 0.5), ('melon', 0.35), ('melon', 0.15), ('melon', 0.24), ('melon', 0.8), ('melon', 0.15), ('melon', 0.15), ('melon', 0.15), ('melon', 0.00), ('melon', 0.00), ('melon', 0.00), ('melon', 0.00), ('melon', 0.00), ('melon', 0.00)]
filtered_words = filter_keywords_sliding_window(tuples)
print(filtered_words)
