import os
import json
import numpy as np

def process_jsons(folder_path, window_lower=5, window_upper=5):
    """
    Filters JSON files in the given folder based on whether the estimated_k is within a specified window compared to real_k,
    and also returns a boolean list and a 9x9 matrix indicating if each file's estimated_k falls within the range.
    
    Parameters:
    folder_path (str): Path to the folder containing JSON files.
    window_lower (int): The lower bound for the allowable difference between estimated_k and real_k.
    window_upper (int): The upper bound for the allowable difference between estimated_k and real_k.

    Returns:
    list: A list of JSON objects that meet the criteria.
    list: A boolean list indicating whether each file's estimated_k falls within the specified range.
    ndarray: A 9x9 numpy matrix of booleans from the boolean list.
    """
    filtered_jsons = []
    matrix = np.full((9, 9), False)

    # Iterate over files in the given folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    estimated_k = data.get('estimated_k')
                    real_k = data.get('real_k')
                    coh = data.get('parameters')['min_coh']
                    print(coh)
                    sim = data.get('parameters')['min_sim']
                    print(sim)

                    # Check if the estimated_k is within the specified window compared to real_k
                    if estimated_k is not None and real_k is not None:
                        difference = estimated_k - real_k
                        if real_k - window_lower <= estimated_k <= real_k + window_upper:
                            filtered_jsons.append(data)
                            matrix[int(coh*10-1)][int(sim*10-1)] = True
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")

    return filtered_jsons, matrix


test_folder = os.path.abspath('tests')
test_folder += f'/Reuters/run1_100'

# Example usage
filtered_data, flags_matrix = process_jsons(test_folder, window_lower=5, window_upper=5)
print(flags_matrix)

