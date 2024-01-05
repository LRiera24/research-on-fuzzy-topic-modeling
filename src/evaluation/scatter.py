import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(success_matrix):
    """
    Plot a scatter plot for each pair of coherence and similarity values,
    differentiating successes (with cross markers) and failures.

    Parameters:
    coherence_list (array-like): List of coherence values.
    similarity_list (array-like): List of similarity values.
    success_matrix (2D array-like): Matrix indicating success (True) or failure (False) 
                                    for each combination of coherence and similarity.
    """
    plt.figure(figsize=(10, 8))
    matches = 0 
    # Plot each point and use different markers for success and failure
    for i, coherence in enumerate([i/10 for i in range(1, 10)]):
        for j, similarity in enumerate([i/10 for i in range(1, 10)]):
            if success_matrix[i][j]:
                plt.scatter(coherence, similarity, color='green')
                matches += 1
            else:
                plt.scatter(coherence, similarity, color='red', marker='x')

    plt.xlabel('Coherencia')
    plt.ylabel('Similitud')
    plt.title(
        f'Relación entre Coherencia y Similitud en la Identificación de Tópicos: {matches} aciertos')

    # Custom legend
    plt.scatter([], [], color='green', label=f'Acierto')
    plt.scatter([], [], color='red', marker='x', label='Fallo')
    plt.legend()
    plt.show()



# # Matriz de aciertos/fallos
# acierto = np.random.choice([True, False], (9, 9))
# print(acierto)

# # Llamando a la función con los datos de ejemplo
# plot_scatter(acierto)
