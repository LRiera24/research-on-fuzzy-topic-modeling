import matplotlib.pyplot as plt
import numpy as np


def plot_scatter(success_matrix, data, name=''):
    """
    Plot a scatter plot for each pair of coherence and similarity values,
    differentiating successes (with cross markers) and failures.

    Parameters:
    coherence_list (array-like): List of coherence values.
    similarity_list (array-like): List of similarity values.
    success_matrix (2D array-like): Matrix indicating success (True) or failure (False) 
                                    for each combination of coherence and similarity.
    """
    plt.figure(figsize=(9, 7))
    matches = 0 
    offset = 0.015
    # Plot each point and use different markers for success and failure
    for i, coherence in enumerate([i/10 for i in range(1, 10)]):
        for j, similarity in enumerate([i/10 for i in range(1, 10)]):
            if success_matrix[i][j]:
                plt.scatter(coherence, similarity, color='green', s=150)
                plt.text(coherence, similarity + offset, f"{data[matches]['estimated_k']}", fontsize=15)
                matches += 1
            else:
                plt.scatter(coherence, similarity, color='red', marker='x', s=150)

    plt.xlabel('Coherencia', fontsize=15)
    plt.ylabel('Similitud', fontsize=15)
    # plt.title(
    #     f'Relación entre Coherencia y Similitud en la Identificación de Tópicos: {matches} aciertos', fontsize=15)

    # Custom legend
    plt.scatter([], [], color='green', label=f'Acierto')
    plt.scatter([], [], color='red', marker='x', label='Fallo')
    plt.legend()
    plt.savefig(f'coh_sim_relation_{name}', bbox_inches='tight', pad_inches=0.2)
    plt.show()



# # Matriz de aciertos/fallos
# acierto = np.random.choice([True, False], (9, 9))
# print(acierto)

# # Llamando a la función con los datos de ejemplo
# plot_scatter(acierto)
