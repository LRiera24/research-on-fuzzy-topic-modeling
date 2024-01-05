import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(test_results):
    """
    Generate a heatmap based on a list of 2D matrices indicating success or failure for each test.

    Parameters:
    num_coherencia (int): Number of coherence values.
    num_similitud (int): Number of similarity values.
    num_pruebas (int): Number of tests to be conducted.
    """
    # Generate a list of 2D matrices (each matrix represents one test)

    # Calculate the average success rate across all tests
    average_success_rate = np.mean(test_results, axis=0)

    # Create a colormap that does not use white for 0
    cmap = sns.color_palette("BuPu", as_cmap=True)
    # Setting color for values under the threshold
    cmap.set_under(color='lightgray')

    plt.figure(figsize=(10, 8))
    sns.heatmap(average_success_rate, cmap=cmap, xticklabels=[
                i/10 for i in range(1, 10)], yticklabels=[round(1 - i/10, 1) for i in range(1, 10)], vmin=0.01)
    plt.xlabel('Valores de Similitud')
    plt.ylabel('Valores de Coherencia')
    plt.title('Mapa de Calor de Frecuencia de Aciertos Promedio')
    plt.show()
