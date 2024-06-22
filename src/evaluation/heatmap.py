import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from matplotlib.colors import LinearSegmentedColormap


def plot_heatmap(test_results, interpolation_factor=10):
    # Convert boolean arrays to int 
    test_results = [result.astype(int) for result in test_results]

    # Calculate the average success rate across all tests
    average_success_rate = np.mean(test_results, axis=0)

    # Coordinates of the original data
    x = np.arange(0, average_success_rate.shape[1])
    y = np.arange(average_success_rate.shape[0] - 1, -1, -1)
    xx, yy = np.meshgrid(x, y)

    # Interpolating data using linear interpolation
    interpolate_function = interpolate.interp2d(x, y, average_success_rate, kind='linear')
    xnew = np.linspace(0, x.max(), average_success_rate.shape[1] * interpolation_factor)
    ynew = np.linspace(0, y.max(), average_success_rate.shape[0] * interpolation_factor)
    new_data = interpolate_function(xnew, ynew)

    # Modify the BuPu colormap to start from a more blueish color and exclude the lightest colors
    original_cmap = sns.color_palette("BuPu", as_cmap=True)
    modified_colors = original_cmap(np.linspace(0.2, 1, 256))  # Start from a point further into the colormap
    modified_cmap = LinearSegmentedColormap.from_list("modified_BuPu", modified_colors)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(new_data, cmap=modified_cmap, annot=False, square=True, linewidths=0, vmin=0, vmax=1)

    # Adjusting the ticks for the original scale
    ax.set_xticks(np.linspace(0, average_success_rate.shape[1] * interpolation_factor - 1, average_success_rate.shape[1]))
    ax.set_yticks(np.linspace(0, average_success_rate.shape[0] * interpolation_factor - 1, average_success_rate.shape[0]))
    ax.set_xticklabels([i/10 for i in range(1, 10)])
    ax.set_yticklabels([round(1 - i/10, 1) for i in range(1, 10)])

    plt.xlabel('Similariry Values')
    plt.ylabel('Coherence Values')
    plt.title('Heatmap of average hit frequency')
    plt.savefig('heatmap', bbox_inches='tight', pad_inches=0.2)
    plt.show()
