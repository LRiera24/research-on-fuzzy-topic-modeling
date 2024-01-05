import matplotlib.pyplot as plt
import numpy as np

def plot_bar_chart(names, similarities, single_color=True):
    plt.figure(figsize=(10, 6))  # Size of the chart

    if single_color:
        plt.bar(names, similarities, color='blue')  # Create bar chart with single color
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))  # Generate different colors for each bar
        plt.bar(names, similarities, color=colors)  # Create bar chart with different colors

    plt.xlabel('Synsets')  # X-axis label
    plt.ylabel('Mean similarity')  # Y-axis label
    plt.title('')  # Chart title
    plt.xticks(rotation=45)  # Rotate names on X-axis for better visualization
    plt.show()  # Display the chart

# names = ["Name1", "Name2", "Name3"]
# similarities = [0.8, 0.6, 0.9]

# # Plot with a single color
# plot_bar_chart(names, similarities, single_color=True)

# # Plot with different colors for each bar
# plot_bar_chart(names, similarities, single_color=False)
