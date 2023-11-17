import matplotlib.pyplot as plt
from sklearn.metrics import auc

class EvaluationPlots:
    def __init__(self):
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.linestyles = ['-', '--', '-.', ':']

    def plot_confusion_matrix(self, confusion_matrix, class_labels):
        # Plot a confusion matrix
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = range(len(class_labels))
        plt.xticks(tick_marks, class_labels, rotation=45)
        plt.yticks(tick_marks, class_labels)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

    def plot_roc_curve(self, fpr, tpr, auc_score):
        # Plot ROC curve
        plt.plot(fpr, tpr, color=self.colors[0], lw=2, label=f"ROC curve (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')

    def plot_precision_recall_curve(self, precision, recall, auc_score):
        # Plot precision-recall curve
        plt.plot(recall, precision, color=self.colors[0], lw=2, label=f"PR curve (AUC = {auc_score:.2f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')

    def plot_class_distribution(self, class_labels, class_counts):
        # Plot class distribution
        plt.bar(class_labels, class_counts, color=self.colors[0])
        plt.xlabel('Class Labels')
        plt.ylabel('Counts')
        plt.title('Class Distribution')

    def plot_precision_recall_multi(self, precision_recall_data, labels):
        # Plot multiple precision-recall curves
        for i, (precision, recall, auc_score) in enumerate(precision_recall_data):
            plt.plot(recall, precision, color=self.colors[i], lw=2, label=f"{labels[i]} (AUC = {auc_score:.2f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')

    def plot_cumulative_gain_curve(self, cumulative_gain_data, labels):
        # Plot multiple cumulative gain curves
        for i, (x, y, auc_score) in enumerate(cumulative_gain_data):
            plt.plot(x, y, color=self.colors[i], linestyle=self.linestyles[i], lw=2,
                     label=f"{labels[i]} (AUC = {auc_score:.2f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Normalized Rank')
        plt.ylabel('Cumulative Gain')
        plt.title('Cumulative Gain Curve')
        plt.legend(loc='best')

    def plot_mean_average_precision(self, map_values, labels):
        # Plot multiple MAP values
        for i, map_score in enumerate(map_values):
            plt.bar(labels[i], map_score, color=self.colors[i])
        plt.ylabel('MAP')
        plt.title('Mean Average Precision (MAP)')

    def plot_reciprocal_rank(self, reciprocal_rank_values, labels):
        # Plot RR values
        plt.bar(labels, reciprocal_rank_values, color=self.colors[0])
        plt.ylabel('Reciprocal Rank')
        plt.title('Reciprocal Rank (RR)')

    def show_plot(self):
        plt.show()
