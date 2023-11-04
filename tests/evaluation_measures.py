import time
import resource
from sklearn.metrics import precision_score, recall_score, f1_score


class InformationRetrievalEvaluator:
    def __init__(self, ranking_results, relevance_labels):
        # List of ranked documents for each query
        self.ranking_results = ranking_results
        # Ground truth relevance labels for each query
        self.relevance_labels = relevance_labels

    def evaluate_precision(self):
        # Calculate precision for each query
        precisions = []
        for query_results, relevance in zip(self.ranking_results, self.relevance_labels):
            precision = precision_score(
                relevance, query_results, average='micro')
            precisions.append(precision)
        return precisions

    def evaluate_recall(self):
        # Calculate recall for each query
        recalls = []
        for query_results, relevance in zip(self.ranking_results, self.relevance_labels):
            recall = recall_score(relevance, query_results, average='micro')
            recalls.append(recall)
        return recalls

    def evaluate_f1_score(self):
        # Calculate F1 score for each query
        f1_scores = []
        for query_results, relevance in zip(self.ranking_results, self.relevance_labels):
            f1 = f1_score(relevance, query_results, average='micro')
            f1_scores.append(f1)
        return f1_scores

    def evaluate_execution_time(self, function):
        # Measure execution time of a specific function
        start_time = time.time()
        function()
        end_time = time.time()
        return end_time - start_time

    def evaluate_memory_usage(self, function):
        # Measure memory usage of a specific function
        start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        function()
        end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return end_memory - start_memory


# Example usage:
# Instantiate the evaluator with ranking results and relevance labels
ranking_results = [[1, 3, 2], [2, 1, 3], [3, 2, 1]]
relevance_labels = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
evaluator = InformationRetrievalEvaluator(ranking_results, relevance_labels)

# Evaluate precision, recall, and F1 score for the ranking results
precisions = evaluator.evaluate_precision()
recalls = evaluator.evaluate_recall()
f1_scores = evaluator.evaluate_f1_score()

# Evaluate execution time and memory usage of a specific function
execution_time = evaluator.evaluate_execution_time(lambda: time.sleep(2))
memory_usage = evaluator.evaluate_memory_usage(lambda: [0] * int(1e7))

print("Precision:", precisions)
print("Recall:", recalls)
print("F1 Score:", f1_scores)
print("Execution Time (s):", execution_time)
print("Memory Usage (KB):", memory_usage)
