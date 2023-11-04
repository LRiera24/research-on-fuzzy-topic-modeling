import time
import resource
from sklearn.metrics import precision_score, recall_score, f1_score

class InformationRetrievalEvaluator:
    def __init__(self, queries, relevance_labels, ir_system):
        self.queries = queries  # List of query strings
        self.relevance_labels = relevance_labels  # List of true ranked results per query
       
        self.ir_system = ir_system  # Information Retrieval system
        self.ranking_results = []   # List of obtained ranked results per query

        self.execution_times = []  # List to store execution times
        self.memory_usages = []  # List to store memory usages
        self.precisions = []  # List to store precision values
        self.recalls = []  # List to store recall values
        self.f1_scores = []  # List to store F1 scores

    def evaluate_ir_system(self):
        for query in self.queries:
            execution_time, memory_usage = self.measure_execution(query)
            precision, recall, f1 = self.evaluate_ranking(self.relevance_labels, self.ranking_results)
            
            # Store the measurements in the lists
            self.execution_times.append(execution_time)
            self.memory_usages.append(memory_usage)
            self.precisions.append(precision)
            self.recalls.append(recall)
            self.f1_scores.append(f1)

    def measure_execution(self, query):
        start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        start_time = time.time()
        # Call your IR system with the query and get ranking results
        self.ranking_results = self.ir_system(query)
        end_time = time.time()
        end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory

        return execution_time, memory_usage

    def evaluate_ranking(self, relevance_labels, ranking_results):
        # Evaluate the ranking results using precision, recall, and F1 score
        # Replace the relevance_labels with the ground truth relevance labels
        precision = precision_score(relevance_labels, ranking_results, average='micro')
        recall = recall_score(relevance_labels, ranking_results, average='micro')
        f1 = f1_score(relevance_labels, ranking_results, average='micro')

        return precision, recall, f1
