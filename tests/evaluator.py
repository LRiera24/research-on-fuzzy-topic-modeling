from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
import resource
import time


class Evaluator:
    def __init__(self, ir_system_component):
        self.ir_system_component = ir_system_component

    def execution(self, query):
        start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        start_time = time.time()
        # Call your IR system with the query and get ranking results
        self.ranking_results = self.ir_system(query)
        end_time = time.time()
        end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory

        return execution_time, memory_usage


class TextClassificationEvaluator(Evaluator):
    def __init__(self, ir_system_component, true_labels, predicted_labels):
        super().__init__(ir_system_component)
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels

    def accuracy(self):
        return accuracy_score(self.true_labels, self.predicted_labels)

    def precision(self):
        return precision_score(self.true_labels, self.predicted_labels)

    def recall(self):
        return recall_score(self.true_labels, self.predicted_labels)

    def f1_score(self):
        return f1_score(self.true_labels, self.predicted_labels)

    def confusion_matrix(self):
        return confusion_matrix(self.true_labels, self.predicted_labels)

    def cohen_kappa(self):
        return cohen_kappa_score(self.true_labels, self.predicted_labels)

    def roc_auc(self, probabilities, positive_label=1):
        fpr, tpr, _ = roc_curve(
            self.true_labels, probabilities, pos_label=positive_label)
        return auc(fpr, tpr)


class InformationRetrievalEvaluator(Evaluator):
    def __init__(self, ir_system_component, relevant_documents, retrieved_documents):
        super().__init__(ir_system_component)
        self.relevant_documents = relevant_documents
        self.retrieved_documents = retrieved_documents

    def precision_at_k(self, k):
        retrieved_k = self.retrieved_documents[:k]
        relevant_k = set(self.relevant_documents).intersection(retrieved_k)
        return len(relevant_k) / k

    def recall_at_k(self, k):
        retrieved_k = self.retrieved_documents[:k]
        relevant_k = set(self.relevant_documents).intersection(retrieved_k)
        return len(relevant_k) / len(self.relevant_documents)

    def f1_score_at_k(self, k):
        precision = self.precision_at_k(k)
        recall = self.recall_at_k(k)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    def average_precision(self):
        precision_sum = 0
        relevant_count = 0
        for i, doc in enumerate(self.retrieved_documents):
            if doc in self.relevant_documents:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        if not relevant_count:
            return 0
        return precision_sum / relevant_count

    def mean_average_precision(self, queries_and_results):
        map_sum = 0
        for relevant_documents, retrieved_documents in queries_and_results:
            evaluator = InformationRetrievalEvaluator(
                relevant_documents, retrieved_documents)
            map_sum += evaluator.average_precision()
        return map_sum / len(queries_and_results)
