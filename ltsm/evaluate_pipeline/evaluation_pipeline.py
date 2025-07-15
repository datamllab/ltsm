import numpy as np
import pandas as pd
import os

class EvaluationPipeline:
    def __init__(self, x_test, y_test,  bias_function="flat", gamma_function=None):
        """
        Initialize the evaluation pipeline with a trained model and user-defined bias.

        Parameters:
        x_test - Predicted values from the model (should be a list of tuples, the start and end of the anomaly range, the data type should be int)
        y_test - True labels for the test set (should be a list of tuples, the start and end of the anomaly range, the data type should be int)

        bias_function - The bias function to use ('flat', 'front_end', 'back_end', 'middle')
        gamma_function - The gamma function to use for the cardinality factor, default is 1/overlap_count
        """
        self.x_test = x_test
        self.y_test = y_test
        self.bias_function = self.get_bias_function(bias_function)
        self.gamma_function = gamma_function if gamma_function else self.default_gamma
        self.results = {}
        self.recall_score = 0
        self.precision_score = 0
        self.f1_score = 0

    def get_bias_function(self, bias_type):
        """Return the selected bias function."""
        bias_functions = {
            "flat": self.flat_bias,
            "front_end": self.front_end_bias,
            "back_end": self.back_end_bias,
            "middle": self.middle_bias,
        }
        return bias_functions.get(bias_type, self.flat_bias)

    @staticmethod
    def flat_bias(i, anomaly_length):
        """Flat bias function."""
        return 1

    @staticmethod
    def front_end_bias(i, anomaly_length):
        """Front-end bias function."""
        return -i + 1

    @staticmethod
    def back_end_bias(i, anomaly_length):
        """Back-end bias function."""
        return i

    @staticmethod
    def middle_bias(i, anomaly_length):
        """Middle bias function."""
        if i < anomaly_length / 2:
            return i
        else:
            return anomaly_length - i + 1

    @staticmethod
    def default_gamma(overlap_count):
        """Default gamma function for cardinality factor."""
        return 1 / overlap_count

    def overlap_size(self, range1, range2):
        """
        Compute the overlap reward ω.
        range1 - (start, end) of one range (true or predicted)
        range2 - (start, end) of the other range

        Returns:
        Overlap reward for the two ranges
        """
        overlap_start = max(range1[0], range2[0])
        overlap_end = min(range1[1], range2[1])
        if overlap_start > overlap_end:
            return 0  # No overlap

        anomaly_length = range1[1] - range1[0] + 1

        my_value = 0
        max_value = 0
        for i in range(1, anomaly_length + 1):
            bias = self.bias_function(i, anomaly_length)
            max_value += bias
            if overlap_start <= range1[0] + i - 1 <= overlap_end:
                my_value += bias

        return my_value / max_value

    def cardinality_factor(self, target_range, comparison_ranges):
        """
        Compute the CardinalityFactor.
        target_range - (start, end) of the range (true or predicted)
        comparison_ranges - List of ranges to compare with (true or predicted)

        Returns:
        Cardinality factor for the target range
        """
        overlap_count = sum(
            1 for comparison_range in comparison_ranges
            if max(target_range[0], comparison_range[0]) <= min(target_range[1], comparison_range[1])
        )
        if overlap_count <= 1:
            return 1
        else:
            return self.gamma_function(overlap_count)

    def recall_t(self, Ri, P, alpha = 0):
        """
        Compute RecallT(Ri, P).
        Ri - A single true anomaly range
        P - List of predicted anomaly ranges
        alpha - Weight for existence reward (default is 0)

        Returns:
        Recall_t - Recall score for a single true range
        """
        # Existence Reward
        total_overlap = sum(
            max(0, min(Ri[1], Pj[1]) - max(Ri[0], Pj[0]) + 1)
            for Pj in P
        )
        existence_reward = 1 if total_overlap >= 1 else 0

        # Overlap Reward
        overlap_reward = self.cardinality_factor(Ri, P) * sum(
            self.overlap_size(Ri, Pj) for Pj in P
        )

        return alpha * existence_reward + (1 - alpha) * overlap_reward

    def precision_t(self, R, Pi):
        """
        Compute PrecisionT(R, Pi).
        R - List of true anomaly ranges
        Pi - A single predicted anomaly range

        Returns:
        precision_t - Precision score for a single predicted range
        """
        # Overlap Reward
        overlap_reward = self.cardinality_factor(Pi, R) * sum(
            self.overlap_size(Pi, Rj) for Rj in R
        )

        return overlap_reward

    def evaluate_recall_score(self):
        """
        Evaluate the model on test data and store the results in the class instance.

        Parameters:
        x_test - Predicted values from the model
        y_test - True labels for the test set

        Returns:
        Range-based recall score
        """
        Nr = len(self.y_test)
        if Nr == 0:
            return 0
        else:
            recall_score = sum(self.recall_t(Ri, self.x_test) for Ri in self.y_test) / Nr
            self.recall_score = recall_score
            return recall_score


    def evaluate_precision_score(self):
        """
        Evaluate the model on test data and store the results in the class instance.

        Parameters:
        x_test - Predicted values from the model
        y_test - True labels for the test set

        Returns:
        Range-based precision score
        """
        Np = len(self.x_test)
        if Np == 0:
            return 0
        else:
            precision_score = sum(self.precision_t(self.y_test, Pi) for Pi in self.x_test) / Np
            self.precision_score = precision_score
            return precision_score

    def evaluate_f1_score(self):
        """
        Evaluate the model on test data and store the results in the class instance.

        Parameters:
        x_test - Predicted values from the model
        y_test - True labels for the test set

        Returns:
        Range-based f1 score
        """
        recall = self.evaluate_recall_score()
        precision = self.evaluate_precision_score()
        if precision + recall == 0:
            self.f1_score = 0
            return 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
            self.f1_score = f1_score
            return f1_score


    def evaluate(self,dataset_name='',model_name=''):
        """
        Evaluate the model on test data and store the results in the class instance.

        Parameters:
        dataset_name - Name of the dataset
        model_name - Name of the model

        Returns:
        Dictionary of evaluation results
        """
        recall = self.evaluate_recall_score()
        precision = self.evaluate_precision_score()
        if precision + recall == 0:
            self.f1_score = 0

        else:
            self.f1_score = 2 * (precision * recall) / (precision + recall)

        # Store results
        self.results = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "precision": self.precision_score,
            "recall": self.recall_score,
            "f1_score": self.f1_score
        }
        print("Evaluation results:", self.results)
        print("Use 'save_results()' method to log the results to a CSV file.")
        return self.results

    def get_results(self):
        """
        Get the evaluation results.

        Returns:
        Dictionary of evaluation results
        """
        return self.results

    def save_results(self, csv_path="./evaluation_result.csv"):
        """
        Log the evaluation results to a CSV file. If the file doesn't exist, it creates a new one;
        if it exists, it appends the new results.

        Parameters:
        csv_path - Path to the CSV file for logging (default is "evaluation_log.csv")

        Returns:
        Updated DataFrame with the appended results
        """
        new_results_df = pd.DataFrame([self.results])

        # Check if the CSV file exists
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            updated_df = pd.concat([existing_df, new_results_df], ignore_index=True)
        else:
            updated_df = new_results_df
        updated_df.to_csv(csv_path, index=False)

        return updated_df

