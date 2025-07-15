import unittest
from ltsm.evaluate_pipeline.evaluation_pipeline import EvaluationPipeline
import random


class TestEvaluationPipeline(unittest.TestCase):
    def setUp(self):
        # Complex data for testing
        # Predicted ranges with overlaps, gaps, and exact matches
        self.x_test = [
            (0, 10), (15, 25), (30, 40), (45, 55), (60, 70)
        ]
        # True ranges with partial overlaps, non-overlapping ranges, and complete overlaps
        self.y_test = [
            (5, 15), (20, 30), (35, 50), (65, 75)
        ]

        # Large, random test set
        self.x_test_large = self.generate_random_ranges(100, 0, 1000)
        self.y_test_large = self.generate_random_ranges(100, 0, 1000)

    @staticmethod
    def generate_random_ranges(count, range_start, range_end):
        """Generate random range pairs."""
        ranges = []
        for _ in range(count):
            start = random.randint(range_start, range_end - 10)
            end = start + random.randint(5, 20)
            ranges.append((start, end))
        return ranges

    def test_overlap(self):
        # Test overlap size with varying ranges
        pipeline = EvaluationPipeline(self.x_test, self.y_test)

        # Overlap between first predicted and first true range
        overlap_1 = pipeline.overlap_size((0, 10), (5, 15))
        self.assertAlmostEqual(overlap_1, 0.545, places=2)  # Partial overlap

        overlap_2 = pipeline.overlap_size((17, 29), (23, 33))
        self.assertAlmostEqual(overlap_2, 0.5384615384615384, places=2)  # Partial overlap

        # Overlap between first predicted and first true range
        overlap_3 = pipeline.overlap_size((16, 20), (20, 35))
        self.assertAlmostEqual(overlap_3, 0.2, places=2)

        # No overlap
        overlap_4 = pipeline.overlap_size((15, 25), (35, 50))
        self.assertEqual(overlap_4, 0)

        # Complete overlap
        overlap_5 = pipeline.overlap_size((30, 40), (30, 40))
        self.assertEqual(overlap_5, 1.0)

    def test_cardinality_factor(self):
        # Test cardinality factor with overlapping ranges
        pipeline = EvaluationPipeline(self.x_test, self.y_test)

        # A range with overlaps
        cardinality_1 = pipeline.cardinality_factor((35, 50), self.y_test)
        self.assertGreaterEqual(cardinality_1, 1.0)

        # A range with no overlaps
        cardinality_2 = pipeline.cardinality_factor((100, 110), self.y_test)
        self.assertEqual(cardinality_2, 1.0)

    def test_large_random_data(self):
        # Test pipeline with large randomized data
        pipeline = EvaluationPipeline(self.x_test_large, self.y_test_large)

        recall = pipeline.evaluate_recall_score()
        precision = pipeline.evaluate_precision_score()
        f1_score = pipeline.evaluate_f1_score()

        # Check scores are within bounds
        self.assertGreaterEqual(recall, 0.0)
        self.assertGreaterEqual(precision, 0.0)
        self.assertGreaterEqual(f1_score, 0.0)
        self.assertLessEqual(recall, 1.0)
        self.assertLessEqual(precision, 1.0)
        self.assertLessEqual(f1_score, 1.0)

    def test_edge_case_empty_inputs(self):
        # Edge case: Empty inputs
        pipeline = EvaluationPipeline([], [])
        recall = pipeline.evaluate_recall_score()
        precision = pipeline.evaluate_precision_score()
        f1_score = pipeline.evaluate_f1_score()

        self.assertEqual(recall, 0.0)
        self.assertEqual(precision, 0.0)
        self.assertEqual(f1_score, 0.0)

    def test_edge_case_no_overlap(self):
        # Edge case: No overlaps
        x_test_no_overlap = [(0, 10), (20, 30)]
        y_test_no_overlap = [(40, 50), (60, 70)]
        pipeline = EvaluationPipeline(x_test_no_overlap, y_test_no_overlap)
        recall = pipeline.evaluate_recall_score()
        precision = pipeline.evaluate_precision_score()
        f1_score = pipeline.evaluate_f1_score()

        self.assertEqual(recall, 0.0)
        self.assertEqual(precision, 0.0)
        self.assertEqual(f1_score, 0.0)

    def test_f1_score_consistency(self):
        # Validate that F1 score is consistent with precision and recall
        pipeline = EvaluationPipeline(self.x_test, self.y_test)
        recall = pipeline.evaluate_recall_score()
        precision = pipeline.evaluate_precision_score()
        f1_score = pipeline.evaluate_f1_score()

        if precision + recall > 0:
            self.assertAlmostEqual(f1_score, 2 * (precision * recall) / (precision + recall), places=2)
        else:
            self.assertEqual(f1_score, 0.0)


if __name__ == "__main__":
    unittest.main()
