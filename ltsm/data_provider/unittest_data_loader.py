import unittest
import torch
import numpy as np
from ltsm.data_provider.data_loader import get_weighted_sampling_data_loader
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class TestWeightedSampler(unittest.TestCase):
    def setUp(self):
        self.num_samples = 1000
        self.time_steps = 10
        self.features = 5
        self.data = np.random.randn(self.num_samples, self.time_steps, self.features).astype(np.float32)
        self.labels = np.random.choice([0, 1], size=self.num_samples, p=[0.3, 0.7])

        self.class_weights = {0: 0.8, 1: 0.2}
        self.weights = np.array([self.class_weights[label] for label in self.labels])
        self.weights=self.weights/np/sum(self.weights)
        
        self.data_tensor = torch.tensor(self.data)
        self.labels_tensor = torch.tensor(self.labels)

        self.dataset = TimeSeriesDataset(self.data_tensor, self.labels_tensor)
        self.batch_size = 32
        self.withreplacement=True
        self.loader = get_weighted_sampling_data_loader(self.dataset, batch_size=self.batch_size,weights = self.weights, withreplacement=self.withreplacement)

    def test_sampler_weighted_proportions(self):
        sampled_label_counts = {0: 0, 1: 0}
        total_samples = 0

        for _, batch_labels in self.dataloader:
            for label in batch_labels.numpy():
                sampled_label_counts[label] += 1
            total_samples += len(batch_labels)
            
            if total_samples >= 1000:  
                break

        total_count = sum(sampled_label_counts.values())
        observed_proportions = {
            label: count / total_count for label, count in sampled_label_counts.items()
        }

        total_weight = sum(self.class_weights.values())
        expected_proportions = {
            label: weight / total_weight for label, weight in self.class_weights.items()
        }

        for label in expected_proportions:
            self.assertAlmostEqual(
                observed_proportions[label],
                expected_proportions[label],
                delta=0.1,  
                msg=f"Label {label} has incorrect sampling proportion."
            )

if __name__ == "__main__":
    unittest.main()

