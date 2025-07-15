import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler as SKStandardScaler

from ltsm.common.base_processor import BaseProcessor
from typing import Tuple, List


class StandardScaler(BaseProcessor):
    """
    Represents a Standard Scaler object that uses Sklearn's Standard Scaler for data processing.

    Attributes:
        module_id (str): The identifier for base processor objects.
    """
    module_id = "standard_scaler"
    
    def __init__(self):
        self._scaler = None

    def process(self, raw_data: np.ndarray, train_data: List[np.ndarray], val_data: List[np.ndarray], test_data: List[np.ndarray], fit_train_only:bool=False, do_anomaly:bool=False)->Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Standardizes the training, validation, and test sets by removing the mean and scaling to unit variance.

        Args:
            raw_data (np.ndarray): The raw data.
            train_data (List[np.ndarray]): The list of training sequences.
            val_data (List[np.ndarray]): The list of validation sequences.
            test_data (List[np.ndarray]): The list of test sequences.
            fit_train_only (bool): Indicates whether the datasets should be scaled based on the training data.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                A tuple of three lists containing the processed training, validation, and test data. 
        """
        scaled_train_data, scaled_val_data, scaled_test_data = [], [], []
        for i, (raw_sequence, train_sequence, val_sequence, test_sequence) in enumerate(zip(
            raw_data,
            train_data,
            val_data,
            test_data,
        )):
            if do_anomaly and i == len(raw_data) - 1: # Skip anomaly label
                scaled_train_data.append(train_sequence)
                scaled_val_data.append(val_sequence)
                scaled_test_data.append(test_sequence)
                continue

            train_sequence = train_sequence.reshape(-1, 1)
            val_sequence = val_sequence.reshape(-1, 1)
            test_sequence = test_sequence.reshape(-1, 1)

            self._scaler = SKStandardScaler()

            if fit_train_only:
                self._scaler.fit(train_sequence)
            else:
                self._scaler.fit(raw_sequence.reshape(-1, 1))

            scaled_train_data.append(self._scaler.transform(train_sequence).flatten())
            scaled_val_data.append(self._scaler.transform(val_sequence).flatten())
            scaled_test_data.append(self._scaler.transform(test_sequence).flatten())

        return scaled_train_data, scaled_val_data, scaled_test_data

    def inverse_process(self, data: np.ndarray)->np.ndarray:
        """
        Scales back the data to its original representation.

        Args:
            data (np.ndarray): The data to scale back.

        Returns:
            np.ndarray: The scaled back data.
        """
        assert self._scaler is not None, "StandardScaler has not been fitted"
        raw_shape = data.shape
        data = self._scaler.inverse_transform(data.reshape(-1, 1))

        return data.reshape(raw_shape)

    def save(self, save_dir: str):
        """
        Saves the scaler to the save_dir directory as a Pickle file named processor.pkl.

        Args:
            save_dir (str): The directory where to store the scaler.
        """
        save_path = os.path.join(save_dir, "processor.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(self._scaler, f)

    def load(self, save_dir):
        """
        Loads the scaler saved at the save_dir directory.

        Args:
            save_dir (str): The directory the scaler was saved.
        """
        save_path = os.path.join(save_dir, "processor.pkl")
        with open(save_path, 'rb') as f:
            self._scaler = pickle.load(f)

