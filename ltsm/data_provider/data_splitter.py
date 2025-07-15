from ltsm.common.base_splitter import DataSplitter
import pandas as pd
import numpy as np

from typing import Tuple, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class SplitterByTimestamp(DataSplitter):
    """
    Data splitter class that splits time-series data by timestamp.
    """
    def __init__(self, seq_len: int, pred_len: int, train_ratio: float, val_ratio: float):
        """
        Initializes the SplitterByTimestamp with the given arguments.

        Args:
            seq_len (int): The number of timesteps used in the input sequence.
            pred_len (int): The number of timesteps the model should predict for the output sequence.
            train_ratio (float): The training set ratio.
            val_ratio (float): The validation set ratio.
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio


    def get_csv_splits(self, df_data: pd.DataFrame, do_anomaly: bool=False) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Splits the .csv data into training-validation-training sets.

        Args:
            df_data (pd.DataFrame): A Pandas DataFrame containing the data to be split.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
                A tuple containing fours lists of sequences for the training, validation, and test sets. 
                The last list contains the row labels of these sequences.
        """
        train_split, val_split, test_split, buff = [], [], [], []
        raw_data = df_data.to_numpy()

        for index, sequence in zip(df_data.index, raw_data):
            if len(sequence) > 0 and isinstance(sequence[0], np.ndarray):
                logging.error("Time-series should be 1D.")
                raise ValueError("Time-series should be 1D.")

            num_train = int(len(sequence) * self.train_ratio)
            num_val = int(len(sequence) * self.val_ratio)
            
            if not do_anomaly:
                if num_train < self.seq_len + self.pred_len:
                    continue
            else:
                if num_train < self.seq_len:
                    continue
            
            
            # We also add the previous seq_len points to the val and test sets
            train_split.append(sequence[:num_train])
            val_split.append(sequence[num_train-self.seq_len:num_train+num_val])
            test_split.append(sequence[num_train+num_val-self.seq_len:])
            buff.append(index)

        return train_split, val_split, test_split, buff
