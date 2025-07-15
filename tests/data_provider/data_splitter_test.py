from ltsm.data_provider.data_splitter import SplitterByTimestamp
import pandas as pd
import numpy as np
import pytest
import math

def test_splitter_by_timestamp_get_csv_splits():
    indices = ["cosine", "linear", "exponential"]
    test_df = pd.DataFrame([[math.cos(i) for i in range(100)],
                            [2*i for i in range(100)],
                            [math.exp(i) for i in range(100)]], 
                            index=indices)
    splitter = SplitterByTimestamp(seq_len=5, 
                                   pred_len=1,
                                   train_ratio=0.7,
                                   val_ratio=0.1)
    train, val, test, buff = splitter.get_csv_splits(test_df)
    assert len(train) == 3
    assert len(val) == 3
    assert len(test) == 3
    assert len(buff) == 3
    assert buff == indices
    for i in range(3):
        assert len(train[i]) == 70
        assert len(val[i]) == 15
        assert len(test[i]) == 25

def test_splitter_by_timestamp_get_csv_splits_invalid_ndim():
    test_df = pd.DataFrame([np.array([1, 2, 3]), np.array([[4, 5], [6, 7], [8, 9]])])
    splitter = SplitterByTimestamp(seq_len=5, 
                                   pred_len=1,
                                   train_ratio=0.7,
                                   val_ratio=0.1)
    with pytest.raises(ValueError):
        train, val, test, buff = splitter.get_csv_splits(test_df)