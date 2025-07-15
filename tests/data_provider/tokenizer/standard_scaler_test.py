from ltsm.data_provider.tokenizer.standard_scaler import StandardScaler

import numpy as np
import pytest
import os

@pytest.fixture
def setup():
    processor = StandardScaler()
    train_data = [np.array([x*i for i in range(100)]) for x in [1, 100, 10000]]
    val_data = [np.array([x*i for i in range(100)]) for x in [1, 100, 10000]]
    test_data = [np.array([x*i for i in range(100)]) for x in [1, 100, 10000]]
    raw_data = [np.concatenate((train_data[x], val_data[x], test_data[x])) for x in range(3)]

    new_train, new_val, new_test = processor.process(raw_data, train_data, val_data, test_data, fit_train_only=True)
    return new_train, new_val, new_test, train_data, val_data, test_data, raw_data, processor

def test_standard_scaler_process_on_train_only(setup):
    new_train, new_val, new_test, train_data, val_data, test_data, raw_data, processor = setup

    assert len(new_train) == len(train_data)
    assert len(new_val) == len(val_data)
    assert len(new_test) == len(test_data)

    means = [np.mean(train_data[i]) for i in range(3)]
    stds = [np.std(train_data[i]) for i in range(3)]
    for i in range(3):
        assert new_train[i].shape == train_data[i].shape
        assert new_val[i].shape == val_data[i].shape
        assert new_test[i].shape == test_data[i].shape
        for j in range(100):
            assert new_train[i][j] == (train_data[i][j] - means[i]) / stds[i]
            assert new_val[i][j] == (val_data[i][j] - means[i]) / stds[i]
            assert new_test[i][j] == (test_data[i][j] - means[i]) / stds[i]

def test_standard_scaler_process(setup):
    new_train, new_val, new_test, train_data, val_data, test_data, raw_data, processor = setup

    assert len(new_train) == len(train_data)
    assert len(new_val) == len(val_data)
    assert len(new_test) == len(test_data)

    means = [np.mean(raw_data[i]) for i in range(3)]
    stds = [np.std(raw_data[i]) for i in range(3)]
    for i in range(3):
        assert new_train[i].shape == train_data[i].shape
        assert new_val[i].shape == val_data[i].shape
        assert new_test[i].shape == test_data[i].shape
        for j in range(100):
            assert new_train[i][j] == (train_data[i][j] - means[i]) / stds[i]
            assert new_val[i][j] == (val_data[i][j] - means[i]) / stds[i]
            assert new_test[i][j] == (test_data[i][j] - means[i]) / stds[i]

def test_standard_scaler_save(tmp_path, setup):
    d = tmp_path / "save_dir"
    d.mkdir()
    new_train, new_val, new_test, train_data, val_data, test_data, raw_data, processor = setup
    processor.save(str(d))
    assert os.path.isfile(f"{str(d)}/processor.pkl")

def test_standard_scaler_load(tmp_path, setup):
    d = tmp_path / "save_dir"
    d.mkdir()
    new_train, new_val, new_test, train_data, val_data, test_data, raw_data, processor = setup
    processor.save(str(d))
    processor._scaler = None
    processor.load(str(d))
    assert processor is not None