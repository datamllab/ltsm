import pandas as pd
import numpy as np
import pytest
import torch
import math
import os
from ltsm.data_provider.dataset import TSPromptDataset, TSTokenDataset
from ltsm.data_provider.data_factory import DatasetFactory
from unittest.mock import call

@pytest.fixture
def setup(tmp_path):
    d = tmp_path / "mock"
    d.mkdir()
    data_path = d / "mock.csv"

    d = tmp_path / "prompt_bank"
    d.mkdir()
    prompt_data_path = d / "prompt_data_normalize_split"
    prompt_data_path.mkdir()

    prompt_data_folder = prompt_data_path / "mock"
    prompt_data_folder.mkdir()
    return data_path, prompt_data_path, prompt_data_folder, DatasetFactory(
        data_paths=[str(data_path)],
        prompt_data_path=str(prompt_data_path),
        data_processing="standard_scaler",
        seq_len=10,
        pred_len=1,
        train_ratio=0.7,
        val_ratio=0.1,
        model="",
        scale_on_train=True,
        do_anomaly=False
    )

# Save functions for threes different formats
def save_csv(file_folder, data):
    file_path = os.path.join(file_folder, 'mock_index_prompt.csv')
    data.to_csv(file_path, index=False)
    return file_path

def save_pth(file_folder, data):
    file_path = os.path.join(file_folder, 'mock_index_prompt.pth.tar')
    torch.save(data, file_path)
    return file_path

def save_npz(file_folder, data):
    file_path = os.path.join(file_folder, 'mock_index_prompt.npz')
    np.savez(file_path, data=data.values, index=data.index, columns=data.columns)
    return file_path

@pytest.mark.parametrize("save_function, expected_shape", [
    (save_csv, (133,)), 
    (save_pth, (133,)), 
    (save_npz, (133,))
])
def test_data_factory__get_prompt(setup, save_function, expected_shape):
    data_path, prompt_data_path, prompt_data_folder, datasetFactory = setup
    data = pd.DataFrame([range(133)])
    save_function(str(prompt_data_folder), data)
    print("prompt_data_folder", prompt_data_folder)
    print("prompt_data_path", prompt_data_path)
    print("data_path", data_path)
    prompt_data = datasetFactory._DatasetFactory__get_prompt(str(prompt_data_path), str(data_path), "index")
    assert len(prompt_data) == expected_shape[0]
    arr = np.random.rand(366)
    concatenated_result = np.concatenate((prompt_data, arr))
    assert len(concatenated_result) == expected_shape[0] + len(arr)

def test_data_factory_load_prompts_invalid_prompt_file(setup):
    data_path, prompt_data_path, prompt_data_folder, datasetFactory = setup

    buff = [i for i in range(4)]
    for index in enumerate(buff):
        p = prompt_data_folder / f"mock_{index}_prompt.pth.tar"
        p.write_text("not valid .pth file content")

    prompt_data, _ = datasetFactory.loadPrompts(str(data_path), str(prompt_data_path), buff)
    assert len(prompt_data) == 4
    for i in range(4):
        assert len(prompt_data[i]) == 0

def test_data_factory_load_csv_prompts(setup):
    data_path, prompt_data_path, prompt_data_folder, datasetFactory = setup
    buff = ["linear", "quadratic", "exponential", "sine"]

    data = np.array([
        [2*i for i in range(100)],
        [i**2 for i in range(100)],
        [math.exp(i) for i in range(100)],
        [math.sin(i) for i in range(100)]
    ])
    train_buf = pd.DataFrame(data, index=buff)

    for i, filename in enumerate(buff):
        p = prompt_data_folder / f"mock_{filename}_prompt.pth.tar"
        train_prompt = pd.DataFrame(train_buf.iloc[i].values)
        train_prompt = train_prompt.T
        assert train_prompt.shape == (1, 100)
        torch.save(train_prompt, str(p))
        
    prompt_data, _ = datasetFactory.loadPrompts(str(data_path), str(prompt_data_path), buff)
    assert len(prompt_data) == 4
    prompt_data_buf = pd.DataFrame(np.array(prompt_data), index=buff)
    assert prompt_data_buf.equals(train_buf)
    

def test_data_factory_load_monash_prompts(setup):
    data_path, prompt_data_path, prompt_data_folder, datasetFactory = setup

    # Create a monash file
    d = data_path.parent / "monash"
    d.mkdir()
    data_path = d / "mock.tsf"
    buff = [i for i in range(4)]

    data = np.array([
        [2*i for i in range(100)],
        [i**2 for i in range(100)],
        [math.exp(i) for i in range(100)],
        [math.sin(i) for i in range(100)]
    ])
    train_buf = pd.DataFrame(data, index=buff)

    for index in buff:
        p = prompt_data_folder / f"T{index+1}_prompt.pth.tar"
        train_prompt = pd.DataFrame(train_buf.iloc[index].values)
        train_prompt = train_prompt.T
        assert train_prompt.shape == (1, 100)
        torch.save(train_prompt, str(p))
        
    prompt_data, _ = datasetFactory.loadPrompts(str(data_path), str(prompt_data_path), buff)
    assert len(prompt_data) == 4
    prompt_data_buf = pd.DataFrame(np.array(prompt_data), index=buff)
    assert prompt_data_buf.equals(train_buf)

def test_data_factory_load_word_prompts(setup):
    data_path, prompt_data_path, prompt_data_folder, datasetFactory = setup
    buff = [i for i in range(4)]

    # Change model to 'LTSM_WordPrompt'
    datasetFactory.model = "LTSM_WordPrompt"
        
    prompt_data, _ = datasetFactory.loadPrompts(str(data_path), str(prompt_data_path), buff)
    assert len(prompt_data) == 4
    for i in range(4):
        assert len(prompt_data[i]) == 1
        assert prompt_data[i][0] == 0

def test_data_factory_load_missing_prompts(setup):
    data_path, prompt_data_path, prompt_data_folder, datasetFactory = setup
    buff = ['one', 'two', 'three']
    prompt_indices = ['one', 'two']
    # Mock data for mock pth file
    train_buf = pd.DataFrame(pd.DataFrame(np.zeros((5, 133)), columns=[i for i in range(133)]))

    for i, filename in enumerate(prompt_indices):
        p = prompt_data_folder / f"mock_{filename}_prompt.pth.tar"
        train_prompt = pd.DataFrame(train_buf.iloc[i].values)
        train_prompt = train_prompt.T
        assert train_prompt.shape == (1, 133)
        torch.save(train_prompt, str(p))

    prompt_data, missing = datasetFactory.loadPrompts(str(data_path), str(prompt_data_path), buff)
    assert len(missing) == 1
    assert missing[0] == 'three'
    assert len(prompt_data) == 3
    assert len(prompt_data[0]) == 133
    assert len(prompt_data[1]) == 133
    assert len(prompt_data[2]) == 0
    

def test_data_factory_createTorchDS_empty(setup):
    data_path, prompt_data_path, prompt_data_folder, datasetFactory = setup
    dataset = datasetFactory.createTorchDS([], [], 1, False)
    assert dataset is None

def test_data_factory_createTorchDS(setup):
    data_path, prompt_data_path, prompt_data_folder, datasetFactory = setup
    seq_len = 10
    pred_len = 1
    prompt_len = 10
    data = [np.array([2*i for i in range(100)])]
    prompt_data = [[0.1*i for i in range(10)]]
    dataset = datasetFactory.createTorchDS(data, prompt_data, 1, False)
    assert type(dataset) == TSPromptDataset
    for i in range(len(dataset)):
        x = dataset[i][0]
        y = dataset[i][1]
        # Assert that the first prompt_len data points in a row is the prompt
        np.testing.assert_equal(x.numpy()[:prompt_len], np.array(prompt_data[0]).reshape(prompt_len, 1))

        # Assert that the next seq_len data points in a row matches data
        np.testing.assert_equal(x.numpy()[prompt_len:prompt_len+seq_len], data[0][i:i+seq_len].reshape(seq_len, 1))

        # Assert that the row prediction matches the next pred_len in data
        np.testing.assert_equal(y.numpy()[:pred_len], data[0][i+seq_len:i+seq_len+pred_len].reshape(pred_len, 1))


def test_data_factory_createTorchDS_tokenizer(setup):
    data_path, prompt_data_path, prompt_data_folder, datasetFactory = setup
    # Set model to 'LTSM_Tokenizer'
    datasetFactory.model = "LTSM_Tokenizer"
    data = [np.array([2*i for i in range(100)])]
    prompt_data = [[0.1*i for i in range(10)]]
    dataset = datasetFactory.createTorchDS(data, prompt_data, 1, False)
    assert type(dataset) == TSTokenDataset


def test_data_factory_getDatasets(mocker, setup):
    data_path, prompt_data_path, prompt_data_folder, datasetFactory = setup

    datasetFactory.data_paths = [
        str(data_path.parent / "fake1.csv"),
        str(data_path.parent / "fake2.csv"),
        str(data_path.parent / "fake3.csv"),
        str(data_path.parent / "fake4.csv")
    ]
    mock_ndarray = np.array([])
    mock_df = pd.DataFrame([])
    mock_torch_ds = TSPromptDataset([], [], 0, 0)
    mocker.patch.object(datasetFactory, 'fetch', return_value=mock_df)
    mocker.patch.object(datasetFactory.splitter, 'get_csv_splits', return_value=([],[],[],[]))
    mocker.patch.object(datasetFactory.processor, 'process', return_value=([mock_ndarray], [mock_ndarray], [mock_ndarray]))
    mocker.patch.object(datasetFactory, 'loadPrompts', return_value=([], []))
    mocker.patch.object(datasetFactory, 'createTorchDS', return_value=mock_torch_ds)

    train_dataset, val_dataset, test_datasets = datasetFactory.getDatasets()

    assert len(train_dataset) == 0
    assert len(val_dataset) == 0
    assert len(test_datasets) == 4
    for i in range(4):
        assert len(test_datasets[i]) == 0

    expected_calls = [call(data_path) for data_path in datasetFactory.data_paths]
    datasetFactory.fetch.assert_has_calls(expected_calls)

    assert datasetFactory.splitter.get_csv_splits.call_count == 4
    assert datasetFactory.processor.process.call_count == 4
    assert datasetFactory.loadPrompts.call_count == 12
    # createTorchDS should be called once each to make train_dataset and val_dataset
    # createTorchDS should be called four more times for each separate test_dataset
    assert datasetFactory.createTorchDS.call_count == 6

def test_data_factory_getDatasets_testset_no_split(mocker, setup):
    data_path, prompt_data_path, prompt_data_folder, datasetFactory = setup

    datasetFactory.data_paths = [
        str(data_path.parent / "fake1.csv"),
        str(data_path.parent / "fake2.csv"),
        str(data_path.parent / "fake3.csv"),
        str(data_path.parent / "fake4.csv")
    ]

    # Set split_test_sets to False
    datasetFactory.split_test_sets = False
    
    mock_ndarray = np.array([])
    mock_df = pd.DataFrame([])
    mock_torch_ds = TSPromptDataset([], [], 0, 0)
    mocker.patch.object(datasetFactory, 'fetch', return_value=mock_df)
    mocker.patch.object(datasetFactory.splitter, 'get_csv_splits', return_value=([],[],[],[]))
    mocker.patch.object(datasetFactory.processor, 'process', return_value=([mock_ndarray], [mock_ndarray], [mock_ndarray]))
    mocker.patch.object(datasetFactory, 'loadPrompts', return_value=([], []))
    mocker.patch.object(datasetFactory, 'createTorchDS', return_value=mock_torch_ds)

    train_dataset, val_dataset, test_datasets = datasetFactory.getDatasets()

    assert len(train_dataset) == 0
    assert len(val_dataset) == 0
    assert len(test_datasets) == 1
    assert len(test_datasets[0]) == 0

    expected_calls = [call(data_path) for data_path in datasetFactory.data_paths]
    datasetFactory.fetch.assert_has_calls(expected_calls)

    assert datasetFactory.splitter.get_csv_splits.call_count == 4
    assert datasetFactory.processor.process.call_count == 4
    assert datasetFactory.loadPrompts.call_count == 12
    assert datasetFactory.createTorchDS.call_count == 3

def test_data_factory_getDatasets_missing_prompts(mocker, setup):
    data_path, prompt_data_path, prompt_data_folder, datasetFactory = setup
    datasetFactory.data_paths = [
        str(data_path.parent / "fake1.csv")
    ]

    datasetFactory.downsample_rate = 1

    # Only create prompt data files for two of the four indices
    train_buf = pd.DataFrame(pd.DataFrame(np.zeros((2, 2)), columns=[i for i in range(2)]))
    prompt_indices = [['one', 'three'], ['one', 'two'], ['one', 'four']]
    for i, folder in enumerate(["train", "val", "test"]):
        parent = prompt_data_path / folder
        parent.mkdir()
        parent = parent / "mock"
        parent.mkdir()
            
        for j, filename in enumerate(prompt_indices[i]):
            p = parent / f"fake1_{filename}_prompt.pth.tar"
            train_prompt = pd.DataFrame(train_buf.iloc[j].values)
            train_prompt = train_prompt.T
            assert train_prompt.shape == (1, 2)
            torch.save(train_prompt, str(p))

    mock_array = [np.array([math.pi for i in range(11)]),
        np.array([math.exp(i) for i in range(11)]),
        np.array([2*i for i in range(11)]),
        np.array([i**2 for i in range(1)])
    ]
    mock_df = pd.DataFrame([])
    #mock_torch_ds = TSPromptDataset([], [], 0, 0)
    mocker.patch.object(datasetFactory, 'fetch', return_value=mock_df)
    mocker.patch.object(datasetFactory.splitter, 'get_csv_splits', return_value=([],[],[],['one', 'two', 'three', 'four']))
    mocker.patch.object(datasetFactory.processor, 'process', return_value=(mock_array, mock_array, mock_array))
    
    #mocker.patch.object(datasetFactory, 'createTorchDS', return_value=mock_torch_ds)
    train_dataset, val_dataset, test_datasets = datasetFactory.getDatasets()

    torch.set_default_dtype(torch.float64)
    truth_value_seq = torch.Tensor([math.pi if i > 1 else 0 for i in range(12)])
    truth_value_seq = torch.reshape(truth_value_seq, (12, 1))
    truth_value_pred = torch.Tensor([math.pi])
    truth_value_pred = torch.reshape(truth_value_pred, (1, 1))

    assert len(train_dataset) == 1
    assert torch.equal(train_dataset[0][0], truth_value_seq)
    assert torch.equal(train_dataset[0][1], truth_value_pred)

    assert len(val_dataset) == 1
    assert torch.equal(val_dataset[0][0], truth_value_seq)
    assert torch.equal(val_dataset[0][1], truth_value_pred)

    assert len(test_datasets) == 1
    assert len(test_datasets[0]) == 1
    assert torch.equal(test_datasets[0][0][0], truth_value_seq)
    assert torch.equal(test_datasets[0][0][1], truth_value_pred)

    expected_calls = [call(data_path) for data_path in datasetFactory.data_paths]
    datasetFactory.fetch.assert_has_calls(expected_calls)

    assert datasetFactory.splitter.get_csv_splits.call_count == 1
    assert datasetFactory.processor.process.call_count == 1
