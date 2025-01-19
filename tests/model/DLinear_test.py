import pytest
from ltsm.models import get_model, LTSMConfig
from transformers import PreTrainedModel
import torch
import numpy as np

@pytest.fixture
def config(tmp_path):
    data_path = tmp_path / "test.csv"
    prompt_data_path = tmp_path / "prompt_normalize_split"
    prompt_data_path.mkdir()
    OUTPUT_PATH = data_path / "output"

    config = {
        "data_path": str(data_path),
        "model": "DLinear",
        "model_name_or_path": "gpt2-medium",
        "pred_len": 96,
        "gradient_accumulation_steps": 64,
        "test_data_path_list": [str(data_path)],
        "prompt_data_path": str(prompt_data_path),
        "enc_in": 1,
        "seq_len": 336, # Equal to the sequence length + the length of prompt
        "train_epochs": 100,
        "patience": 10,
        "lradj": 'TST',
        "pct_start": 0.2,
        "freeze": 0,
        "itr": 1,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "downsample_rate": 20,
        "output_dir": str(OUTPUT_PATH),
        "eval": 0,
        "individual": 0,
    }
    return LTSMConfig(**config)

def test_model_initialization(config):
    model = get_model(config)
    assert model is not None
    assert isinstance(model, PreTrainedModel)


def test_parameter_count(config):
    model = get_model(config)
    param_count = sum([p.numel() for p in model.parameters() if p.requires_grad])

    expected_param_count = 2*(config.seq_len*config.pred_len + config.pred_len)

    assert param_count == expected_param_count

def test_forward_output_shape(config):
    torch.set_default_dtype(torch.float64)
    model = get_model(config)
    batch_size = 32
    channel = 16
    input_length = config.seq_len
    input = torch.tensor(np.zeros((batch_size, input_length, channel)))
    output = model(input)
    assert output.size() == torch.Size([batch_size, config.pred_len, channel])