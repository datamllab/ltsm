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
        "model": "PatchTST",
        "model_name_or_path": "gpt2-medium",
        "pred_len": 96,
        "gradient_accumulation_steps": 64,
        "test_data_path_list": [str(data_path)],
        "prompt_data_path": str(prompt_data_path),
        "enc_in": 1,
        "e_layers": 3,
        "n_heads": 16,
        "d_model": 128,
        "d_ff": 256,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "head_dropout": 0,
        "seq_len": 336,
        "patch_len": 16,
        "stride": 8,
        "des": 'Exp',
        "train_epochs": 1000,
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
        "fc_dropout": 0.05,
        "head_dropout": 0.0,
        "patch_len": 16,
        "padding_patch": 'end',
        "revin": 1,
        "affine": 0,
        "subtract_last": 0,
        "decomposition": 0,
        "kernel_size": 25,
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

    patch_num = int((config.seq_len - config.patch_len) / config.stride + 1)
    # multi-head self-attention parameter count (W_Q, W_K, W_V, to_out)
    expected_param_count = 4*(config.d_model * config.d_model + config.d_model)
    # feed-forward nn parameter count
    expected_param_count += 2*config.d_model*config.d_ff + config.d_model + config.d_ff
    # layer norm parameter count
    expected_param_count += 4*config.d_model

    # multiply by number of encoder layers
    expected_param_count *= config.e_layers

    # Input encoding parameter count
    expected_param_count += config.patch_len*config.d_model + config.d_model

    # Positional encoding parameter count
    expected_param_count += patch_num*config.d_model

    # RevIn parameter count
    expected_param_count += 2

    # Flatten Head parameter count
    expected_param_count += config.d_model*patch_num*config.pred_len + config.pred_len

    assert param_count == expected_param_count

def test_forward_output_shape(config):
    model = get_model(config)
    batch_size = 32
    channel = 16
    input_length = config.seq_len
    input = torch.tensor(np.zeros((batch_size, input_length, channel))).float()
    output = model(input)
    assert output.size() == torch.Size([batch_size, config.pred_len, channel])