import pytest
from ltsm.models import get_model
from ltsm.models.base_config import PatchTSTConfig
from ltsm.common.base_training_pipeline import TrainingConfig
from transformers import PreTrainedModel
import torch
import numpy as np

@pytest.fixture
def config(tmp_path):
    data_path = tmp_path / "test.csv"
    prompt_data_path = tmp_path / "prompt_normalize_split"
    prompt_data_path.mkdir()
    OUTPUT_PATH = data_path / "output"

    train_params = {
        "data_path": str(data_path),
        "model": "PatchTST",
        "model_name_or_path": "gpt2-medium",
        "gradient_accumulation_steps": 64,
        "test_data_path_list": [str(data_path)],
        "prompt_data_path": str(prompt_data_path),
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
        "des": 'Exp',
        "eval": 0
    }
    config = {
        "pred_len": 96,
        "enc_in": 1,
        "seq_len": 336,
        "patch_len": 16,
        "decomposition": False,
        "stride": 8,
        "e_layers": 3,
        "n_heads": 16,
        "d_model": 128,
        "d_ff": 256,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "head_dropout": 0,
        "revin": True,
        "affine": True,
        "subtract_last": False,
        "individual": False
    }

    patchtst_config = PatchTSTConfig(**config)
    return TrainingConfig(patchtst_config, **train_params)

def test_model_initialization(config):
    model = get_model(config.model_config, model_name=config.train_params["model"], local_pretrain=config.train_params["local_pretrain"])
    assert model is not None
    assert isinstance(model, PreTrainedModel)


def test_parameter_count(config):
    model = get_model(config.model_config, model_name=config.train_params["model"], local_pretrain=config.train_params["local_pretrain"])
    param_count = sum([p.numel() for p in model.parameters() if p.requires_grad])

    patch_num = int((config.model_config.seq_len - config.model_config.patch_len) / config.model_config.stride + 1)
    # multi-head self-attention parameter count (W_Q, W_K, W_V, to_out)
    expected_param_count = 4*(config.model_config.d_model * config.model_config.d_model + config.model_config.d_model)
    # feed-forward nn parameter count
    expected_param_count += 2*config.model_config.d_model*config.model_config.d_ff + config.model_config.d_model + config.model_config.d_ff
    # layer norm parameter count
    expected_param_count += 4*config.model_config.d_model

    # multiply by number of encoder layers
    expected_param_count *= config.model_config.e_layers

    # Input encoding parameter count
    expected_param_count += config.model_config.patch_len*config.model_config.d_model + config.model_config.d_model

    # Positional encoding parameter count
    expected_param_count += patch_num*config.model_config.d_model

    # RevIn parameter count
    expected_param_count += 2

    # Flatten Head parameter count
    expected_param_count += config.model_config.d_model*patch_num*config.model_config.pred_len + config.model_config.pred_len

    assert param_count == expected_param_count

def test_forward_output_shape(config):
    model = get_model(config.model_config, model_name=config.train_params["model"], local_pretrain=config.train_params["local_pretrain"])
    batch_size = 32
    channel = 16
    input_length = config.model_config.seq_len
    input = torch.tensor(np.zeros((batch_size, input_length, channel))).float()
    output = model(input)
    assert output.size() == torch.Size([batch_size, config.model_config.pred_len, channel])