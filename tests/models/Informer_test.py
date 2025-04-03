import pytest
from ltsm.models import get_model
from ltsm.models.base_config import InformerConfig
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
        "model": "Informer",
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
        "eval": 0,
        "des": 'Exp',
        "padding_patch": 'end',
        "local_pretrain": "None"
    }

    config = {
        "pred_len": 96,
        "enc_in": 1,
        "e_layers": 3,
        "d_layers": 1,
        "n_heads": 16,
        "d_model": 128,
        "d_ff": 256,
        "dropout": 0.2,
        "fc_dropout": 0.2,
        "head_dropout": 0,
        "seq_len": 336,
        "output_attention": 0,
        "freq": "h",
        "embed": "timeF",
        "factor": 1,
        "c_out": 862,
        "distil": True,
        "embed_type": 0,
        "dec_in": 7,
        "activation": "gelu"
    }
    informer_config = InformerConfig(**config)
    return TrainingConfig(model_config=informer_config, **train_params)

def test_model_initialization(config):
    model = get_model(config.model_config, model_name=config.train_params["model"], local_pretrain=config.train_params["local_pretrain"])
    assert model is not None
    assert isinstance(model, PreTrainedModel)

def test_parameter_count(config):
    model = get_model(config.model_config, model_name=config.train_params["model"], local_pretrain=config.train_params["local_pretrain"])
    param_count = sum([p.numel() for p in model.parameters() if p.requires_grad])

    # Encoder Embedding parameter count
    expected_param_count = config.model_config.d_model*config.model_config.enc_in*3 + 4*config.model_config.d_model

    # Decoder Embedding parameter count
    expected_param_count += config.model_config.d_model*config.model_config.dec_in*3 + 4*config.model_config.d_model

    # Encoder parameter count
    # Encoder layer Conv
    encoder_param_count = 2*config.model_config.d_model*config.model_config.d_ff + config.model_config.d_model + config.model_config.d_ff
    # Encoder Layer Norm
    encoder_param_count += 4*config.model_config.d_model
    # Attention Layer
    encoder_param_count += 4*(config.model_config.d_model*config.model_config.d_model + config.model_config.d_model)
    # Multiply by number of encoder layers
    encoder_param_count *= config.model_config.e_layers

    # Conv layer
    encoder_param_count += (config.model_config.e_layers-1)*(config.model_config.d_model*config.model_config.d_model*3 + 3*config.model_config.d_model)
    # Layer Norm
    encoder_param_count += 2*config.model_config.d_model

    expected_param_count += encoder_param_count

    # Decoder layer parameter count
    # Decoder Conv layers
    decoder_param_count = 2*config.model_config.d_model*config.model_config.d_ff + config.model_config.d_model + config.model_config.d_ff
    # Decoder Layer Norm
    decoder_param_count += 6*config.model_config.d_model
    # Attention Layer
    decoder_param_count += 8*(config.model_config.d_model*config.model_config.d_model + config.model_config.d_model)
    # Multiply by number of decoder layers
    decoder_param_count *= config.model_config.d_layers

    # Layer Norm parameter count
    decoder_param_count += 2*config.model_config.d_model

    # Projection layer parameter count
    decoder_param_count += config.model_config.d_model*config.model_config.c_out+config.model_config.c_out

    expected_param_count += decoder_param_count

    assert param_count == expected_param_count


def test_forward_output_shape(config):
    torch.set_default_dtype(torch.float64)
    model = get_model(config.model_config, model_name=config.train_params["model"], local_pretrain=config.train_params["local_pretrain"])
    batch_size = 32
    input_length = config.model_config.seq_len
    input = torch.tensor(np.zeros((batch_size, input_length, config.model_config.enc_in)))
    input_mark = torch.tensor(np.zeros((batch_size, input_length, 4)))
    dec_inp = torch.tensor(np.zeros((batch_size, input_length, config.model_config.dec_in)))
    dec_mark = torch.tensor(np.zeros((batch_size, input_length, 4)))
    output = model(input, input_mark, dec_inp, dec_mark)
    assert output.size() == torch.Size([batch_size, config.model_config.pred_len, config.model_config.c_out])