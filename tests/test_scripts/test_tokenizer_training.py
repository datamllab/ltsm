import numpy as np
import torch
from torch import nn
import os
import argparse
import random
import sys

sys.path.append("/home/yc146/github_open_ltsm/ltsm")

from ltsm.data_provider.data_loader import HF_Dataset
from ltsm.data_provider.tokenizer.tokenizer_processor import TokenizerConfig
from ltsm.data_pipeline.tokenizer_pipeline import TokenizerTrainingPipeline, tokenizer_get_args, tokenizer_seed_all
from ltsm.models import get_model
from ltsm.models.utils import freeze_parameters, print_trainable_parameters
from peft import get_peft_model, LoraConfig

from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
)

def run():
    config = tokenizer_get_args()
    seed = config.seed
    tokenizer_seed_all(seed)
    model = get_model(config)

    if config.lora:
        peft_config = LoraConfig(
            target_modules=["c_attn"],  # ["q", "v"],
            inference_mode=False,
            r=config.lora_dim,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    elif config.freeze:
        freeze_parameters(model)

    print_trainable_parameters(model)


    model_optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=config.tmax, eta_min=1e-8)

    pipeline = TokenizerTrainingPipeline(config, model, model_optim, lr_scheduler)
    
    pipeline.run()


if __name__ == "__main__":
    run()
