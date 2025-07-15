import numpy as np
import torch
from torch import nn

from ltsm.data_pipeline import StatisticalTrainingPipeline, get_args, seed_all
from ltsm.data_provider.data_loader import HF_Dataset
from ltsm.models.utils import freeze_parameters, print_trainable_parameters
from peft import get_peft_config, get_peft_model, LoraConfig

from transformers import (
    EvalPrediction,
)

def run():
    config = get_args()
    seed = config.seed
    seed_all(seed)

    model = get_model(config)

    if config.lora:
        peft_config = LoraConfig(
            target_modules=["c_attn"],
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

    # Optimizer settings
    model_optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=config.tmax, eta_min=1e-8)

    # Evaluation metrics
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds)
        if preds.shape != p.label_ids.shape:
            label_ids = np.squeeze(p.label_ids)
        else:
            label_ids = p.label_ids
        return {
                "mse": ((preds - label_ids) ** 2).mean().item(),
                "mae": (np.abs(preds - label_ids)).mean().item()
        }

    # Loss function
    def compute_loss(model, inputs, return_outputs=False):
        outputs = model(inputs["input_data"])
        loss = nn.functional.mse_loss(outputs, inputs["labels"])
        return (loss, outputs) if return_outputs else loss

    # Data collator
    def collate_fn(batch):
        return {
            'input_data': torch.from_numpy(np.stack([x['input_data'] for x in batch])).type(torch.float32),
            'labels': torch.from_numpy(np.stack([x['labels'] for x in batch])).type(torch.float32),
        }

    # Prediction step
    @torch.no_grad()
    def prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=None):
        # CSV
        input_data = inputs["input_data"].to(model.module.device)
        labels = inputs["labels"].to(model.module.device)
        outputs = model(input_data)
        loss = nn.functional.mse_loss(outputs, labels)
        return (loss, outputs, labels)


    pipeline = StatisticalTrainingPipeline(config, 
                                           model=model, 
                                           collate_fn=collate_fn, 
                                           prediction_step=prediction_step, 
                                           compute_loss=compute_loss,
                                           compute_metrics=compute_metrics)
    pipeline.run()


if __name__ == "__main__":
    run()