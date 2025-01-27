import argparse
from ltsm.models import get_model, LTSMConfig
import torch
from torch import nn
import numpy as np
from peft import get_peft_model, LoraConfig
from transformers import (
    EvalPrediction,
)

class ModelManager:
    """
    Manages model creation, parameter settings, optimizer, and evaluation metrics for training.

    Attributes:
        args (argparse.Namespace): Configuration and hyperparameters for model training.
        model (torch.nn.Module): The model to be trained, created based on configuration.
        optimizer (torch.optim.Optimizer): Optimizer for model parameter updates.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    """
    def __init__(self, args: argparse.Namespace):
        """
        Initializes the ModelManager with provided arguments and default values for model, optimizer, and scheduler.

        Args:
            args (argparse.Namespace): Training configurations and hyperparameters.
        """
        self.args = args
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def print_trainable_parameters(self):
        """
        Prints the names of parameters in the model that are trainable.
        """
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"{n} is trainable...")

    def freeze_parameters(self):
        """
        Sets certain model parameters to non-trainable, and specific parameters to trainable, based on predefined
        lists of layer names to freeze or keep trainable.
        """
        freeze_param_buf = ["gpt2"]
        for n, p in self.model.named_parameters():
            if any(fp in n for fp in freeze_param_buf):
                p.requires_grad = False
                print(f"{n} has been freeezed")

        trainable_param_buf = ["ln", "wpe", "in_layer", "out_layer", "lora"]
        for n, p in self.model.named_parameters():
            if any(fp in n for fp in trainable_param_buf):
                p.requires_grad = True

    def create_model(self):
        """
        Initializes and configures the model based on specified arguments, including options for
        freezing parameters or applying LoRA (Low-Rank Adaptation).

        Returns:
            torch.nn.Module: The configured model ready for training.
        """
        model_config = LTSMConfig(**vars(self.args))
        self.model = get_model(model_config)

        if self.args.lora:
            peft_config = LoraConfig(
                target_modules=["c_attn"],
                inference_mode=False,
                r=self.args.lora_dim,
                lora_alpha=32,
                lora_dropout=0.1
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        elif self.args.freeze:
            self.freeze_parameters()

        self.print_trainable_parameters()

        # Optimizer settings
        return self.model
    
    def set_optimizer(self):
        """
        Configures the optimizer and learning rate scheduler for the model training.

        Uses Adam optimizer and cosine annealing learning rate scheduler.
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.tmax, eta_min=1e-8)

    def compute_metrics(self, p: EvalPrediction):
        """
        Computes evaluation metrics for model predictions.

        Args:
            p (EvalPrediction): Contains predictions and label IDs.

        Returns:
            dict: Dictionary containing Mean Squared Error (MSE) and Mean Absolute Error (MAE).
        """
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
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the loss for model training.

        Args:
            model (torch.nn.Module): The model used for predictions.
            inputs (dict): Input data and labels.
            return_outputs (bool): If True, returns both loss and model outputs.

        Returns:
            torch.Tensor or tuple: The computed loss, and optionally the outputs.
        """
        if self.args.model == 'Informer':
            input_data_mark = inputs["timestamp_input"].to(model.module.device)
            label_mark = inputs["timestamp_labels"].to(model.module.device)
            outputs = model(inputs["input_data"], input_data_mark, inputs["labels"], label_mark)
        else:
            outputs = model(inputs["input_data"])
        loss = nn.functional.mse_loss(outputs, inputs["labels"])
        return (loss, outputs) if return_outputs else loss
    
    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """
        Makes a prediction step, computing loss and returning model outputs without gradients.

        Args:
            model (torch.nn.Module): The model used for predictions.
            inputs (dict): Input data and labels.
            prediction_loss_only (bool): If True, returns only the loss.
            ignore_keys (list): Keys to ignore in inputs.

        Returns:
            tuple: The loss, outputs, and labels.
        """
        input_data = inputs["input_data"].to(model.module.device)
        labels = inputs["labels"].to(model.module.device)
        if self.args.model == 'Informer':
            input_data_mark = inputs["timestamp_input"].to(model.module.device)
            label_mark = inputs["timestamp_labels"].to(model.module.device)
            outputs = model(input_data, input_data_mark, labels, label_mark)
        else:
            outputs = model(input_data)
        loss = nn.functional.mse_loss(outputs, labels)
        return (loss, outputs, labels)
    
    def collate_fn(self, batch):
        """
        Collates a batch of data into tensors for model training.

        Args:
            batch (list): List of data samples with 'input_data' and 'labels' keys.

        Returns:
            dict: Collated batch with 'input_data' and 'labels' tensors.
        """
        if self.args.model == 'Informer':
            return {
                'input_data': torch.from_numpy(np.stack([x['input_data'] for x in batch])).type(torch.float32),
                'labels': torch.from_numpy(np.stack([x['labels'] for x in batch])).type(torch.float32),
                'timestamp_input': torch.from_numpy(np.stack([x['timestamp_input'] for x in batch])).type(torch.float32),
                'timestamp_labels': torch.from_numpy(np.stack([x['timestamp_labels'] for x in batch])).type(torch.float32)
            }

        return {
            'input_data': torch.from_numpy(np.stack([x['input_data'] for x in batch])).type(torch.float32),
            'labels': torch.from_numpy(np.stack([x['labels'] for x in batch])).type(torch.float32),
        }
