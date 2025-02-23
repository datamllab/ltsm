"""Pipeline for Anormly Data Detection
    Main Difference from the LTSM : 
        - pred_len == seq_len
        - label is the anomaly label of input seq_len
        - loss is CE/BCE

"""

import numpy as np
import torch
import argparse
import random
import ipdb
from torch import nn
import json

from ltsm.data_provider.data_loader import HF_Dataset
from ltsm.common.base_training_pipeline import BaseTrainingPipeline
from ltsm.models import LTSMConfig

from sklearn.metrics import precision_score, recall_score, f1_score

from transformers import (
    Trainer,
    TrainingArguments
)


def compute_loss(model, inputs, return_outputs=False):
    """
    Computes the loss for model training.

    Args:
        model (torch.nn.Module): The model used for predictions.
        inputs (dict): Input data and labels.
        return_outputs (bool): If True, returns both loss and model outputs.

    Returns:
        torch.Tensor or tuple: The computed loss, and optionally the outputs.
    """
    outputs = model(inputs["input_data"]) # output should be B, L, M
    labels = inputs["labels"]
    #print(outputs.shape, labels.shape)
    #B, L, M, _ = outputs.shape
    loss = nn.functional.cross_entropy(outputs, labels)
    #loss = nn.functional.cross_entropy(outputs.reshape(B*L,-1), inputs["labels"][:,1:].long().reshape(B*L))
    return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    print(preds.shape, p.label_ids.shape)
    preds = np.squeeze(preds)
    if preds.shape != p.label_ids.shape:
        label_ids = np.squeeze(p.label_ids)
    else:
        label_ids = p.label_ids
    print(preds.shape, label_ids.shape)
    preds_class = (preds > 0.5).astype(int)
    
    return {
            "precision": precision_score(label_ids, preds_class, average="micro"),
            "recall": recall_score(label_ids, preds_class, average="micro"),
            "f1": f1_score(label_ids, preds_class, average="micro")              
    }

class AnomalyTrainingPipeline(BaseTrainingPipeline):
    """
    A pipeline for managing the training and evaluation process of a machine learning model.

    Attributes:
        args (argparse.Namespace): Arguments containing training configuration and hyperparameters.
        model_manager (ModelManager): An instance responsible for creating, managing, and optimizing the model.
    """
    def __init__(self, config: LTSMConfig, **kwargs):
        """
        Initializes the TrainingPipeline with given arguments and a model manager.

        Args:
            args (argparse.Namespace): Contains training settings such as output directory, batch size,
                                       learning rate, and other hyperparameters.
        """
        super().__init__(config, compute_loss=compute_loss, compute_metrics=compute_metrics, **kwargs)
        # Training settings
        self.training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            evaluation_strategy="steps",
            num_train_epochs=config.train_epochs,
            fp16=False,
            save_steps=100,
            eval_steps=25,
            logging_steps=5,
            learning_rate=config.learning_rate,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            save_total_limit=10,
            remove_unused_columns=False,
            push_to_hub=False,
            load_best_model_at_end=True,
        )
        

    def run(self):
        """
        Runs the training and evaluation process for the model.

        The process includes:
            - Logging configuration and training arguments.
            - Creating a model with the model manager.
            - Setting up training and evaluation parameters.
            - Loading and formatting training and evaluation datasets.
            - Training the model and saving metrics and state.
            - Evaluating the model on test datasets and logging metrics.
        """
        self.log_info(self.config.to_dict())
        train_dataset, eval_dataset, test_datasets, _ = self.get_datasets()
        train_dataset, eval_dataset= HF_Dataset(train_dataset), HF_Dataset(eval_dataset)
        
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=None,
            optimizers=(self.optimizer, self.scheduler),
        )

        # Overload the trainer API
        if not self.args.eval:
            trainer.compute_loss = self.compute_loss
            trainer.prediction_step = self.prediction_step        
            train_results = trainer.train()
            trainer.save_model()
            trainer.log_metrics("train", train_results.metrics)
            trainer.save_metrics("train", train_results.metrics)
            trainer.save_state()

        # Testing settings
        for test_dataset in test_datasets:
            trainer.compute_loss = self.compute_loss
            trainer.prediction_step = self.prediction_step
            test_dataset = HF_Dataset(test_dataset)

            metrics = trainer.evaluate(test_dataset)
            trainer.log_metrics("Test", metrics)
            trainer.save_metrics("Test", metrics)

def anomaly_get_args():
    parser = argparse.ArgumentParser(description='LTSM')

    parser.add_argument('--config_path', type=str, required=True, help='config path')
    args, unknown = parser.parse_known_args()
    config_path = args.config_path

    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    args = argparse.Namespace(**config_dict)

    if args.pred_len is None:
        # self.log_info(f"Anomaly Mode, Set pred_len to seq_len")
        args.pred_len = args.seq_len
    
    if 'output_dir_template' in config_dict:
        args.output_dir = config_dict['output_dir_template'].format(
            learning_rate=args.learning_rate,
            downsample_rate=args.downsample_rate,
            freeze=args.freeze,
            train_epochs=args.train_epochs,
            pred_len=args.pred_len
        )
    # self.log_info(f"Output Dir: {args.output_dir}")
    config = LTSMConfig.from_dict(vars(args))

    if hasattr(args, "config") and args.config:
        config.load(args.config)

    return config


def anomaly_seed_all(fixed_seed):
    random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)