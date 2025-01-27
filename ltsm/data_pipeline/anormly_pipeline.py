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

from ltsm.data_provider.data_factory import get_datasets
from ltsm.data_provider.data_loader import HF_Dataset
from ltsm.data_pipeline.model_manager import ModelManager

from sklearn.metrics import precision_score, recall_score, f1_score

import logging
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class AnomalyModelManager(ModelManager):
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
        outputs = model(inputs["input_data"]) # output should be B, L, M
        labels = inputs["labels"]
        #print(outputs.shape, labels.shape)
        #B, L, M, _ = outputs.shape
        loss = nn.functional.cross_entropy(outputs, labels)
        #loss = nn.functional.cross_entropy(outputs.reshape(B*L,-1), inputs["labels"][:,1:].long().reshape(B*L))
        return (loss, outputs) if return_outputs else loss
    
    def compute_metrics(self, p):
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
    

class CustomTrainer(Trainer):
    """
    Custom Trainer class that extends the Trainer class from the Transformers library.
    This class is used to add custom logging to the Trainer.
    """
    def training_step(self, model, inputs):
        # this func is used to get more information during training
        # here is used to check the existence of label 1 in the batch
        labels = inputs["labels"]
        has_label_one = (labels == 1.).any().item() if labels is not None else False
        self.current_label_check = has_label_one
        
        return super().training_step(model, inputs)
    
    def log(self, logs):
        # this func add the custom log to Trainer
        if hasattr(self, "current_label_check"):
            logs["has_label_one"] = self.current_label_check
        super().log(logs)

class AnomalyTrainingPipeline():
    """
    A pipeline for managing the training and evaluation process of a machine learning model.

    Attributes:
        args (argparse.Namespace): Arguments containing training configuration and hyperparameters.
        model_manager (ModelManager): An instance responsible for creating, managing, and optimizing the model.
    """
    def __init__(self, args: argparse.Namespace):
        """
        Initializes the TrainingPipeline with given arguments and a model manager.

        Args:
            args (argparse.Namespace): Contains training settings such as output directory, batch size,
                                       learning rate, and other hyperparameters.
        """
        self.args = args
        self.model_manager = AnomalyModelManager(args)

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
        logging.info(self.args)
    
        model = self.model_manager.create_model()
        
        # Training settings
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            evaluation_strategy="steps",
            num_train_epochs=self.args.train_epochs,
            fp16=False,
            save_steps=100,
            eval_steps=25,
            logging_steps=1,
            learning_rate=self.args.learning_rate,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            save_total_limit=10,
            remove_unused_columns=False,
            push_to_hub=False,
            load_best_model_at_end=True,
        )

        train_dataset, eval_dataset, test_datasets, _ = get_datasets(self.args)
        train_dataset, eval_dataset= HF_Dataset(train_dataset), HF_Dataset(eval_dataset)
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            data_collator=self.model_manager.collate_fn,
            compute_metrics=self.model_manager.compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=None,
            optimizers=(self.model_manager.optimizer, self.model_manager.scheduler),
        )

        # Overload the trainer API
        if not self.args.eval:
            trainer.compute_loss = self.model_manager.compute_loss
            trainer.prediction_step = self.model_manager.prediction_step    
            train_results = trainer.train()
            trainer.save_model()
            trainer.log_metrics("train", train_results.metrics)
            trainer.save_metrics("train", train_results.metrics)
            trainer.save_state()

        # Testing settings
        for test_dataset in test_datasets:
            trainer.compute_loss = self.model_manager.compute_loss
            trainer.prediction_step = self.model_manager.prediction_step
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
        logging.info(f"Anomaly Mode, Set pred_len to seq_len")
        args.pred_len = args.seq_len
    
    if 'output_dir_template' in config_dict:
        args.output_dir = config_dict['output_dir_template'].format(
            learning_rate=args.learning_rate,
            downsample_rate=args.downsample_rate,
            freeze=args.freeze,
            train_epochs=args.train_epochs,
            pred_len=args.pred_len
        )
    logging.info(f"Output Dir: {args.output_dir}")

    return args


def anomaly_seed_all(fixed_seed):
    random.seed(fixed_seed)
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)