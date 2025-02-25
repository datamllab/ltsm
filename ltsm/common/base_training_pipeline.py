from ltsm.models import get_model, LTSMConfig
from ltsm.models.utils import freeze_parameters
from ltsm.data_provider import DatasetFactory
from ltsm.data_provider.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import EvalPrediction
from peft import get_peft_model, LoraConfig
import logging
import sys

class BaseTrainingPipeline:
    def __init__(self, 
                 config: LTSMConfig, 
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 collate_fn=None, 
                 compute_loss=None, 
                 compute_metrics=None, 
                 prediction_step=None):
        """
        Initializes the TrainingPipeline with given arguments and a model manager.

        Args:
            args (argparse.Namespace): Contains training settings such as output directory, batch size,
                                       learning rate, and other hyperparameters.
        """
        self.config = config

        if not model:
            self.model = get_model(config)

        if self.config.lora:
            peft_config = LoraConfig(
                target_modules=["c_attn"],
                inference_mode=False,
                r=self.config.lora_dim,
                lora_alpha=32,
                lora_dropout=0.1
            )
            self.model = get_peft_model(self.model, peft_config)
        elif self.config.freeze:
            freeze_parameters(self.model)

        if not optimizer:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        else:
            self.optimizer = optimizer
        
        if not scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.tmax, eta_min=1e-8)
        else:
            self.scheduler = scheduler

        self.collate_fn = collate_fn
        self.compute_loss = compute_loss
        self.compute_metrics = compute_metrics
        self.prediction_step = prediction_step

        handlers = [logging.StreamHandler(sys.stdout)]
        if hasattr(config, "log_file"):
            handlers.append(logging.FileHandler(config.log_file))

        logging.basicConfig(
            handlers=handlers,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
        self.logger = logging.getLogger(self.__class__.__name__)


    def log_info(self, message: str):
        """
        Logs an informational message.
        Args:
            message (str): The message to log.
        """
        self.logger.info(message)

    def log_warning(self, message: str):
        """
        Logs a warning message.
        Args:
            message (str): The message to log.
        """
        self.logger.warning(message)

    def log_error(self, message: str):
        """
        Logs an error message.
        Args:
            message (str): The message to log.
        """
        self.logger.error(message)

    def log_debug(self, message: str):
        """
        Logs a debug message.
        Args:
            message (str): The message to log.
        """
        self.logger.debug(message)
    
    def log_exception(self, message: str):
        """
        Logs an exception message.
        Args:
            message (str): The message to log.
        """
        self.logger.exception(f"Exception occurred: {message}")

    def run(self):
        """
        Runs the training pipeline, including dataset loading, model training, and evaluation.
        This method orchestrates the entire training process, including data preparation,
        model training, and evaluation, while logging relevant information.
        Returns:
            None
        """
        pass

    def get_datasets(self): 
        """
        Generates and returns the training, validation, and test datasets along with the data processor.

        Depending on the model specified in the configuration, this method creates datasets using either
        a custom dataset factory for "LTSM" models or predefined dataset classes for other models.

        Returns:
            tuple: A tuple containing:
                - train_dataset: The dataset used for training.
                - val_dataset: The dataset used for validation.
                - test_datasets: A list containing the datasets used for testing.
                - processor: The data processor used for scaling or other preprocessing tasks.
        """
        if "LTSM" in self.config.model:
            # Create datasets
            dataset_factory = DatasetFactory(
                data_paths=self.config.data_path,
                prompt_data_path=self.config.prompt_data_path,
                data_processing=self.config.data_processing,
                seq_len=self.config.seq_len,
                pred_len=self.config.pred_len,
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
                model=self.config.model,
                split_test_sets=False,
                downsample_rate=self.config.downsample_rate,
                do_anomaly=self.config.do_anomaly
            )
            train_dataset, val_dataset, test_datasets = dataset_factory.getDatasets()
            processor = dataset_factory.processor
        else:
            timeenc = 0 if self.config.embed != 'timeF' else 1
            Data = Dataset_Custom
            if self.config.data == "ETTh1" or self.config.data == "ETTh2":
                Data = Dataset_ETT_hour
            elif self.config.data == "ETTm1" or self.config.data == "ETTm2":
                Data = Dataset_ETT_minute

            train_dataset = Data(
                data_path=self.config.data_path[0],
                split='train',
                size=[self.config.seq_len, self.config.pred_len],
                freq=self.config.freq,
                timeenc=timeenc,
                features=self.config.features
            )
            val_dataset = Data(
                data_path=self.config.data_path[0],
                split='val',
                size=[self.config.seq_len, self.config.pred_len],
                freq=self.config.freq,
                timeenc=timeenc,
                features=self.config.features
            )
            test_datasets = [Data(
                data_path=self.config.data_path[0],
                split='test',
                size=[self.config.seq_len, self.config.pred_len],
                freq=self.config.freq,
                timeenc=timeenc,
                features=self.config.features
            )]
            processor = train_dataset.scaler

        return train_dataset, val_dataset, test_datasets, processor
    
    def get_data_loaders(self):
        """
        Creates and returns DataLoader objects for training, validation, and testing datasets.
        This method initializes DataLoader instances with specified batch size, shuffling, and number of workers.
        It also logs the sizes of the datasets.
        Returns:
            tuple: A tuple containing:
                - train_loader: DataLoader for the training dataset.
                - val_loader: DataLoader for the validation dataset.
                - test_loader: DataLoader for the testing dataset.
                - processor: The data processor used for scaling or other preprocessing tasks.
        """
        train_dataset, val_dataset, test_datasets, processor = self.get_datasets()
        self.log_info(f"Data loaded, train size {len(train_dataset)}, val size {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # split_test_data set to False, length of test_datasets is 1
        test_loader = DataLoader(
            test_datasets[0],
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        return train_loader, val_loader, test_loader, processor

    @staticmethod
    def default_compute_metrics(p: EvalPrediction):
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
    
    @staticmethod
    def default_compute_loss(model, inputs, return_outputs=False):
        """
        Computes the loss for model training.

        Args:
            model (torch.nn.Module): The model used for predictions.
            inputs (dict): Input data and labels.
            return_outputs (bool): If True, returns both loss and model outputs.

        Returns:
            torch.Tensor or tuple: The computed loss, and optionally the outputs.
        """
        outputs = model(inputs["input_data"])
        loss = nn.functional.mse_loss(outputs, inputs["labels"])
        return (loss, outputs) if return_outputs else loss
    
    @staticmethod
    @torch.no_grad()
    def default_prediction_step(model, inputs, prediction_loss_only=False, ignore_keys=None):
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
        outputs = model(input_data)
        loss = nn.functional.mse_loss(outputs, labels)
        return (loss, outputs, labels)
    
    @staticmethod
    def default_collate_fn(batch):
        """
        Collates a batch of data into tensors for model training.

        Args:
            batch (list): List of data samples with 'input_data' and 'labels' keys.

        Returns:
            dict: Collated batch with 'input_data' and 'labels' tensors.
        """
        return {
            'input_data': torch.from_numpy(np.stack([x['input_data'] for x in batch])).type(torch.float32),
            'labels': torch.from_numpy(np.stack([x['labels'] for x in batch])).type(torch.float32),
        }
