import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers.trainer_utils import EvalPrediction
from ltsm.common.base_training_pipeline import BaseTrainingPipeline
from ltsm.data_provider import TSDataset

@pytest.fixture
def mock_pipeline(mocker):
    config = mocker.Mock()
    config.lora = False
    config.log_file = "out.log"
    config.tmax = 10
    config.freeze = False
    model_mock = mocker.Mock()
    model_mock.parameters.return_value = []
    mock_get_model = mocker.patch("ltsm.common.base_training_pipeline.get_model", return_value=None)
    mock_adam = mocker.Mock(spec=torch.optim.Adam)
    mock_scheduler = mocker.Mock(spec=torch.optim.lr_scheduler.CosineAnnealingLR)
    
    pipeline = BaseTrainingPipeline(config, optimizer=mock_adam, scheduler=mock_scheduler)
    assert pipeline.scheduler is not None
    assert pipeline.optimizer is not None
    assert pipeline.collate_fn is None
    assert pipeline.compute_loss is None
    assert pipeline.compute_metrics is None
    assert pipeline.prediction_step is None
    assert pipeline.logger is not None
    mock_get_model.assert_called_once()

    return pipeline

def test_create_model_lora_enabled(mocker):
    config = mocker.Mock()
    config.lora = True
    config.lora_dim = 10
    config.learning_rate = 1e-3
    config.log_file = "out.log"
    config.tmax = 10
    model_mock = mocker.Mock()
    model_mock.parameters.return_value = []
    mock_get_model = mocker.patch("ltsm.common.base_training_pipeline.get_model", return_value=None)
    mock_get_peft_model = mocker.patch("ltsm.common.base_training_pipeline.get_peft_model")
    mock_adam = mocker.Mock(spec=torch.optim.Adam)
    mock_adam.param_groups = []

    pipeline = BaseTrainingPipeline(config, optimizer=mock_adam)
    assert pipeline.scheduler is not None
    assert pipeline.optimizer is not None
    assert pipeline.collate_fn is None
    assert pipeline.compute_loss is None
    assert pipeline.compute_metrics is None
    assert pipeline.prediction_step is None
    assert pipeline.logger is not None
    mock_get_model.assert_called_once()
    mock_get_peft_model.assert_called_once()

def test_log_info(mock_pipeline, mocker):
    spy = mocker.spy(mock_pipeline.logger, "info")
    mock_pipeline.log_info("Test message")

    spy.assert_called_once_with("Test message")

def test_log_warning(mock_pipeline, mocker):
    spy = mocker.spy(mock_pipeline.logger, "warning")
    mock_pipeline.log_warning("Test message")

    spy.assert_called_once_with("Test message")

def test_log_error(mock_pipeline, mocker):
    spy = mocker.spy(mock_pipeline.logger, "error")
    mock_pipeline.log_error("Test message")

    spy.assert_called_once_with("Test message")

def test_log_debug(mock_pipeline, mocker):
    spy = mocker.spy(mock_pipeline.logger, "debug")
    mock_pipeline.log_debug("Test message")

    spy.assert_called_once_with("Test message")
    
def test_log_exception(mock_pipeline, mocker):
    spy = mocker.spy(mock_pipeline.logger, "exception")
    mock_pipeline.log_exception("Test message")

    spy.assert_called_once_with("Exception occurred: Test message")

def test_get_datasets(mock_pipeline, mocker):
    mock_pipeline.config.model = "LTSM"

    MockDatasetFactory = mocker.patch("ltsm.common.base_training_pipeline.DatasetFactory")
    mock_factory = MockDatasetFactory.return_value
    mock_factory.getDatasets.return_value = ("train_ds", "val_ds", ["test_ds"])
    mock_factory.processor = "processor_mock"

    train_dataset, val_dataset, test_datasets, processor = mock_pipeline.get_datasets()

    assert train_dataset == "train_ds"
    assert val_dataset == "val_ds"
    assert test_datasets == ["test_ds"]
    assert processor == "processor_mock"

def test_get_data_loaders(mock_pipeline, mocker):
    mock_pipeline.config.batch_size = 1
    dataset = TSDataset([[0.5, 0.5, 0.3, 0.4]], 1, 1)
    mock_get_datasets = mocker.patch.object(mock_pipeline, 
                                            "get_datasets", 
                                            return_value=(
                                                dataset, 
                                                dataset, 
                                                [dataset], 
                                                "processor")
                                            )
    train_loader, val_loader, test_loader, processor = mock_pipeline.get_data_loaders()
    
    mock_get_datasets.assert_called_once()
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert processor == "processor"

    train_data = list(train_loader)
    val_data = list(val_loader)
    test_data = list(test_loader)
    
    assert len(train_data) == 3
    assert len(val_data) == 3
    assert len(test_data) == 3


def test_default_compute_metrics():
    preds = np.array([1.0, 2.0, 3.0])
    labels = np.array([1.5, 2.5, 3.5])
    p = EvalPrediction(predictions=preds, label_ids=labels)
    metrics = BaseTrainingPipeline.default_compute_metrics(p)
    
    assert "mse" in metrics
    assert "mae" in metrics
    assert pytest.approx(metrics["mse"]) == ((preds - labels) ** 2).mean()
    assert pytest.approx(metrics["mae"]) == np.abs(preds - labels).mean()

def test_default_compute_loss(mocker):
    model = mocker.MagicMock()
    inputs = {"input_data": torch.tensor([[1.0], [2.0]]), "labels": torch.tensor([[1.5], [2.5]])}
    model.return_value = torch.tensor([[1.2], [2.2]])
    
    loss = BaseTrainingPipeline.default_compute_loss(model, inputs)
    expected_loss = torch.nn.functional.mse_loss(model.return_value, inputs["labels"])
    
    assert torch.isclose(loss, expected_loss)

def test_default_prediction_step(mocker):    
    model = mocker.MagicMock()
    model.module.device = "cpu"
    inputs = {"input_data": torch.tensor([[1.0], [2.0]]), "labels": torch.tensor([[1.5], [2.5]])}
    model.return_value = torch.tensor([[1.2], [2.2]])
    
    loss, outputs, labels = BaseTrainingPipeline.default_prediction_step(model, inputs)
    expected_loss = torch.nn.functional.mse_loss(outputs, labels)
    
    assert torch.isclose(loss, expected_loss)
    assert torch.equal(outputs, model.return_value)
    assert torch.equal(labels, inputs["labels"])

def test_default_collate_fn():
    batch = [
        {"input_data": np.array([1.0, 2.0]), "labels": np.array([3.0])},
        {"input_data": np.array([4.0, 5.0]), "labels": np.array([6.0])}
    ]
    collated_batch = BaseTrainingPipeline.default_collate_fn(batch)
    
    assert isinstance(collated_batch["input_data"], torch.Tensor)
    assert isinstance(collated_batch["labels"], torch.Tensor)
    assert collated_batch["input_data"].shape == (2, 2)
    assert collated_batch["labels"].shape == (2, 1)