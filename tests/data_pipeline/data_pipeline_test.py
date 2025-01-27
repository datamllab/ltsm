import pytest
from ltsm.data_provider.dataset import TSDataset
from ltsm.data_pipeline import TrainingPipeline, ModelManager
from transformers import (
    Trainer,
    TrainingArguments
)

import argparse

@pytest.fixture
def mock_args():
    #Fixture for creating mock arguments
    arg_dict = {
        'model': 'LTSM',
        'data_path':'./datasets',
        'prompt_data_path':'./prompt_bank',
        'output_dir': './output',
        'seq_len': 256,
        'pred_len': 12,
        'train_ratio': 0.7,
        'val_ratio': 0.1,
        'learning_rate': 5e-5,
        'downsample_rate': 10,
        'train_epochs': 8,
        'batch_size': 100,
        'eval': False,
        'lora': False,
        'freeze': False,
        'data_processing': 'standard_scaler',
        'gradient_accumulation_steps': 1
    }
        
    return argparse.Namespace(**arg_dict)

@pytest.fixture
def pipeline(mock_args):
    # Fixture to create pipeline
    return TrainingPipeline(mock_args)

def test_initialization(pipeline, mock_args):
    #Test that TrainingPipeline initializes correctly

    assert pipeline.args == mock_args
    assert isinstance(pipeline.model_manager, ModelManager)

def test_run_training(mocker, pipeline):
    # Mock dataset loading and Trainer behavior
    mocker.patch.object(pipeline.model_manager, 'create_model', return_value=(None))
    mock_get_datasets = mocker.patch('ltsm.data_pipeline.data_pipeline.get_datasets', return_value=(TSDataset([], 0, 0), TSDataset([], 0, 0), [None, None, None, None], None))
    mock_trainer = mocker.patch('ltsm.data_pipeline.data_pipeline.Trainer')
    mock_trainer.evaluate.return_value = None
    
    pipeline.run()

    # Ensure datasets are loaded and Trainer is instantiated
    mock_get_datasets.assert_called_once_with(pipeline.args)
    assert pipeline.model_manager.create_model.call_count == 1

    # Check if train is called when eval is False
    if not pipeline.args.eval:
        assert mock_trainer.return_value.train.called
        assert mock_trainer.return_value.save_model.called
    
    assert mock_trainer.return_value.evaluate.call_count == 4
    assert mock_trainer.return_value.save_metrics.call_count == 5
    assert mock_trainer.return_value.log_metrics.call_count == 5


def test_run_evaluation_only(mocker, pipeline):
    pipeline.args.eval = True  # Set eval-only mode
    # Mock dataset loading and Trainer behavior
    mocker.patch.object(pipeline.model_manager, 'create_model', return_value=(None))
    mock_get_datasets = mocker.patch('ltsm.data_pipeline.data_pipeline.get_datasets', return_value=(TSDataset([], 0, 0), TSDataset([], 0, 0), [None, None, None, None], None))
    mock_trainer = mocker.patch('ltsm.data_pipeline.data_pipeline.Trainer')
   
    pipeline.run()

    # Ensure datasets are loaded and Trainer is instantiated
    mock_get_datasets.assert_called_once_with(pipeline.args)
    assert pipeline.model_manager.create_model.call_count == 1

    # Ensure training is skipped and only evaluation is called
    assert not mock_trainer.return_value.train.called
    assert mock_trainer.return_value.evaluate.called
    assert mock_trainer.return_value.save_metrics.called