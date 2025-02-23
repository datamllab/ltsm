import pytest
from ltsm.models import LTSMConfig
from ltsm.data_provider.dataset import TSDataset
from ltsm.data_pipeline import StatisticalTrainingPipeline
from transformers import TrainingArguments

@pytest.fixture
def mock_args():
    #Fixture for creating mock arguments
    arg_dict = {
        'model': 'LTSM',
        'model_name_or_path': 'gpt2-medium',
        'log_file': 'log.txt',
        'gpt_layers': 3,
        'data_path':'./datasets',
        'prompt_data_path':'./prompt_bank',
        'output_dir': './output',
        'patch_size': 16,
        'pretrain': True,
        'stride': 2,
        'seq_len': 256,
        'pred_len': 12,
        'prompt_len': 8,
        'train_ratio': 0.7,
        'val_ratio': 0.1,
        'tmax': 10,
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
    
    return LTSMConfig(**arg_dict)

@pytest.fixture
def pipeline(mock_args):
    # Fixture to create pipeline
    return StatisticalTrainingPipeline(mock_args)

def test_initialization(pipeline, mock_args):
    #Test that StatisticalTrainingPipeline initializes correctly

    for arg, value in vars(mock_args).items():
        assert getattr(pipeline.config, arg) == value
    assert isinstance(pipeline.training_args, TrainingArguments)

def test_run_training(mocker, pipeline):
    # Mock dataset loading and Trainer behavior
    mock_get_datasets = mocker.patch.object(pipeline, 'get_datasets', return_value=(TSDataset([], 0, 0), TSDataset([], 0, 0), [None, None, None, None], None))
    mock_trainer = mocker.patch('ltsm.data_pipeline.stat_pipeline.Trainer')
    mock_trainer.evaluate.return_value = None
    
    pipeline.run()

    # Ensure datasets are loaded and Trainer is instantiated
    mock_get_datasets.assert_called_once()

    # Check if train is called when eval is False
    if not pipeline.config.eval:
        assert mock_trainer.return_value.train.called
        assert mock_trainer.return_value.save_model.called
    
    assert mock_trainer.return_value.evaluate.call_count == 4
    assert mock_trainer.return_value.save_metrics.call_count == 5
    assert mock_trainer.return_value.log_metrics.call_count == 5


def test_run_evaluation_only(mocker, pipeline):
    pipeline.config.eval = True  # Set eval-only mode
    # Mock dataset loading and Trainer behavior
    mock_get_datasets = mocker.patch.object(pipeline, 'get_datasets', return_value=(TSDataset([], 0, 0), TSDataset([], 0, 0), [None, None, None, None], None))
    mock_trainer = mocker.patch('ltsm.data_pipeline.stat_pipeline.Trainer')
   
    pipeline.run()

    # Ensure datasets are loaded and Trainer is instantiated
    mock_get_datasets.assert_called_once()

    # Ensure training is skipped and only evaluation is called
    assert not mock_trainer.return_value.train.called
    assert mock_trainer.return_value.evaluate.called
    assert mock_trainer.return_value.save_metrics.called