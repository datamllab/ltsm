import pytest
from transformers import PretrainedConfig, PreTrainedModel
from ltsm.models import register_model, get_model, model_dict

def test_register_model(mocker):
    mock_model = mocker.MagicMock(spec=PreTrainedModel)
    register_model(mock_model, "MockModel1")
    assert "MockModel1" in model_dict
    assert model_dict["MockModel1"] == mock_model

    with pytest.raises(AssertionError, match="Reader MockModel1 already registered"):
        register_model(mock_model, "MockModel1")

def test_get_model(mocker):
    mock_model = mocker.MagicMock(spec=PreTrainedModel)
    mock_config = mocker.MagicMock(spec=PretrainedConfig)
    register_model(mock_model, "MockModel2")
    
    instance = get_model(mock_config, "MockModel2")
    mock_model.assert_called_once_with(mock_config)
    assert isinstance(instance, mocker.MagicMock)

def test_get_model_invalid_name():
    with pytest.raises(ValueError, match="Model NonExistentModel is not registered"):
        get_model(PretrainedConfig(), "NonExistentModel")

def test_get_model_local_pretrain(mocker):
    mock_from_pretrained = mocker.patch("transformers.PretrainedConfig.from_pretrained")
    mock_model = mocker.MagicMock(spec=PreTrainedModel)
    register_model(mock_model, "MockModel3")
    
    mock_from_pretrained.return_value = mocker.MagicMock()
    instance = get_model(PretrainedConfig(), "MockModel3", local_pretrain="path/to/pretrained")
    mock_model.from_pretrained.assert_called_once_with("path/to/pretrained", mock_from_pretrained.return_value)
    assert isinstance(instance, mocker.MagicMock)

def test_get_model_hf_hub(mocker):
    mock_from_pretrained = mocker.patch("transformers.PreTrainedModel.from_pretrained")
    mock_model = mocker.MagicMock(spec=PreTrainedModel)
    register_model(mock_model, "MockModel4")
    
    instance = get_model(PretrainedConfig(), "MockModel4", hf_hub_model="mock-hub-model")
    mock_model.from_pretrained.assert_called_once_with("mock-hub-model", PretrainedConfig())
    assert isinstance(instance, mocker.MagicMock)