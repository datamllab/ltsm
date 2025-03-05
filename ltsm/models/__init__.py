from .ltsm_stat_model import LTSM
from .ltsm_wordprompt import LTSM_WordPrompt
from .ltsm_ts_tokenizer import LTSM_Tokenizer
from .PatchTST import PatchTST
from .DLinear import DLinear
from .Informer import Informer
from transformers import PretrainedConfig, PreTrainedModel

model_dict = {}

def register_model(module, module_name: str):
    """
    Registers a PreTrainedModel module into the model dictionary.

    Args:
        module: A Python module or class that implements a PreTrainedModel.
        module_name (str): The key name for the module in the model dictionary.

    Raises:
        AssertionError: If a model with the same name is already registered
    """
    assert module_name not in model_dict, f"Reader {module_name} already registered"
    model_dict[module_name] = module

register_model(LTSM, 'LTSM')
register_model(LTSM_WordPrompt, 'LTSM_WordPrompt')
register_model(LTSM_Tokenizer, 'LTSM_Tokenizer')
register_model(PatchTST, 'PatchTST')
register_model(DLinear, 'DLinear')
register_model(Informer, 'Informer')

def get_model(config: PretrainedConfig, model_name: str, local_pretrain: str = None) -> PreTrainedModel:
    """
    Factory method to create a model by name.
    
    Args:
        config (PreTrainedConfig): The configuration for the model.
        model_name (str): The name of the model to instantiate.
        local_pretrain (bool): If True, load the model from a local pretraining path.
    
    Returns:
        torch.nn.Module: Instantiated model.
    
    Raises:
        ValueError: If the model name is not found in model_dict.
    """
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} is not registered. Available models: {list(model_dict.keys())}")
    
    # Check for local pretraining
    if local_pretrain is None or local_pretrain == "None":
        return model_dict[model_name](config)
    else:
        model_config = PretrainedConfig.from_pretrained(local_pretrain)
        return model_dict[model_name].from_pretrained(local_pretrain, model_config)


__all__ = {
    register_model,
    get_model,
    PatchTST,
    DLinear,
    Informer,
    LTSM,
    LTSM_WordPrompt,
    LTSM_Tokenizer
}