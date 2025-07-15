from ltsm.data_provider.tokenizer.standard_scaler import StandardScaler

processor_dict = {}

def register_processor(module):
    assert module.module_id not in processor_dict, f"Processor {module.module_id} alreader registered"
    processor_dict[module.module_id] = module

register_processor(StandardScaler)
