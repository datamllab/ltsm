"""
TODO Apr 7, 2025 ~ Apr 13. 2025
- Select different models based on task_type

"""
from ltsm.models.base_config import LTSMConfig
from ltsm.models import get_model
from ltsm.data_provider.prompt_generator import prompt_generate_split, prompt_normalization_split 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from huggingface_hub import login
from ltsm.data_reader.csv_reader import CSVReader
from ltsm.data_provider.tokenizer.standard_scaler import StandardScaler
from pydantic import BaseModel
import os


def inference(file: str, task_type: str = "ts-classification") -> str:
    """
    Currently just a minimal working example.

    Task: according to different task requirements, select different models, and save inference results.

    Models can be selected:
    - LTSM : forecasting
    - DLinear
    - Informer
    - PatchTST
    """
    
    #login(token="Hugging Face Token")  # Login to Hugging Face Hub if needed
    config = LTSMConfig()
    model = get_model(config, "LTSM", local_pretrain=None, hf_hub_model="LSC2204/LTSM-bundle")
    
    task_type = task_type
    files = file.split()
    print(f"[TS Inferencer] Received inference request with task_type: {task_type}")

    dataList = []
    base_path = os.path.join(os.path.dirname(__file__), "cache")
    os.makedirs(base_path, exist_ok=True)
    for index, file in enumerate(files):
        df = CSVReader(file).fetch()
        input_data = df.to_numpy().transpose()
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)
        tensor_data = torch.tensor(input_data, dtype=torch.float32)
        tensor_data = tensor_data.unsqueeze(0)
        tensor_data_length = tensor_data.shape[1]
        # Pad tensor to match pretrained LTSM input size (336 seq_len + 133 prompt_len)
        tensor_data = torch.nn.functional.pad(tensor_data, (0, 0, 133+336-tensor_data_length, 0), mode='constant', value=tensor_data[0, tensor_data_length-1, 0])
        with torch.no_grad():
            model.eval()
            output = model(tensor_data)
        
        output_np = output.squeeze(0).detach().numpy()
        output_path = os.path.join(base_path, f"{index}.csv")
        pd.DataFrame(output_np).to_csv(output_path, index=False)
        dataList.append(output_path)
    LTSM_Output = " ".join(dataList)

    return LTSM_Output




    
#inference()





