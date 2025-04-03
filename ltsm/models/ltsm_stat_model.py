import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from .base_config import LTSMConfig
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from transformers import AutoModel, AutoConfig, AutoTokenizer

class LTSM(PreTrainedModel):    
    config_class = LTSMConfig
    def __init__(self, configs, *model_args, **model_kwargs):
        super().__init__(configs)
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len + configs.prompt_len - self.patch_size) // self.stride + 1
        self.d_type = torch.bfloat16
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.configs = configs
        
        if configs.pretrain:
            print("Loading the pretraining weight.")
            self.llm_config = AutoConfig.from_pretrained(configs.model_name_or_path)
            self.llm = AutoModel.from_pretrained(configs.model_name_or_path)  # loads a pretrained GPT-2 base model
        else:
            raise NotImplementedError("You must load the pretraining weight.")

        self.model_prune(configs)
        print("model = {}".format(self.llm))

        self.in_layer = nn.Linear(configs.patch_size, self.llm_config.hidden_size)
        self.out_layer = nn.Linear(self.llm_config.hidden_size * self.patch_num, configs.pred_len)
        
        self.cnt = 0

    def model_prune(self, configs):
        if "gpt2" in configs.model_name_or_path:
            self.llm.h = self.llm.h[:configs.gpt_layers]
        elif "phi" in configs.model_name_or_path or "llama" in configs.model_name_or_path or "gemma" in configs.model_name_or_path:
            self.llm.layers = self.llm.layers[:configs.gpt_layers]
        else:
            raise NotImplementedError(f"No implementation in model prune for {self.llm}.")
    
    def forward(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()

        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach()
        x /= stdev
        x = rearrange(x, 'b l m -> b m l')
        
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        outputs = self.in_layer(x).to(dtype=torch.bfloat16)

        outputs = self.llm(inputs_embeds=outputs).last_hidden_state
        outputs = outputs.to(dtype=x.dtype)
        
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
