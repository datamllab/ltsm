import torch
from .base_config import LTSMConfig
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModel, AutoConfig


class LTSM_Tokenizer(PreTrainedModel):
    config_class = LTSMConfig
    def __init__(self, configs):
        super().__init__(configs)
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain

        self.d_type = torch.bfloat16
        self.pred_len = configs.pred_len    

        if configs.pretrain:
            print("Loading the pretraining weight.")
            self.llm_config = AutoConfig.from_pretrained(configs.model_name_or_path)
            self.llm = AutoModel.from_pretrained(configs.model_name_or_path)  # loads a pretrained GPT-2 base model
        else:
            raise NotImplementedError("You must load the pretraining weight.")

        self.model_prune(configs)
        print("gpt2 = {}".format(self.llm))
            
    def model_prune(self, configs):
        if "gpt2" in configs.model_name_or_path:
            self.llm.h = self.llm.h[:configs.gpt_layers]
        elif "phi" in configs.model_name_or_path or "llama" in configs.model_name_or_path or "gemma" in configs.model_name_or_path:
            self.llm.layers = self.llm.layers[:configs.gpt_layers]
        else:
            raise NotImplementedError(f"No implementation in model prune for {self.llm}.")

    def forward(self, x):
        x = x.int().unsqueeze(-1)
        # x = x.int().to(self.llm.device)
        # import ipdb; ipdb.set_trace()
        outputs = self.llm(input_ids = x).last_hidden_state
        outputs = outputs[:, -self.pred_len:, :]

        return outputs