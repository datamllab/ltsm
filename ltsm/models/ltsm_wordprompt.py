import json
import torch
import torch.nn as nn

from .base_config import LTSMConfig
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModel, AutoConfig, AutoTokenizer

from .utils import Normalize, FlattenHead, ReprogrammingLayer
from .embed import PatchEmbedding


class LTSM_WordPrompt(PreTrainedModel):
    config_class = LTSMConfig
    def __init__(self, configs):
        super().__init__(configs)
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.d_model
        self.patch_len = configs.patch_size
        self.stride = configs.stride
        self.pretrain = configs.pretrain

        with open(configs.prompt_data_path, 'r') as f:
            self.index2prompt = json.load(f)
        if configs.pretrain:
            print("Loading the pretraining weight.")
            self.llm_config = AutoConfig.from_pretrained(configs.model_name_or_path)
            self.llm_model = AutoModel.from_pretrained(configs.model_name_or_path)  # loads a pretrained GPT-2 base model
            self.tokenizer = AutoTokenizer.from_pretrained(configs.model_name_or_path)
        else:
            raise NotImplementedError("You must load the pretraining weight.")

        self.model_prune(configs)
        print("model = {}".format(self.llm_model))
            
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    
    def model_prune(self, configs):
        if "gpt2" in configs.model_name_or_path:
            self.llm_model.h = self.llm_model.h[:configs.gpt_layers]
        elif "phi" in configs.model_name_or_path or "llama" in configs.model_name_or_path or "gemma" in configs.model_name_or_path:
            self.llm_model.layers = self.llm_model.layers[:configs.gpt_layers]
        else:
            raise NotImplementedError(f"No implementation in model prune for {self.llm_model}.")
        
        
    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
    
    def forward(self, x_enc):
        index = x_enc[:, 0, 0]
        index = index.tolist()
        x_enc = x_enc[:,1:,:]
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)
        # ipdb.set_trace()
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.index2prompt[str(int(index[b]))]}<|end_prompt|>"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.float32))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]  # (batch, patch_num, d_ff)

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')
        
        return dec_out[:, -self.pred_len:, :]
