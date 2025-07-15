# code from https://github.com/yuqinie98/PatchTST, with minor modifications
import torch
from torch import Tensor

from .base_config import PatchTSTConfig
from ltsm.layers.PatchTST_backbone import PatchTST_backbone
from ltsm.layers.PatchTST_layers import series_decomp
from transformers import PreTrainedModel

class PatchTST(PreTrainedModel):
    config_class = PatchTSTConfig
    
    def __init__(self, config: PatchTSTConfig, **kwargs):
        super().__init__(config)

        self.decomposition = config.decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(config.kernel_size)
            self.model_trend = PatchTST_backbone(
                config.enc_in,
                config.seq_len,
                config.pred_len,
                config.patch_len, 
                config.stride,
                config.max_seq_len,
                config.n_layers,
                config.d_model,
                config.n_heads,
                config.d_k,
                config.d_v,
                config.d_ff,
                config.norm,
                config.attn_dropout,
                config.dropout,
                config.activation,
                config.key_padding_mask,
                config.padding_var,
                config.attn_mask,
                config.res_attention,
                config.pre_norm,
                config.store_attn,
                config.pe,
                config.learn_pe,
                config.fc_dropout,
                config.head_dropout,
                config.padding_patch,
                config.pretrain_head,
                config.head_type,
                config.individual,
                config.revin,
                config.affine,
                config.subtract_last,
                config.verbose
            )
            self.model_res = PatchTST_backbone(
                config.enc_in, 
                config.seq_len, 
                config.pred_len, 
                config.patch_len, 
                config.stride,
                config.max_seq_len,
                config.n_layers,
                config.d_model,
                config.n_heads,
                config.d_k,
                config.d_v,
                config.d_ff,
                config.norm,
                config.attn_dropout,
                config.dropout,
                config.activation,
                config.key_padding_mask,
                config.padding_var,
                config.attn_mask,
                config.res_attention,
                config.pre_norm,
                config.store_attn,
                config.pe,
                config.learn_pe,
                config.fc_dropout,
                config.head_dropout,
                config.padding_patch,
                config.pretrain_head,
                config.head_type,
                config.individual,
                config.revin,
                config.affine,
                config.subtract_last,
                config.verbose
            )
        else:
            self.model = PatchTST_backbone(
                config.enc_in, 
                config.seq_len, 
                config.pred_len, 
                config.patch_len, 
                config.stride,
                config.max_seq_len,
                config.n_layers,
                config.d_model,
                config.n_heads,
                config.d_k,
                config.d_v,
                config.d_ff,
                config.norm,
                config.attn_dropout,
                config.dropout,
                config.activation,
                config.key_padding_mask,
                config.padding_var,
                config.attn_mask,
                config.res_attention,
                config.pre_norm,
                config.store_attn,
                config.pe,
                config.learn_pe,
                config.fc_dropout,
                config.head_dropout,
                config.padding_patch,
                config.pretrain_head,
                config.head_type,
                config.individual,
                config.revin,
                config.affine,
                config.subtract_last,
                config.verbose
            )

    def forward(self, x: Tensor):
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)  # [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)  # [Batch, Input length, Channel]
        else:
            x = x.permute(0, 2, 1)  # [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0, 2, 1)  # [Batch, Input length, Channel]
        return x