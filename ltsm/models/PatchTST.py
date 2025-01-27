# code from https://github.com/yuqinie98/PatchTST, with minor modifications
import torch
from torch import Tensor

from .ltsm_base import LTSMConfig
from ltsm.layers.PatchTST_backbone import PatchTST_backbone
from ltsm.layers.PatchTST_layers import series_decomp
from transformers import PreTrainedModel

class PatchTST(PreTrainedModel):
    config_class = LTSMConfig
    
    def __init__(self, config: LTSMConfig, **kwargs):
        super().__init__(config)

        self.decomposition = config.decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(config.kernel_size)
            self.model_trend = PatchTST_backbone(config.enc_in, 
                                                 config.seq_len, 
                                                 config.pred_len, 
                                                 config.patch_len, 
                                                 config.stride,
                                                 **kwargs)
            self.model_res = PatchTST_backbone(config.enc_in, 
                                               config.seq_len, 
                                               config.pred_len, 
                                               config.patch_len, 
                                               config.stride,
                                               **kwargs)
        else:
            self.model = PatchTST_backbone(config.enc_in, 
                                           config.seq_len, 
                                           config.pred_len, 
                                           config.patch_len, 
                                           config.stride,
                                           **kwargs)

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