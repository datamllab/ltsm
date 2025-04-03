from transformers import PretrainedConfig
from dataclasses import dataclass
from typing import Optional
from torch import Tensor

@dataclass
class LTSMConfig(PretrainedConfig):
    """
    LTSMConfig is a configuration class for the LTSM model.
    It contains all the necessary parameters to initialize the model.
    """

    def __init__(self, seq_len: int=336, pred_len: int=96, patch_size: int=16, pretrain: bool=True, stride: int=8, prompt_len: int=133, 
                 gpt_layers: int=3, model_name_or_path: str="gpt2-medium", d_ff: int=512, d_model: int=1024, enc_in: int=1, 
                 dropout: float=0.2, n_heads: int=16, prompt_data_path: str=None, **kwargs):
        
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.pretrain = pretrain
        self.stride = stride
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.prompt_len = prompt_len
        self.gpt_layers = gpt_layers
        self.model_name_or_path = model_name_or_path
        self.d_ff = d_ff
        self.d_model = d_model
        self.enc_in = enc_in
        self.dropout = dropout
        self.n_heads = n_heads
        self.prompt_data_path = prompt_data_path


@dataclass
class DLinearConfig(PretrainedConfig):
    """
    DLinearConfig is a configuration class for the DLinear model.
    It contains all the necessary parameters to initialize the model.
    """

    def __init__(self, seq_len: int=336, pred_len: int=96, individual: bool=0, enc_in: int=1, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.enc_in = enc_in

@dataclass
class InformerConfig(PretrainedConfig):
    """
    InformerConfig is a configuration class for the Informer model.
    It contains all the necessary parameters to initialize the model.
    """

    def __init__(self, seq_len=336, pred_len=96, enc_in=1, dec_in=7, d_model=1024, n_heads=16, e_layers=2, d_ff=512,
                 dropout=0.2, activation='gelu', output_attention=False, embed_type=0, freq='h', factor=1, 
                 distil=True, c_out=862, embed='timeF', **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.output_attention = output_attention
        self.embed_type = embed_type
        self.factor = factor
        self.freq = freq
        self.distil = distil
        self.c_out = c_out
        self.embed = embed


@dataclass
class PatchTSTConfig(PretrainedConfig):
    """
    PatchTSTConfig is a configuration class for the PatchTST model.
    It contains all the necessary parameters to initialize the model.
    """

    def __init__(self, seq_len=336, pred_len=96, enc_in=1, patch_len=16, stride=8, decomposition=False, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, embed='timeF', **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.stride = stride
        self.decomposition = decomposition
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.norm = norm
        self.attn_dropout = attn_dropout
        self.dropout = dropout
        self.activation = act
        self.key_padding_mask = key_padding_mask
        self.padding_var = padding_var
        self.attn_mask = attn_mask
        self.res_attention = res_attention
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.pe = pe
        self.learn_pe = learn_pe
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.padding_patch = padding_patch
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.verbose = verbose,
        self.embed = embed
        