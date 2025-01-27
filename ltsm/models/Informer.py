# code from https://github.com/yuqinie98/PatchTST, with minor modifications
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from ltsm.utils.masking import TriangularCausalMask, ProbMask
from ltsm.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from ltsm.layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from ltsm.layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp
import numpy as np
from transformers import PreTrainedModel
from .ltsm_base import LTSMConfig

class Informer(PreTrainedModel):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    config_class = LTSMConfig

    def __init__(self, config: LTSMConfig, **kwargs):
        super().__init__(config)
        self.pred_len = config.pred_len
        self.output_attention = config.output_attention

        # Embedding
        if config.embed_type == 0:
            self.enc_embedding = DataEmbedding(config.enc_in, config.d_model, config.embed, config.freq,
                                            config.dropout)
            self.dec_embedding = DataEmbedding(config.dec_in, config.d_model, config.embed, config.freq,
                                           config.dropout)
        elif config.embed_type == 1:
            self.enc_embedding = DataEmbedding(config.enc_in, config.d_model, config.embed, config.freq,
                                                    config.dropout)
            self.dec_embedding = DataEmbedding(config.dec_in, config.d_model, config.embed, config.freq,
                                                    config.dropout)
        elif config.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(config.enc_in, config.d_model, config.embed, config.freq,
                                                    config.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(config.dec_in, config.d_model, config.embed, config.freq,
                                                    config.dropout)

        elif config.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(config.enc_in, config.d_model, config.embed, config.freq,
                                                    config.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(config.dec_in, config.d_model, config.embed, config.freq,
                                                    config.dropout)
        elif config.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(config.enc_in, config.d_model, config.embed, config.freq,
                                                    config.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(config.dec_in, config.d_model, config.embed, config.freq,
                                                    config.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, config.factor, attention_dropout=config.dropout,
                                      output_attention=config.output_attention),
                        config.d_model, config.n_heads),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation
                ) for l in range(config.e_layers)
            ],
            [
                ConvLayer(
                    config.d_model
                ) for l in range(config.e_layers - 1)
            ] if config.distil else None,
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, config.factor, attention_dropout=config.dropout, output_attention=False),
                        config.d_model, config.n_heads),
                    AttentionLayer(
                        ProbAttention(False, config.factor, attention_dropout=config.dropout, output_attention=False),
                        config.d_model, config.n_heads),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
            projection=nn.Linear(config.d_model, config.c_out, bias=True)
        )

    def forward(self, x_enc: Tensor, x_mark_enc: Tensor, x_dec: Tensor, x_mark_dec: Tensor,
                enc_self_mask: Tensor=None, dec_self_mask: Tensor=None, dec_enc_mask: Tensor=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]