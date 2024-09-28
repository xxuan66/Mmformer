import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_utils.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from model_utils.SelfAttention_Family import FullAttention, AttentionLayer
from model_utils.Embed import DataEmbedding
from model_utils.Conv_Blocks import Inception_Block_V1
from torch.nn.parallel import parallel_apply

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        ……

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        MutliScaleFeaturesProcessor = [ScaleProcessor(scale, linear) for scale, linear in
                                       zip([self.scale1, self.scale2, self.scale3],
                                           [self.ScaleFeatureLinear1, self.ScaleFeatureLinear2,
                                            self.ScaleFeatureLinear3])]
        ……

        MultiPeriodFeatures = parallel_apply(self.MultiPeriodProcessors,[multiscalef1, multiscalef2, multiscalef3])

        dec_out_final = ……
      
        enc_out_encoder, _ = self.encoder(dec_out_final_pos, attn_mask=None)

        dec_out_final = self.projection(enc_out_encoder)[:, :self.pred_len, :]

        dec_out_final = dec_out_final * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out_final = dec_out_final + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out_final

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out
