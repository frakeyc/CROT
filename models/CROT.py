import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.Seasonal_Trend import TrendSeasonalScorer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
import math
import torchsort

# Add registry import
from core.registry import register_model


@register_model("CROT", paper="PatchiTransformer", year=2026)
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size
        self.use_mark = configs.use_mark
        self.use_local_loss = configs.use_local_loss
        self.use_global_loss = configs.use_global_loss
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Sort Score
        self.trend_seasonal_score = TrendSeasonalScorer(configs.use_auxiliary_network, configs.seq_len, configs.d_model, 
                                                        configs.embed, configs.freq, configs.dropout, kernel_sizes = configs.trend_ks)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=True), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=True), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        
    def sort(self, enc_out, x_enc):
        scores = self.trend_seasonal_score(x_enc, enc_out)
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True) # [B,N]
        _, original_indices = torch.sort(sorted_indices, dim=-1, descending=False) # [B,N]
        sorted_x = enc_out[torch.arange(enc_out.shape[0])[:, None],sorted_indices, :]
        return sorted_x, sorted_scores, original_indices, sorted_indices

    def batched_attention_smoothing(self, attn, num_iter=30, alpha=0.2):
        batch_size, N, _ = attn.shape
        device = attn.device
        # 1. 对称化注意力矩阵
        A = (attn + attn.transpose(1, 2)) / 2  # [batch, N, N]
        # 2. 计算度矩阵 D
        deg = A.sum(dim=2)  # [batch, N]
        D_inv_sqrt = torch.diag_embed(1.0 / (deg + 1e-8).sqrt())  # [batch, N, N]
        # 3. 构造归一化拉普拉斯 L = I - D^{-1/2} A D^{-1/2}
        I = torch.eye(N, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        L = I - torch.bmm(torch.bmm(D_inv_sqrt, A), D_inv_sqrt)  # [batch, N, N]
        # 4. 随机初始化信号
        x = torch.randn(batch_size, N, device=device)
        # 5. 多次平滑迭代
        for _ in range(num_iter):
            x = x - alpha * torch.bmm(L, x.unsqueeze(2)).squeeze(2)
        # 6. 去除主分量（全1向量方向）
        x = x - x.mean(dim=1, keepdim=True)
        return x  # [batch_size, N]

    def global_loss(self, attns, scores):
        fiedler = self.batched_attention_smoothing(attns)  # [B,N]
        x_soft_rank = torchsort.soft_rank(fiedler, regularization_strength=0.1)
        y_soft_rank = torchsort.soft_rank(scores, regularization_strength=0.1)
        loss =  torch.mean((x_soft_rank - y_soft_rank) ** 2)
        return loss
    
    def local_loss(self, attns_num, attns_size, scores):
        mask_num = torch.eye(attns_num.shape[1], dtype=torch.bool).unsqueeze(0)  # [1, N, N]
        mask_size = torch.eye(attns_size.shape[1], dtype=torch.bool).unsqueeze(0)  # [1, P, P]
        attns_num[mask_num.expand_as(attns_num)] = 0
        attns_size[mask_size.expand_as(attns_size)] = 0

        attns_num_max = attns_num.max(-1)[0] # [B*P,N]
        attns_size_max = attns_size.max(-1)[0].reshape(-1,attns_num.shape[1],attns_size.shape[1]).transpose(1,2).reshape(-1,attns_num.shape[1]) # [B*P,N]
        idxs = torch.argwhere(attns_num_max - attns_size_max > 0) # [M,2]
        if idxs.shape[0] == 0:
            return torch.tensor(0.0, device=scores.device)
        i, j = idxs[:,0], idxs[:,1]

        attns = attns_num[i,j] - attns_size_max[i,j].unsqueeze(-1) # [M,N]
        dists = (scores.unsqueeze(2)-scores.unsqueeze(1)).abs()[i,j] # [M,N]
        idxs = torch.argwhere(attns > 0) # [M',2]
        if idxs.shape[0] == 0:
            return torch.tensor(0.0, device=scores.device)
        i, j = idxs[:,0], idxs[:,1]

        loss = (attns[i,j] * dists[i,j]).mean()
        return loss

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, _, NO = x_enc.shape
        # Embedding
        enc_out, mark_enc_out = self.enc_embedding(x_enc, x_mark_enc)
        sorted_out, sorted_scores, original_indices, sorted_indices = self.sort(enc_out, x_enc)
        B, N, D = sorted_out.shape

        # Patchify
        patch_size = self.patch_size
        pad_num = math.ceil(N / patch_size) * patch_size - N
        if pad_num:
            enc_patch = torch.cat([sorted_out, sorted_out[:,-pad_num:,:]],1)
            enc_scores = torch.cat([sorted_scores, sorted_scores[:,-pad_num:]],1)
        else:
            enc_patch = sorted_out
            enc_scores = sorted_scores

        enc_patch = enc_patch.reshape(B,-1,patch_size,D)
        _,P,S,_ = enc_patch.shape
        if self.use_mark:
            enc_patch = torch.cat([enc_patch, mark_enc_out.unsqueeze(1).repeat(1,enc_patch.shape[1],1,1)], 2)

        score_patch_size = enc_scores.reshape(B,-1,patch_size).reshape(-1,patch_size)
        score_patch_num = enc_scores.reshape(B,-1,patch_size).transpose(1,2).reshape(B*patch_size,-1)

        # Group Transformer
        enc_out, attns_num, attns_size = self.encoder(enc_patch, attn_mask=None)
        enc_out = enc_out[:,:,:S,:].reshape(B,-1,D)
        
        attns_num = torch.stack(attns_num, 0).mean(0).mean(1).reshape(B,-1,P,P)[:,:S,:,:].reshape(-1,P,P) # [B*P,N,N]
        attns_size = torch.stack(attns_size, 0).mean(0).mean(1)[:,:S,:][:,:,:S] # [B*N,P,P]

        enc_patch = enc_out[torch.arange(B)[:, None],original_indices, :]

        # Axuiliary Loss
        if self.use_global_loss:
            global_size_loss = self.global_loss(attns_size, score_patch_size)
            global_num_loss = self.global_loss(attns_num, score_patch_num)
        else:
            global_size_loss, global_num_loss = torch.tensor(0.0, device=enc_patch.device), torch.tensor(0.0, device=enc_patch.device)
        if self.use_local_loss:
            local_num_loss = self.local_loss(attns_num, attns_size, score_patch_num)
        else:
            local_num_loss = torch.tensor(0.0, device=enc_patch.device)

        dec_out = self.projection(enc_patch).permute(0, 2, 1)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out, [global_num_loss*0.01, global_size_loss, local_num_loss]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, sort_loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :], sort_loss  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
