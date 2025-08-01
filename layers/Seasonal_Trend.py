import torch
import torch.nn as nn
from .Autoformer_EncDec import series_decomp
from .Embed import DataEmbedding_inverted

class TrendSeasonalScorer(nn.Module):
    def __init__(self, auxiliary, seq_len, d_model, embed, freq, dropout, kernel_sizes = 8):
        super().__init__()
        self.auxiliary = auxiliary
        if auxiliary:
            self.decompsition = series_decomp(kernel_sizes)
            self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)

            self.trend_proj = nn.Linear(d_model, 1)
            self.seasonal_proj = nn.Linear(d_model, 1)
        self.emb_proj = nn.Linear(d_model, 1)

    def forward(self, x, enc):
        # x: [batch, time_steps, variates]
        scores = self.emb_proj(enc).squeeze(-1)
        if self.auxiliary:
            seasonal, trends = self.decompsition(x)
            seasonal, trends = self.enc_embedding(seasonal, trends)
            
            trends_score = self.trend_proj(trends).squeeze(-1)  # [batch, variates]
            seasonal_score = self.seasonal_proj(seasonal).squeeze(-1)  # [batch, variates]
            scores = scores + trends_score + seasonal_score
        return scores