"""消融实验模型变体。

用于验证 Autoformer 和 PatchTST 的关键模块贡献：
1. AutoformerNoDecomp   — 关闭序列分解，验证 Series Decomposition
2. AutoformerNoAutocorr — 替换 Auto-Correlation 为标准 MHA，验证 Auto-Correlation
3. PatchTSTNoPatch      — 逐时间点线性投影替代 patch embedding，验证 Patching
4. PatchTSTChannelMix   — 混合多变量输入替代 channel independence，验证 Channel Independence
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Autoformer 消融变体
# ---------------------------------------------------------------------------


class _StandardMHAttention(nn.Module):
    """标准多头自注意力，替代 Auto-Correlation。"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        out, _ = self.mha(x, x, x)
        return out


class AutoformerNoDecompLayer(nn.Module):
    """关闭分解的 Autoformer 编码层。

    与原版区别：不做 SeriesDecomposition，趋势分支输出置零。
    """

    def __init__(self, d_model, n_heads, d_ff=128, factor=3, dropout=0.1, kernel_size=25):
        super().__init__()
        from .autoformer import AutoCorrelation

        self.auto_corr = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 不做分解，直接走 auto_corr + FFN，趋势输出为 0
        h = self.norm1(x + self.auto_corr(x))
        h = self.norm2(h + self.ffn(h))
        return h, torch.zeros_like(x)


class AutoformerNoDecompEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers=2, d_ff=128, factor=3, dropout=0.1, kernel_size=25):
        super().__init__()
        self.layers = nn.ModuleList([
            AutoformerNoDecompLayer(d_model, n_heads, d_ff, factor, dropout, kernel_size)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        trends = []
        for layer in self.layers:
            x, trend = layer(x)
            trends.append(trend)
        return x, trends


class AutoformerNoDecomp(nn.Module):
    """消融：关闭 Series Decomposition。

    输入: (batch, lookback, features)
    输出: (batch, horizon, features)
    """

    def __init__(self, input_size, d_model=64, n_heads=4, n_encoder_layers=2,
                 n_decoder_layers=1, d_ff=128, factor=3, dropout=0.1,
                 horizon=96, kernel_size=25, lookback=96):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")

        self.input_size = input_size
        self.d_model = d_model
        self.horizon = horizon
        self.lookback = lookback

        # 不使用 decomposition，直接投影原序列
        self.input_projection = nn.Linear(input_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "positional_encoding",
            self._generate_positional_encoding(lookback, d_model),
            persistent=False,
        )

        self.encoder = AutoformerNoDecompEncoder(
            d_model, n_heads, n_encoder_layers, d_ff, factor, dropout, kernel_size,
        )
        from .autoformer import AutoformerDecoder
        self.decoder = AutoformerDecoder(
            d_model, n_heads, horizon, n_decoder_layers, d_ff, factor, dropout, kernel_size, lookback,
        )

        self.seasonal_projection = nn.Linear(d_model, input_size)
        self.trend_projection = nn.Linear(lookback, horizon)

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: pe[0, :, 1::2].shape[-1]])
        return pe

    def _fit_length(self, x):
        if x.shape[1] == self.lookback:
            return x
        return F.interpolate(
            x.transpose(1, 2), size=self.lookback, mode="linear", align_corners=False,
        ).transpose(1, 2)

    def forward(self, x):
        x = self._fit_length(x)
        # 不做分解，直接投影
        seasonal = self.input_projection(x)
        seasonal = seasonal + self.positional_encoding[:, :seasonal.shape[1], :]
        seasonal = self.dropout(seasonal)

        enc_out, _ = self.encoder(seasonal)
        dec_out, _ = self.decoder(enc_out)
        seasonal_pred = self.seasonal_projection(dec_out)

        # 趋势分支置零
        trend_pred = torch.zeros(x.shape[0], self.horizon, self.input_size, device=x.device)
        return seasonal_pred + trend_pred


class AutoformerNoAutocorrLayer(nn.Module):
    """用标准 MHA 替代 Auto-Correlation 的 Autoformer 编码层。"""

    def __init__(self, d_model, n_heads, d_ff=128, dropout=0.1, kernel_size=25):
        super().__init__()
        from .autoformer import SeriesDecomposition

        self.decomposition1 = SeriesDecomposition(kernel_size)
        self.decomposition2 = SeriesDecomposition(kernel_size)

        self.mha = _StandardMHAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        trend1, seasonal = self.decomposition1(x)
        seasonal = self.norm1(seasonal + self.mha(seasonal))
        seasonal = self.norm2(seasonal + self.ffn(seasonal))
        trend2, seasonal = self.decomposition2(seasonal)
        return seasonal, trend1 + trend2


class AutoformerNoAutocorrEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers=2, d_ff=128, dropout=0.1, kernel_size=25):
        super().__init__()
        self.layers = nn.ModuleList([
            AutoformerNoAutocorrLayer(d_model, n_heads, d_ff, dropout, kernel_size)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        trends = []
        for layer in self.layers:
            x, trend = layer(x)
            trends.append(trend)
        return x, trends


class AutoformerNoAutocorr(nn.Module):
    """消融：用标准 MHA 替代 Auto-Correlation。

    输入: (batch, lookback, features)
    输出: (batch, horizon, features)
    """

    def __init__(self, input_size, d_model=64, n_heads=4, n_encoder_layers=2,
                 n_decoder_layers=1, d_ff=128, factor=3, dropout=0.1,
                 horizon=96, kernel_size=25, lookback=96):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")

        self.input_size = input_size
        self.d_model = d_model
        self.horizon = horizon
        self.lookback = lookback

        from .autoformer import SeriesDecomposition
        self.decomposition = SeriesDecomposition(kernel_size)
        self.input_projection = nn.Linear(input_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "positional_encoding",
            self._generate_positional_encoding(lookback, d_model),
            persistent=False,
        )

        self.encoder = AutoformerNoAutocorrEncoder(
            d_model, n_heads, n_encoder_layers, d_ff, dropout, kernel_size,
        )
        from .autoformer import AutoformerDecoder
        self.decoder = AutoformerDecoder(
            d_model, n_heads, horizon, n_decoder_layers, d_ff, factor, dropout, kernel_size, lookback,
        )

        self.seasonal_projection = nn.Linear(d_model, input_size)
        self.trend_projection = nn.Linear(lookback, horizon)

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: pe[0, :, 1::2].shape[-1]])
        return pe

    def _fit_length(self, x):
        if x.shape[1] == self.lookback:
            return x
        return F.interpolate(
            x.transpose(1, 2), size=self.lookback, mode="linear", align_corners=False,
        ).transpose(1, 2)

    def forward(self, x):
        x = self._fit_length(x)
        trend_init, seasonal_init = self.decomposition(x)

        seasonal = self.input_projection(seasonal_init)
        seasonal = seasonal + self.positional_encoding[:, :seasonal.shape[1], :]
        seasonal = self.dropout(seasonal)

        enc_out, _ = self.encoder(seasonal)
        dec_out, _ = self.decoder(enc_out)
        seasonal_pred = self.seasonal_projection(dec_out)

        trend_pred = self.trend_projection(trend_init.transpose(1, 2)).transpose(1, 2)
        return seasonal_pred + trend_pred


# ---------------------------------------------------------------------------
# PatchTST 消融变体
# ---------------------------------------------------------------------------


class PatchTSTNoPatch(nn.Module):
    """消融：逐时间点线性投影替代 Patch Embedding。

    不做 patching，每个时间点独立线性投影到 d_model，再经 Transformer 编码。
    保留 Channel Independence。

    输入: (batch, lookback, features)
    输出: (batch, horizon, features)
    """

    def __init__(self, input_size, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, patch_len=16, stride=8, dropout=0.1, horizon=96):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.horizon = horizon
        self.lookback = 96  # 默认回看窗口

        # 逐时间点线性投影替代 patch embedding
        self.point_projection = nn.Linear(1, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, self.lookback, d_model) * 0.02)

        from .patchtst import PatchTSTEncoder
        self.encoder = PatchTSTEncoder(d_model, n_heads, n_layers, d_ff, dropout)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        B, L, C = x.shape
        # Channel Independence: (B, L, C) -> (B*C, L, 1)
        x = x.permute(0, 2, 1).contiguous().view(B * C, L, 1)

        # 逐时间点投影
        h = self.point_projection(x)  # (B*C, L, d_model)
        if h.shape[1] <= self.position_encoding.shape[1]:
            h = h + self.position_encoding[:, :h.shape[1], :]

        encoded = self.encoder(h)  # (B*C, L, d_model)
        pred = self.head(encoded.transpose(1, 2))  # (B*C, horizon)

        output = pred.view(B, C, self.horizon).permute(0, 2, 1).contiguous()
        return output


class PatchTSTChannelMix(nn.Module):
    """消融：混合多变量输入替代 Channel Independence。

    不按变量独立建模，直接将所有变量拼接后投影到 d_model，经 Transformer 编码后
    一次性输出所有变量的预测。

    输入: (batch, lookback, features)
    输出: (batch, horizon, features)
    """

    def __init__(self, input_size, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, patch_len=16, stride=8, dropout=0.1, horizon=96):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.horizon = horizon
        self.lookback = 96

        # 直接投影所有变量
        self.input_projection = nn.Linear(input_size, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1, self.lookback, d_model) * 0.02)

        from .patchtst import PatchTSTEncoder
        self.encoder = PatchTSTEncoder(d_model, n_heads, n_layers, d_ff, dropout)

        # 一次性输出所有变量
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, horizon * input_size),
        )

    def forward(self, x):
        B, L, C = x.shape
        # 直接混合所有变量: (B, L, C) -> (B, L, d_model)
        h = self.input_projection(x)
        if h.shape[1] <= self.position_encoding.shape[1]:
            h = h + self.position_encoding[:, :h.shape[1], :]

        encoded = self.encoder(h)  # (B, L, d_model)
        pred = self.head(encoded.transpose(1, 2))  # (B, horizon * input_size)
        output = pred.view(B, self.horizon, self.input_size)
        return output
