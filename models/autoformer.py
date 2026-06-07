"""轻量版 Autoformer 模型实现。

这个版本面向本项目的 96 回看窗口和 96/168/336 预测步长，目标是让
ETTh1、ETTm1、ECL 都能在 notebook 中跑完可验证的训练轮次。

保留的核心思想：
1. Series Decomposition：显式分解趋势和季节性成分。
2. Auto-Correlation：用 FFT 找主要周期延迟，再做时间延迟聚合。
3. Direct Forecast：用一次性时间投影替代昂贵的解码器注意力。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    """移动平均时序分解。

    Args:
        kernel_size: 移动平均窗口。偶数会自动加 1，保证输出长度稳定。
    """

    def __init__(self, kernel_size=25):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x):
        """分解输入。

        Args:
            x: (B, L, C)
        Returns:
            trend: (B, L, C)
            seasonal: (B, L, C)
        """
        padding = (self.kernel_size - 1) // 2
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (padding, padding), mode="replicate")
        trend = self.avg_pool(x_pad).transpose(1, 2)
        seasonal = x - trend
        return trend, seasonal


class AutoCorrelation(nn.Module):
    """快速 Auto-Correlation 层。

    原始逐 batch/head 的 top-k delay gather 在 CPU 或大变量数据集上开销较大。
    这里改为按 batch/head 平均后的全局 top-k lag，并用 torch.roll 聚合，计算量
    更稳定，仍然通过 FFT 选择主要周期。
    """

    def __init__(self, d_model, n_heads, factor=3, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = max(1, int(factor))

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Args: x: (B, L, D)."""
        B, L, _ = x.shape
        H = self.n_heads
        D = self.d_head

        q = self.W_Q(x).view(B, L, H, D).transpose(1, 2)
        k = self.W_K(x).view(B, L, H, D).transpose(1, 2)
        v = self.W_V(x).view(B, L, H, D).transpose(1, 2)

        q_mean = q.mean(dim=-1)
        k_mean = k.mean(dim=-1)

        q_fft = torch.fft.rfft(q_mean, n=2 * L, dim=-1)
        k_fft = torch.fft.rfft(k_mean, n=2 * L, dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=2 * L, dim=-1)[..., :L]

        if L > 1:
            corr_score = corr.mean(dim=(0, 1))
            corr_score = corr_score.clone()
            corr_score[0] = -torch.inf
            top_k = min(self.factor, L - 1)
            top_score, top_lag = torch.topk(corr_score, k=top_k, dim=-1)
            weights = F.softmax(top_score, dim=-1)
        else:
            top_lag = torch.zeros(1, dtype=torch.long, device=x.device)
            weights = torch.ones(1, dtype=x.dtype, device=x.device)

        mixed = torch.zeros_like(v)
        for lag, weight in zip(top_lag.tolist(), weights):
            mixed = mixed + weight * torch.roll(v, shifts=-lag, dims=2)

        mixed = self.dropout(mixed)
        mixed = mixed.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_O(mixed)


class AutoformerEncoderLayer(nn.Module):
    """轻量 Autoformer 编码层。"""

    def __init__(self, d_model, n_heads, d_ff=128, factor=3, dropout=0.1, kernel_size=25):
        super().__init__()
        self.decomposition1 = SeriesDecomposition(kernel_size)
        self.decomposition2 = SeriesDecomposition(kernel_size)

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
        trend1, seasonal = self.decomposition1(x)
        seasonal = self.norm1(seasonal + self.auto_corr(seasonal))
        seasonal = self.norm2(seasonal + self.ffn(seasonal))
        trend2, seasonal = self.decomposition2(seasonal)
        return seasonal, trend1 + trend2


class AutoformerEncoder(nn.Module):
    """Autoformer 编码器。"""

    def __init__(
        self,
        d_model,
        n_heads,
        n_layers=2,
        d_ff=128,
        factor=3,
        dropout=0.1,
        kernel_size=25,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                AutoformerEncoderLayer(d_model, n_heads, d_ff, factor, dropout, kernel_size)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        trends = []
        for layer in self.layers:
            x, trend = layer(x)
            trends.append(trend)
        return x, trends


class AutoformerDecoder(nn.Module):
    """Direct-forecast 解码器。

    兼容旧构造参数，但内部不再执行 horizon 上的自注意力和交叉注意力。
    """

    def __init__(
        self,
        d_model,
        n_heads,
        horizon,
        n_layers=1,
        d_ff=128,
        factor=3,
        dropout=0.1,
        kernel_size=25,
        lookback=96,
    ):
        super().__init__()
        del n_heads, factor, kernel_size
        self.horizon = horizon
        self.lookback = lookback
        self.time_projection = nn.Linear(lookback, horizon)
        self.refine_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout),
                )
                for _ in range(max(1, n_layers))
            ]
        )

    def _fit_length(self, x):
        if x.shape[1] == self.lookback:
            return x
        return F.interpolate(
            x.transpose(1, 2),
            size=self.lookback,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, enc_out):
        enc_out = self._fit_length(enc_out)
        dec_out = self.time_projection(enc_out.transpose(1, 2)).transpose(1, 2)
        for layer in self.refine_layers:
            dec_out = dec_out + layer(dec_out)
        return dec_out, []


class AutoformerModel(nn.Module):
    """轻量 Autoformer 时序预测模型。

    输入: (batch, lookback, features)
    输出: (batch, horizon, features)
    """

    def __init__(
        self,
        input_size,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=1,
        d_ff=128,
        factor=3,
        dropout=0.1,
        horizon=96,
        kernel_size=25,
        lookback=96,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model 必须能被 n_heads 整除")

        self.input_size = input_size
        self.d_model = d_model
        self.horizon = horizon
        self.lookback = lookback

        self.decomposition = SeriesDecomposition(kernel_size)
        self.input_projection = nn.Linear(input_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "positional_encoding",
            self._generate_positional_encoding(lookback, d_model),
            persistent=False,
        )

        self.encoder = AutoformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            kernel_size=kernel_size,
        )
        self.decoder = AutoformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            horizon=horizon,
            n_layers=n_decoder_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            kernel_size=kernel_size,
            lookback=lookback,
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
            x.transpose(1, 2),
            size=self.lookback,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, x):
        x = self._fit_length(x)
        trend_init, seasonal_init = self.decomposition(x)

        seasonal = self.input_projection(seasonal_init)
        seasonal = seasonal + self.positional_encoding[:, : seasonal.shape[1], :]
        seasonal = self.dropout(seasonal)

        enc_out, _ = self.encoder(seasonal)
        dec_out, _ = self.decoder(enc_out)
        seasonal_pred = self.seasonal_projection(dec_out)

        trend_pred = self.trend_projection(trend_init.transpose(1, 2)).transpose(1, 2)
        return seasonal_pred + trend_pred
