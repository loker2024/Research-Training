"""Autoformer 模型实现

核心创新：
1. Series Decomposition：显式分解趋势和季节性成分
2. Auto-Correlation 机制：基于 FFT 的周期性注意力，取代点积注意力
3. Top-K 周期选择：只关注最显著的周期
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SeriesDecomposition(nn.Module):
    """时序分解模块

    将输入分解为趋势（Trend）和季节性（Seasonal）两个成分：
    - 趋势：使用移动平均提取低频变化
    - 季节性：原始序列减去趋势
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        # 使用平均池化实现移动平均
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            trend: (B, L, D) - 趋势成分
            seasonal: (B, L, D) - 季节性成分
        """
        # 移动平均提取趋势
        x_t = x.transpose(1, 2)  # (B, D, L)
        trend = self.avg_pool(x_t)
        # 处理边界：使用镜像填充
        trend = trend[:, :, :x.shape[1]]
        trend = trend.transpose(1, 2)  # (B, L, D)

        # 季节性 = 原始 - 趋势
        seasonal = x - trend

        return trend, seasonal


class AutoCorrelation(nn.Module):
    """Auto-Correlation 机制

    核心思想：
    1. 使用 FFT 计算序列的自相关系数
    2. 选择 Top-K 最显著的周期
    3. 按照这些周期进行时间延迟聚合（向量化实现）

    复杂度：O(L log L)（FFT）
    """

    def __init__(self, d_model, n_heads, factor=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.factor = factor  # Top-K 周期数量

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def time_delay_agg(self, V, top_indices, top_weights):
        """时间延迟聚合（向量化实现，无循环）

        Args:
            V: (B, H, L, D) - 值
            top_indices: (B, H, top_k) - Top-K 周期索引
            top_weights: (B, H, top_k) - Top-K 权重
        """
        B, H, L, D = V.shape
        top_k = top_indices.shape[2]

        # 预计算所有可能的移位索引
        # 创建基础索引: (L,)
        base_idx = torch.arange(L, device=V.device)

        # 初始化输出
        output = torch.zeros_like(V)

        # 向量化处理每个 top_k
        for i in range(top_k):
            # 获取当前周期: (B, H)
            period = top_indices[:, :, i]
            weight = top_weights[:, :, i].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)

            # 计算移位索引: (B, H, L)
            # 使用广播: (B, H, 1) - (L,) -> (B, H, L)
            shift_idx = (base_idx.unsqueeze(0).unsqueeze(0) - period.unsqueeze(-1)) % L

            # 使用 gather 进行向量化移位
            # 首先将 V 展平为 (B*H, L, D)
            V_flat = V.reshape(B * H, L, D)
            shift_idx_flat = shift_idx.reshape(B * H, L).unsqueeze(-1).expand(-1, -1, D)

            # Gather: (B*H, L, D)
            shifted_flat = torch.gather(V_flat, 1, shift_idx_flat)
            shifted = shifted_flat.reshape(B, H, L, D)

            # 加权累加
            output = output + weight * shifted

        return output

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        """
        B, L, _ = x.shape
        H = self.n_heads
        D = self.d_model // self.n_heads

        Q = self.W_Q(x).view(B, L, H, D).transpose(1, 2)  # (B, H, L, D)
        K = self.W_K(x).view(B, L, H, D).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, D).transpose(1, 2)

        # 使用 FFT 计算自相关（按头计算，不展平 D 维度）
        # Q: (B, H, L, D) -> 需要计算每个头的自相关

        # 简化方案：对每个头的 Q 和 K 求均值后计算自相关
        # 或者使用标准注意力作为快速近似
        Q_mean = Q.mean(dim=-1)  # (B, H, L)
        K_mean = K.mean(dim=-1)  # (B, H, L)

        # FFT
        Q_fft = torch.fft.rfft(Q_mean, n=2 * L)
        K_fft = torch.fft.rfft(K_mean, n=2 * L)

        # 互功率谱
        cross_power = Q_fft * torch.conj(K_fft)
        corr = torch.fft.irfft(cross_power, n=2 * L)[:, :L]  # (B, H, L)

        # Top-K 周期选择
        top_k = min(int(self.factor * math.log(L)), L // 2)
        top_k = max(top_k, 1)  # 至少选择 1 个

        # 选择 Top-K 周期（跳过 lag=0）
        corr[:, :, 0] = 0  # 零延迟不感兴趣
        top_corr, top_indices = corr.topk(top_k, dim=-1)  # (B, H, top_k)
        top_weights = F.softmax(top_corr, dim=-1)

        # 时间延迟聚合（向量化）
        output = self.time_delay_agg(V, top_indices, top_weights)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.W_O(output)

        return output


class AutoformerEncoderLayer(nn.Module):
    """Autoformer 编码器层

    结构：
    1. 序列分解 -> 季节性部分进入 Auto-Correlation
    2. Auto-Correlation 注意力
    3. FFN
    4. 再次序列分解
    """

    def __init__(self, d_model, n_heads, d_ff=256, factor=3, dropout=0.1, kernel_size=25):
        super().__init__()
        self.decomposition1 = SeriesDecomposition(kernel_size)
        self.decomposition2 = SeriesDecomposition(kernel_size)

        self.auto_corr = AutoCorrelation(d_model, n_heads, factor, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            x: (B, L, D) - 更新后的季节性成分
            trend: (B, L, D) - 提取的趋势成分
        """
        # 第一次分解
        trend1, seasonal1 = self.decomposition1(x)

        # Auto-Correlation 注意力
        attn_out = self.auto_corr(seasonal1)
        seasonal1 = self.norm1(seasonal1 + attn_out)

        # FFN
        ffn_out = self.ffn(seasonal1)
        seasonal2 = self.norm2(seasonal1 + ffn_out)

        # 第二次分解
        trend2, seasonal_out = self.decomposition2(seasonal2)

        # 累积趋势
        trend = trend1 + trend2

        return seasonal_out, trend


class AutoformerEncoder(nn.Module):
    """Autoformer 编码器"""

    def __init__(self, d_model, n_heads, n_layers=3, d_ff=256, factor=3,
                 dropout=0.1, kernel_size=25):
        super().__init__()
        self.layers = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, d_ff, factor, dropout, kernel_size)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        """返回最终季节性输出和累积趋势"""
        trends = []
        for layer in self.layers:
            x, trend = layer(x)
            trends.append(trend)
        return x, trends


class AutoformerDecoder(nn.Module):
    """Autoformer 解码器

    结构与编码器类似，但包含交叉注意力。
    """

    def __init__(self, d_model, n_heads, horizon, n_layers=2, d_ff=256,
                 factor=3, dropout=0.1, kernel_size=25):
        super().__init__()
        self.horizon = horizon
        self.d_model = d_model

        # 可学习的查询 token
        self.query_token = nn.Parameter(torch.randn(1, horizon, d_model))

        # 交叉注意力层（标准多头注意力，更高效）
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'decomposition1': SeriesDecomposition(kernel_size),
                'decomposition2': SeriesDecomposition(kernel_size),
                'self_auto_corr': AutoCorrelation(d_model, n_heads, factor, dropout),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                    nn.Dropout(dropout)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model),
            }))

    def forward(self, enc_out):
        """
        Args:
            enc_out: 编码器输出 (B, L_enc, D)
        Returns:
            dec_out: (B, horizon, D)
            trends: 趋势列表
        """
        B = enc_out.shape[0]

        # 初始化解码器输入
        dec_out = self.query_token.expand(B, -1, -1)

        trends = []
        for layer in self.layers:
            # 分解
            trend1, seasonal1 = layer['decomposition1'](dec_out)
            trends.append(trend1)

            # 自注意力（Auto-Correlation）
            self_attn = layer['self_auto_corr'](seasonal1)
            seasonal1 = layer['norm1'](seasonal1 + self_attn)

            # 交叉注意力（标准多头注意力）
            # Q: seasonal1 (B, horizon, D), K/V: enc_out (B, L_enc, D)
            cross_out, _ = self.cross_attention(seasonal1, enc_out, enc_out)
            seasonal2 = layer['norm2'](seasonal1 + cross_out)

            # FFN
            ffn_out = layer['ffn'](seasonal2)
            seasonal3 = layer['norm3'](seasonal2 + ffn_out)

            # 再次分解
            trend2, dec_out = layer['decomposition2'](seasonal3)
            trends.append(trend2)

        return dec_out, trends


class AutoformerModel(nn.Module):
    """Autoformer 时序预测模型

    输入: (batch, lookback, features)
    输出: (batch, horizon, features)

    核心创新：
    1. 序列分解：显式建模趋势和季节性
    2. Auto-Correlation：基于 FFT 的周期性注意力
    """

    def __init__(self, input_size, d_model=128, n_heads=8, n_encoder_layers=3,
                 n_decoder_layers=2, d_ff=256, factor=3, dropout=0.1,
                 horizon=96, kernel_size=25):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.horizon = horizon

        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)

        # 位置编码
        self.positional_encoding = self._generate_positional_encoding(1000, d_model)

        # 编码器
        self.encoder = AutoformerEncoder(
            d_model, n_heads, n_encoder_layers, d_ff, factor, dropout, kernel_size
        )

        # 解码器
        self.decoder = AutoformerDecoder(
            d_model, n_heads, horizon, n_decoder_layers, d_ff, factor, dropout, kernel_size
        )

        # 趋势分解（用于最终预测）
        self.final_decomposition = SeriesDecomposition(kernel_size)

        # 趋势预测头
        self.trend_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, input_size)
        )

        # 季节性预测头
        self.seasonal_projection = nn.Linear(d_model, input_size)

        self.dropout = nn.Dropout(dropout)

    def _generate_positional_encoding(self, max_len, d_model):
        """生成正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x: (batch, lookback, features)
        Returns:
            output: (batch, horizon, features)
        """
        B, L, _ = x.shape

        # 分解输入
        trend_init, seasonal_init = self.final_decomposition(x)

        # 投影
        seasonal = self.input_projection(seasonal_init)
        seasonal = seasonal + self.positional_encoding[:, :L, :].to(seasonal.device)
        seasonal = self.dropout(seasonal)

        # 编码器
        enc_out, enc_trends = self.encoder(seasonal)

        # 解码器
        dec_out, dec_trends = self.decoder(enc_out)

        # 季节性预测
        seasonal_pred = self.seasonal_projection(dec_out)

        # 趋势预测：使用编码器和解码器的趋势
        # 简单实现：对趋势进行线性外推
        trend_mean = trend_init.mean(dim=1, keepdim=True).expand(-1, self.horizon, -1)
        trend_pred = self.trend_projection(
            self.input_projection(trend_mean)
        )

        # 最终预测 = 趋势 + 季节性
        output = trend_pred + seasonal_pred

        return output
