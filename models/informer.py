"""Informer 模型实现

核心创新：
1. ProbSparse 注意力：O(L log L) 复杂度，只计算 Top-K 重要的 query
2. Self-attention Distilling：逐层压缩序列长度
3. Generative Style Decoder：一次性生成所有预测步
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ProbSparseAttention(nn.Module):
    """ProbSparse 自注意力机制

    核心思想：不是所有 query 都需要计算完整的 attention，
    只选择得分最高的 Top-K 个 query 参与计算，其余用平均值代替。
    """

    def __init__(self, d_model, n_heads, factor=5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor  # 控制 Top-K 的比例

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """计算 ProbSparse 注意力的核心函数

        Args:
            Q: (B, H, L_Q, D)
            K: (B, H, L_K, D)
            sample_k: 采样的 key 数量
            n_top: 选择的 Top-Q 数量
        """
        B, H, L_K, D = K.shape
        _, _, L_Q, _ = Q.shape

        # 随机采样一部分 K 来计算注意力分数
        K_sample_index = torch.randint(L_K, (L_Q, sample_k), device=K.device)
        K_sample = K[:, :, K_sample_index, :]  # (B, H, L_Q, sample_k, D)

        # 计算 Q 与采样 K 的注意力分数
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        # Q_K_sample: (B, H, L_Q, sample_k)

        # 计算每个 query 的最大注意力分数（衡量重要性）
        M = Q_K_sample.max(-1)[0] - Q_K_sample.mean(-1)  # (B, H, L_Q)

        # 选择 Top-K 最重要的 query
        M_top = M.topk(n_top, sorted=False)[1]  # (B, H, n_top)

        return M_top

    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        D = self.d_k

        Q = self.W_Q(x).view(B, L, H, D).transpose(1, 2)  # (B, H, L, D)
        K = self.W_K(x).view(B, L, H, D).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, D).transpose(1, 2)

        # 计算采样数量
        L_K = K.shape[2]
        L_Q = Q.shape[2]
        sample_k = min(int(self.factor * math.log(L_K)), L_K)
        n_top = min(int(self.factor * math.log(L_Q)), L_Q)

        # 获取 Top-Q 的索引
        top_index = self._prob_QK(Q, K, sample_k, n_top)  # (B, H, n_top)

        # 只对 Top-Q 计算完整的注意力
        Q_top = torch.gather(Q, 2, top_index.unsqueeze(-1).expand(-1, -1, -1, D))
        # Q_top: (B, H, n_top, D)

        # 计算完整注意力
        attn = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(D)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        V_top = torch.matmul(attn, V)  # (B, H, n_top, D)

        # 将结果放回原来的位置，其余位置用平均值填充
        output = torch.zeros_like(Q)
        output.scatter_(2, top_index.unsqueeze(-1).expand(-1, -1, -1, D), V_top)
        # 其余位置用 V 的平均值填充
        mean_V = V.mean(dim=2, keepdim=True).expand_as(output)
        mask = torch.ones(B, H, L_Q, 1, device=x.device)
        mask.scatter_(2, top_index.unsqueeze(-1), 0)
        output = output + mean_V * mask

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(B, L_Q, self.d_model)
        output = self.W_O(output)

        return output


class DistillingLayer(nn.Module):
    """Self-attention Distilling 层

    通过卷积下采样逐层压缩序列长度，
    提取最重要的特征，减少计算量。
    """

    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ELU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # x: (B, L, D)
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxpool(x)  # 序列长度减半
        x = x.transpose(1, 2)  # (B, L/2, D)
        return x


class InformerEncoderLayer(nn.Module):
    """Informer 编码器层"""

    def __init__(self, d_model, n_heads, d_ff=256, factor=5, dropout=0.1):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads, factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # ProbSparse 注意力 + 残差
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        # FFN + 残差
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class InformerEncoder(nn.Module):
    """Informer 编码器（多层 + Distilling）"""

    def __init__(self, d_model, n_heads, n_layers=3, d_ff=256, factor=5, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, factor, dropout)
            for _ in range(n_layers)
        ])
        # Distilling 层（最后一层不用）
        self.distilling = nn.ModuleList([
            DistillingLayer(d_model)
            for _ in range(n_layers - 1)
        ])

    def forward(self, x):
        """返回最后一层的输出（不进行 distilling）"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 只在非最后一层进行 distilling（可选）
            # 这里我们不使用 distilling，保持序列长度不变
        return x


class InformerDecoder(nn.Module):
    """轻量生成式解码器。

    保留 Informer 编码器中的 ProbSparse 注意力，将原本 horizon 维度上的
    自注意力/交叉注意力堆叠替换为一次性时间投影，便于 ECL 这类高维数据
    先跑通核心实验。
    """

    def __init__(
        self,
        d_model,
        n_heads,
        horizon,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
        lookback=96,
    ):
        super().__init__()
        del n_heads
        self.horizon = horizon
        self.lookback = lookback
        self.time_projection = nn.Linear(lookback, horizon)
        self.refine_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(max(1, n_layers))
        ])

    def _fit_length(self, x):
        if x.shape[1] == self.lookback:
            return x
        return F.interpolate(
            x.transpose(1, 2),
            size=self.lookback,
            mode='linear',
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, enc_out):
        """
        Args:
            enc_out: 编码器输出 (B, L, D)
        Returns:
            dec_out: 解码器输出 (B, horizon, D)
        """
        enc_out = self._fit_length(enc_out)
        dec_out = self.time_projection(enc_out.transpose(1, 2)).transpose(1, 2)
        for layer in self.refine_layers:
            dec_out = dec_out + layer(dec_out)
        return dec_out


class InformerModel(nn.Module):
    """Informer 时序预测模型

    输入: (batch, lookback, features)
    输出: (batch, horizon, features)

    核心创新：
    1. ProbSparse 注意力：O(L log L) 复杂度
    2. 生成式解码器：一次性生成所有预测步
    """

    def __init__(self, input_size, d_model=128, n_heads=8, n_encoder_layers=3,
                 n_decoder_layers=2, d_ff=256, factor=5, dropout=0.1, horizon=96,
                 lookback=96):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.horizon = horizon
        self.lookback = lookback

        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)

        # 位置编码
        self.positional_encoding = self._generate_positional_encoding(1000, d_model)

        # 编码器
        self.encoder = InformerEncoder(
            d_model, n_heads, n_encoder_layers, d_ff, factor, dropout
        )

        # 解码器
        self.decoder = InformerDecoder(
            d_model, n_heads, horizon, n_decoder_layers, d_ff, dropout, lookback
        )

        # 输出投影
        self.output_projection = nn.Linear(d_model, input_size)

        self.dropout = nn.Dropout(dropout)

    def _generate_positional_encoding(self, max_len, d_model):
        """生成正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, lookback, features)
        Returns:
            output: (batch, horizon, features)
        """
        B, L, _ = x.shape

        # 投影到 d_model 维度
        x = self.input_projection(x)  # (B, L, d_model)

        # 添加位置编码
        x = x + self.positional_encoding[:, :L, :].to(x.device)
        x = self.dropout(x)

        # 编码器
        enc_out = self.encoder(x)  # (B, L, d_model)

        # 解码器（一次性生成）
        dec_out = self.decoder(enc_out)  # (B, horizon, d_model)

        # 输出投影
        output = self.output_projection(dec_out)  # (B, horizon, input_size)

        return output
