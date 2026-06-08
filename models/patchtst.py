"""PatchTST 模型实现

核心创新：
1. Patching：将时间序列切分为 patches（类似 ViT 处理图像）
2. Channel Independence：每个变量独立建模（不混合不同变量）
3. Transformer 编码器处理 patches

优势：
- 减少序列长度（L/P 个 patches 而不是 L 个时间点）
- 保留局部语义（一个 patch 包含相邻时间点）
- 减少变量间的干扰
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PatchEmbedding(nn.Module):
    """Patch 嵌入层

    将时间序列切分为固定长度的 patches，然后投影到 d_model 维度。

    输入: (B, L, 1)  # 单变量
    输出: (B, num_patches, d_model)
    """

    def __init__(self, patch_len, stride, d_model, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

        # 使用卷积实现 patch 切分和嵌入
        self.projection = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, L, 1) - 单变量时间序列
        Returns:
            patches: (B, num_patches, d_model)
        """
        # 转换为 (B, 1, L) 用于 Conv1d
        x = x.transpose(1, 2)

        # 卷积切分 + 嵌入
        patches = self.projection(x)  # (B, d_model, num_patches)
        patches = patches.transpose(1, 2)  # (B, num_patches, d_model)

        patches = self.norm(patches)
        patches = self.dropout(patches)

        return patches


class PositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, max_len, d_model):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        """x: (B, L, D)"""
        return x + self.position_embedding[:, :x.shape[1], :]


class PatchTSTBlock(nn.Module):
    """PatchTST Transformer 块"""

    def __init__(self, d_model, n_heads, d_ff=256, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
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
        # 自注意力
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class PatchTSTEncoder(nn.Module):
    """PatchTST 编码器"""

    def __init__(self, d_model, n_heads, n_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            PatchTSTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FlattenHead(nn.Module):
    """展平头：将 patches 展平后映射到预测长度"""

    def __init__(self, n_vars, d_model, patch_len, horizon, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.patch_len = patch_len
        self.horizon = horizon

        # 预测头
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(d_model * patch_len, horizon),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: (B, num_patches, d_model) - 单变量
        Returns:
            output: (B, horizon)
        """
        # 取最后 patch_len 个 patches
        x = x[:, -self.patch_len:, :]  # (B, patch_len, d_model)
        output = self.head(x)  # (B, horizon)
        return output


class PatchTSTModel(nn.Module):
    """PatchTST 时序预测模型

    输入: (batch, lookback, features)
    输出: (batch, horizon, features)

    核心创新：
    1. Patching：将时间序列切分为 patches
    2. Channel Independence：每个变量独立建模

    对于多变量数据，模型对每个变量独立运行相同的网络。
    """

    def __init__(self, input_size, d_model=128, n_heads=8, n_layers=3,
                 d_ff=256, patch_len=16, stride=8, dropout=0.1, horizon=96):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride

        # 计算 patch 数量
        # 假设 lookback=96, patch_len=16, stride=8
        # num_patches = (96 - 16) / 8 + 1 = 11
        self.num_patches = (96 - patch_len) // stride + 1  # 默认 lookback=96

        # Patch 嵌入（共享权重）
        self.patch_embedding = PatchEmbedding(patch_len, stride, d_model, dropout)

        # 位置编码
        self.position_encoding = PositionalEncoding(self.num_patches, d_model)

        # Transformer 编码器（共享权重）
        self.encoder = PatchTSTEncoder(d_model, n_heads, n_layers, d_ff, dropout)

        # 预测头（每个变量一个头，或者共享）
        # 这里使用共享的 FlattenHead
        self.head = FlattenHead(input_size, d_model, min(4, self.num_patches), horizon, dropout)

        # 或者使用简单的全局平均池化 + 线性层
        self.simple_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B, d_model, 1)
            nn.Flatten(),
            nn.Linear(d_model, horizon)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, lookback, features) - 多变量时间序列
        Returns:
            output: (batch, horizon, features)

        Channel Independence：对每个变量独立处理
        """
        B, L, C = x.shape

        # Channel Independence：每个变量仍独立建模，但合并到 batch 维度一次性计算。
        # (B, L, C) -> (B, C, L) -> (B*C, L, 1)
        x = x.permute(0, 2, 1).contiguous().view(B * C, L, 1)

        patches = self.patch_embedding(x)  # (B*C, num_patches, d_model)
        patches = self.position_encoding(patches)
        encoded = self.encoder(patches)  # (B*C, num_patches, d_model)

        encoded_t = encoded.transpose(1, 2)
        pred = self.simple_head(encoded_t)  # (B*C, horizon)

        output = pred.view(B, C, self.horizon).permute(0, 2, 1).contiguous()
        return output


class PatchTSTConvModel(nn.Module):
    """PatchTST 变体：使用卷积替代部分注意力

    对于较短的序列，纯注意力可能不是最优的。
    这个变体混合了卷积和注意力。
    """

    def __init__(self, input_size, d_model=128, n_heads=8, n_layers=3,
                 d_ff=256, patch_len=16, stride=8, dropout=0.1, horizon=96):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (96 - patch_len) // stride + 1

        # Patch 嵌入
        self.patch_embedding = PatchEmbedding(patch_len, stride, d_model, dropout)

        # 卷积特征提取
        self.conv_layers = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),  # 深度可分离
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
            nn.GELU(),
        )

        # Transformer 编码器
        self.encoder = PatchTSTEncoder(d_model, n_heads, max(1, n_layers - 1), d_ff, dropout)

        # 预测头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, horizon)
        )

    def forward(self, x):
        B, L, C = x.shape
        predictions = []

        for i in range(C):
            x_var = x[:, :, i:i+1]

            # Patch 嵌入
            patches = self.patch_embedding(x_var)  # (B, num_patches, d_model)

            # 卷积特征提取
            conv_out = self.conv_layers(patches.transpose(1, 2)).transpose(1, 2)
            patches = patches + conv_out  # 残差

            # Transformer
            encoded = self.encoder(patches)

            # 预测
            pred = self.head(encoded.transpose(1, 2))
            predictions.append(pred)

        output = torch.stack(predictions, dim=-1)
        return output
