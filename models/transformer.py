"""Transformer 基线模型"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """添加位置编码

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器层"""

    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """前向传播

        Args:
            src: (batch, seq_len, d_model)
            src_mask: (seq_len, seq_len)
            src_key_padding_mask: (batch, seq_len)

        Returns:
            src: (batch, seq_len, d_model)
        """
        # 自注意力
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class TransformerModel(nn.Module):
    """Transformer 时序预测模型"""

    def __init__(self, input_size, d_model=128, nhead=8, num_layers=2,
                 dim_feedforward=256, dropout=0.1, horizon=96):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.horizon = horizon

        # 输入投影层
        self.input_proj = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer 编码器层
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.ModuleList(
            [encoder_layer for _ in range(num_layers)]
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_size * horizon)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        """前向传播

        Args:
            x: (batch, lookback, features)

        Returns:
            output: (batch, horizon, features)
        """
        batch_size = x.size(0)

        # 输入投影
        x = self.input_proj(x)  # (batch, lookback, d_model)

        # 添加位置编码
        x = self.pos_encoder(x)

        # Transformer 编码器
        for layer in self.transformer_encoder:
            x = layer(x)

        # 取最后一个时间步的输出
        last_out = x[:, -1, :]  # (batch, d_model)

        # 预测未来 horizon 个时间步
        output = self.output_layer(last_out)  # (batch, features * horizon)

        # 重塑为 (batch, horizon, features)
        output = output.view(batch_size, self.horizon, self.input_size)

        return output

    def predict(self, x):
        """预测函数（自动处理设备）"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
