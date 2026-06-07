"""LSTM 基线模型"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM 时序预测模型

    输入: (batch, lookback, features)
    输出: (batch, horizon, features)
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 dropout=0.1, horizon=96):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # 输出层：将 LSTM 输出映射到预测步长
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size * horizon)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
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

        # LSTM 前向传播
        lstm_out, _ = self.lstm(x)  # (batch, lookback, hidden_size)

        # 取最后一个时间步的输出
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)

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
