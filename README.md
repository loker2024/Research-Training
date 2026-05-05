# Research Training: Long-Term Time Series Forecasting

本仓库围绕“基于 LSTM、Transformer 及高效变体的长时序预测”课题展开，第一版实现聚焦于可复现实验骨架：

- 按时间顺序划分训练、验证、测试集，避免未来信息泄露。
- 使用滑动窗口构造多变量或单变量预测样本。
- 提供 LSTM、Transformer、PatchTST-lite 三类 PyTorch 模型。
- 支持 MSE、MAE、MAPE 指标、预测曲线与残差图。
- 预留时序分解消融开关，服务后续 Autoformer/Informer 扩展。

## 文件结构

```text
.
├── notebooks/
│   └── long_term_forecasting_v1.ipynb
├── docs/
│   └── v1_implementation.md
├── data/
│   └── README.md
├── requirements.txt
├── 选题-时间序列预测.md
└── 基于 LSTM、Transformer、Informer、Autoformer 与 PatchTST 的长时序预测实验报告.pdf
```

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/long_term_forecasting_v1.ipynb
```

notebook 默认使用合成数据跑通流程。下载 ECL、ETT、ILI 或 Traffic 数据后，将 CSV 放入 `data/`，并在 notebook 的配置单元中修改 `DATA_PATH`、`TARGET_COL` 和 `TIME_COL`。

