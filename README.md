# Research Training: Long-Term Time Series Forecasting

基于 LSTM、Transformer、Informer、Autoformer 与 PatchTST 的长时序预测研究项目。

## 项目范围

当前实验聚焦 3 个数据集：

| 数据集 | 变量数 | 频率 | 切分方式 |
| --- | ---: | --- | --- |
| ETTh1 | 7 | 小时 | 前 12 月 / 4 月 / 4 月 |
| ETTm1 | 7 | 15 分钟 | 前 12 月 / 4 月 / 4 月 |
| ECL | 321 | 小时 | 70% / 10% / 20% |

回看窗口为 96。当前预处理 notebook 生成了 24、48、96、168、336 五组预测步长，核心实验可按项目要求优先报告 96、168、336。

## 目录结构

```text
Research-Training/
├── data/           # 原始数据与预处理数据，本地保存，不上传 GitHub
├── docs/           # 项目文档、进度记录、报告资料
├── models/         # LSTM、Transformer、Informer、Autoformer、PatchTST 与训练框架
├── notebooks/      # 数据准备、基线训练、变体训练 notebook
├── papers/         # 相关论文 PDF
├── results/        # 实验结果，本地保存，不上传 GitHub
└── checkpoints/    # 模型权重，本地保存，不上传 GitHub
```

## 环境配置

```bash
pip install -r requirements.txt
```

如果使用本地虚拟环境，建议命名为 `myenv`，该目录已被 `.gitignore` 忽略。

## 运行顺序

1. 运行 `notebooks/data_preparation.ipynb` 完成数据下载、归一化和滑动窗口预处理。
2. 运行 `notebooks/train_baseline.ipynb` 训练 LSTM 与 Transformer 基线。
3. 运行 `notebooks/train_variants.ipynb` 训练 Informer、Autoformer 与 PatchTST。
4. 每完成独立步骤后更新 `docs/progress.md`。

## 当前进度

进度记录见 `docs/progress.md`，实施步骤见 `docs/项目步骤.md`。
