# AGENTS.md

本文件记录本项目的全局记忆、协作约定和技术偏好，供后续 AI 编程助手或维护者快速理解项目上下文。

## 1. 项目背景

项目名称：Research-Training

课题方向：长时序预测研究。

核心研究内容：

- 基于 LSTM、Transformer 及其高效变体开展时间序列预测实验。
- 重点关注 Informer、Autoformer、PatchTST 等长序列预测模型。
- 探索多尺度时序分解、外生变量融合、概率预测、长期依赖建模和计算效率问题。
- 公开数据集优先考虑 ETT、ECL、ILI、Traffic。

当前第一版代码：

- notebook：`notebooks/long_term_forecasting_v1.ipynb`
- 已实现：LSTM、Transformer、PatchTST-lite。
- 已包含：滑动窗口、时间顺序切分、训练集标准化、MSE/MAE/MAPE、预测曲线、残差图、分解消融开关。
- 尚未完整实现：Informer 的 ProbSparse Attention、Autoformer 的 Auto-Correlation block、完整 PatchTST 论文实现。

## 2. 目录约定

```text
.
├── AGENTS.md
├── README.md
├── data/
│   ├── README.md
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   ├── ETTm2.csv
│   ├── ECL.csv
│   ├── ILI.csv
│   └── Traffic.csv
├── docs/
│   ├── datasets.md
│   └── v1_implementation.md
├── notebooks/
│   └── long_term_forecasting_v1.ipynb
├── requirements.txt
├── 选题-时间序列预测.md
└── 基于 LSTM、Transformer、Informer、Autoformer 与 PatchTST 的长时序预测实验报告.pdf
```

目录使用约定：

- `notebooks/`：存放实验 notebook。用户明确要求 notebook 文件，因此主要实验代码优先写在这里。
- `docs/`：存放 Markdown 文档。所有项目说明文档默认使用 Markdown。
- `data/`：存放本地数据集和数据说明。数据文件不上传 GitHub。
- `outputs/`、`checkpoints/`、`runs/`：存放训练输出、模型权重、日志和图表，默认不上传 GitHub。

## 3. Git 与数据约定

必须遵守：

- 不要把 `data/*.csv`、`data/*.zip`、训练输出、checkpoint 上传到 GitHub。
- 当前 `.gitignore` 已忽略数据文件和输出目录。
- 可以提交 `data/README.md`，但不要提交真实数据。
- 提交前必须运行 `git status --short --ignored data` 或等价命令确认数据文件为 ignored。

当前本地已下载的数据集：

| 文件 | 说明 | 约大小 |
| --- | --- | ---: |
| `data/ETTh1.csv` | ETT 小时级数据 1 | 2.5 MB |
| `data/ETTh2.csv` | ETT 小时级数据 2 | 2.3 MB |
| `data/ETTm1.csv` | ETT 15 分钟级数据 1 | 9.9 MB |
| `data/ETTm2.csv` | ETT 15 分钟级数据 2 | 9.2 MB |
| `data/ECL.csv` | Electricity/ECL 用电量数据 | 91 MB |
| `data/ILI.csv` | national illness/ILI 数据 | 65 KB |
| `data/Traffic.csv` | Traffic 交通流量数据 | 130 MB |

远程仓库：

- `origin`: `https://github.com/loker2024/Research-Training.git`
- 主分支：`main`

## 4. 文档约定

所有新写的项目说明文档默认使用 Markdown 格式。

已有文档：

- `README.md`：项目总览和快速开始。
- `docs/v1_implementation.md`：第一版 PyTorch 实现说明。
- `docs/datasets.md`：ETT、ECL、ILI、Traffic 数据集介绍。
- `data/README.md`：数据目录说明。

写文档时的偏好：

- 面向初学者，先解释概念，再给操作步骤。
- 中文为主，可以保留必要英文术语，例如 `seq_len`、`pred_len`、`MSE`。
- 表格优先用于对比数据集、模型、实验设置。
- 不写空泛介绍，尽量落到本项目的文件路径、配置项和实验步骤。

## 5. 技术偏好

主要技术栈：

- Python
- PyTorch
- pandas
- NumPy
- Matplotlib
- Jupyter Notebook

实现偏好：

- 代码要有详细中文注释，尤其是数据切分、标准化、模型输入输出形状、训练循环和指标计算。
- 优先保持 notebook 可读、可教学、可逐步运行。
- 第一版不追求复杂工程化，后续再按需要拆分为 `src/` 模块。
- 数据切分必须按时间顺序，不允许随机划分原始时间轴。
- 标准化只能在训练集上拟合，再应用到验证集和测试集。
- 指标至少包含 MSE、MAE、MAPE。
- 可视化至少包含预测值与真实值对比曲线、残差图。
- 实验结果应记录模型名、数据集名、`seq_len`、`pred_len`、参数量、训练耗时和指标。

## 6. 模型路线

推荐迭代顺序：

1. 保持第一版 notebook 稳定可运行。
2. 使用 ETTh1 跑通真实数据训练。
3. 增加多模型对比循环，统一导出结果表。
4. 增加多预测步长实验，例如 24、48、96、168、336。
5. 增加 Informer 简化版或核心 ProbSparse Attention。
6. 增加 Autoformer 的序列分解和 Auto-Correlation block。
7. 将 notebook 中稳定代码抽取为 `src/`，保留 notebook 做实验入口。
8. 汇总图表和实验结果，更新研究报告。

## 7. 数据集使用建议

初学者优先顺序：

1. `ETTh1.csv`：数据适中，字段清晰，优先用于跑通流程。
2. `ETTm1.csv`：时间粒度更细，适合观察长输入窗口效果。
3. `ECL.csv`：多变量更多，适合验证 PatchTST 类模型。
4. `ILI.csv`：小样本、强季节性，适合写误差分析。
5. `Traffic.csv`：维度高、计算更重，适合作为进阶实验。

常见配置：

```python
cfg = ExperimentConfig(
    data_path="../data/ETTh1.csv",
    time_col="date",
    target_col="OT",
    use_synthetic_demo=False,
)
```

ECL 和 Traffic 的列很多，目标列可先使用 `OT`。如果机器性能有限，后续可以在数据读取阶段只选择部分变量。

## 8. 运行与验证

基础安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/long_term_forecasting_v1.ipynb
```

当前注意事项：

- 之前检查时系统 Python 未安装 `torch`，因此训练烟测需要先安装依赖。
- 如果下载依赖需要联网，可能需要用户授权网络访问。
- notebook JSON 和代码语法曾通过静态检查，但真实训练仍应在安装依赖后验证。

建议验证顺序：

1. 先用 notebook 默认合成数据运行 1 到 2 个 epoch。
2. 再切换到 `data/ETTh1.csv`，确认真实数据流程可跑通。
3. 最后扩大到 ECL 或 Traffic。

## 9. 协作注意事项

- 不要删除或覆盖用户已有研究报告和选题文件。
- 不要把大型数据和训练产物提交到 GitHub。
- 修改 notebook 后，至少检查 JSON 可解析。
- 修改文档后，保持 Markdown 链接有效。
- 提交前检查工作区，确认只提交本次任务相关文件。
- 推送 GitHub 前确认分支为 `main`，远程为 `origin`。

