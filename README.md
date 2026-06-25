# Long-Term Time Series Forecasting: A Comparative Study

基于 LSTM、Transformer、Informer、Autoformer 与 PatchTST 的长时序预测模型比较研究。

## 研究概述

本项目在统一实验框架下比较五类模型在长时序预测任务中的表现，覆盖 2 个数据集、5 个预测步长、5 个模型，并结合调参对比、消融实验、单变量/多变量对比和推理效率 benchmark 进行系统分析。

**核心结论**：PatchTST 在 10 个数据集—预测步长任务中获得 9 次最低 MSE，整体表现最优；Autoformer 在部分任务中仍有竞争力。消融实验验证了序列分解和通道独立建模的关键作用。

## 数据集

| 数据集 | 变量数 | 频率 | 切分方式 |
| --- | ---: | --- | --- |
| ETTh1 | 7 | 小时 | 前 12 月 / 4 月 / 4 月 |
| ETTm1 | 7 | 15 分钟 | 前 12 月 / 4 月 / 4 月 |

- 回看窗口：96
- 预测步长：24、48、96、168、336

## 模型

| 模型 | 定位 | 关键思想 |
| --- | --- | --- |
| LSTM | RNN 基线 | 门控循环结构 |
| Transformer | 标准注意力基线 | 全局自注意力 |
| Informer | 稀疏注意力变体 | ProbSparse attention |
| Autoformer | 分解式模型 | Series decomposition + Auto-Correlation |
| PatchTST | Patch 化模型 | Patching + channel independence |

## 主要结果

### v2 主实验（50 组：2 数据集 × 5 步长 × 5 模型）

| 模型 | 平均 MSE | 平均 MAE | 平均 R² | 参数量 | 推理延迟 (bs=1) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **PatchTST** | **0.4626** | **0.4497** | **0.6366** | 113K | 1.37 ms |
| Autoformer | 0.5016 | 0.4779 | 0.6061 | 111K | 7.21 ms |
| Transformer | 0.8217 | 0.6740 | 0.3547 | 338K | 1.48 ms |
| Informer | 0.8901 | 0.6860 | 0.3010 | 193K | 2.66 ms |
| LSTM | 0.9871 | 0.6998 | 0.2249 | 579K | 2.27 ms |

### 消融实验（24 组）

| 消融项 | ΔMSE/% |
| --- | ---: |
| Autoformer 去序列分解 | +66.2% |
| PatchTST 去通道独立 | +139.2% |
| PatchTST 去 Patching | +7.7% |
| Autoformer 去 Auto-Correlation | -3.1% |

### 单变量 vs 多变量（100 组，50 对配对任务）

- 单变量胜出：43 次
- 多变量胜出：7 次（全部来自 PatchTST）

## 项目结构

```text
Research-Training/
├── archive/              # 实验结果归档
│   ├── v1_results/       # 未调参基线（50 组）
│   ├── v2_results/       # 调参后主实验（50 组）
│   ├── v3_results/       # 单/多变量阶段性验证（40 组）
│   ├── v4_results/       # 消融 + 完整单/多变量对比（124 组）
│   └── docs/             # 历史开发记录
├── configs/              # 实验配置文件
├── data/                 # 原始数据集（本地，不上传）
├── docs/
│   ├── report/           # 论文（Markdown + LaTeX + PDF）
│   ├── analysis_results2.md
│   ├── best_model_params.md
│   └── knowledge/        # 参考资料
├── models/               # 模型实现
│   ├── lstm.py
│   ├── transformer.py
│   ├── informer.py
│   ├── autoformer.py
│   ├── patchtst.py
│   ├── trainer.py
│   └── ablation.py
├── notebooks/            # Jupyter 笔记本
│   ├── baseline/         # 基线训练
│   ├── tuning/           # 超参搜索
│   └── visualization/    # 可视化
├── scripts/              # 训练、调参、评估脚本
└── 参考资料/              # 论文 PDF 与模板
```

## 环境配置

```bash
pip install -r requirements.txt
```

依赖：`torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `tensorboard`, `jupyter`

## 论文

- Markdown 版：`docs/report/experiment_paper.md`
- LaTeX 源码：`docs/report/overleaf/`
- PDF：`docs/report/overleaf_project.pdf`

## 结果文件索引

| 内容 | 路径 |
| --- | --- |
| v2 主实验汇总 | `archive/v2_results/.../summaries/csv/full_val_best_e50p10_tb_seed42_summary.csv` |
| v1/v2 调参对比 | `archive/v1_results/.../summaries/csv/formal_seed42_all.csv` |
| v4 消融汇总 | `archive/v4_results/experiments/ablation_study/summaries/csv/` |
| v4 单/多变量对比 | `archive/v4_results/experiments/univariate_multivariate_comparison/summaries/csv/` |
| 推理时间 benchmark | `archive/v2_results/.../summaries/csv/pure_forward_inference_benchmark_by_model.csv` |

## 许可

[MIT License](LICENSE)
