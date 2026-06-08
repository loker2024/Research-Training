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

推荐按平台使用现有环境：

```bash
# Mac
conda activate miniMac

# Windows
myenv\Scripts\activate
```

`myenv/` 已被 `.gitignore` 忽略，用于 Windows 本地环境；Mac 侧使用 Conda 的 `miniMac`。

## 运行顺序

1. 运行 `notebooks/data_preparation.ipynb` 或 `python scripts/preprocess_data.py --datasets ETTh1 --horizons 96` 完成数据下载、归一化和滑动窗口预处理。
2. 运行 `notebooks/train_baseline.ipynb` 训练 LSTM 与 Transformer 基线。
3. 运行 `notebooks/train_variants.ipynb` 或 `python scripts/run_experiments.py --datasets ETTh1 --horizons 96 --models autoformer` 训练 Informer、Autoformer 与 PatchTST。
4. 运行 `python scripts/summarize_results.py --datasets ETTh1 --horizons 24,48,96,168,336 --output-prefix ETTh1_quick_summary` 汇总已保存结果。
5. 运行 `python scripts/organize_results.py --overwrite` 按步长和模型整理 `results/`。
6. 每完成独立步骤后更新 `docs/progress.md`。

常用命令示例：

```bash
# 使用配置文件运行，适合正式实验和复现实验
python scripts/run_experiments.py --config configs/core_experiment_smoke.json

# ETTh1/ETTm1 正式核心实验配置（运行时间较长）
python scripts/run_experiments.py --config configs/core_experiment_etth1_ettm1_formal.json

# 中断后续跑：配置文件已默认开启 skip_existing，也可命令行显式开启
python scripts/run_experiments.py \
  --config configs/core_experiment_etth1_ettm1_formal.json \
  --skip-existing

# 将结果分类为按步长/按模型两个视图
python scripts/organize_results.py --overwrite

# 优化变体重训结果使用 run tag，避免覆盖旧结果
python scripts/run_experiments.py \
  --datasets ETTh1,ETTm1 \
  --horizons 24,48,96,168,336 \
  --models informer,patchtst \
  --run-tag optv2 \
  --seed 42

# ECL 高维数据先使用单独 smoke 目录验证流程
python scripts/preprocess_data.py \
  --datasets ECL \
  --horizons 96 \
  --output-dir data/processed_smoke \
  --max-samples-per-split 256

python scripts/run_experiments.py \
  --data-dir data/processed_smoke \
  --datasets ECL \
  --horizons 96 \
  --models informer,patchtst \
  --run-tag ecl_smoke_optv2 \
  --sample-limit 64 \
  --batch-size 8 \
  --seed 42
```

结果目录分类：

```text
results/
├── by_horizon/h24/{dataset}/{model}/{run_tag}/
├── by_model/{model}/{dataset}/h24/{run_tag}/
├── summaries/
└── RESULTS_INDEX.md
```

## 当前进度

进度记录见 `docs/progress.md`，实施步骤见 `docs/项目步骤.md`。
