# v1_md 目录配置清单

> 整理时间：2026-06-10
> 实验版本：formal_seed42 / ablation_seed42
> 预测步长：24, 48, 96, 168, 336
> 模型：LSTM, Transformer, Informer, Autoformer, PatchTST（正式实验）；消融变体（消融实验）

---

## formal/ — 正式核心实验结果（Markdown 表格）

| 文件名 | 数据集 | 步长 | 说明 |
|--------|--------|------|------|
| `formal_seed42_etth1_h24_summary.md` | ETTh1 | 24 | ETTh1 h24 五模型正式结果表 |
| `formal_seed42_etth1_h48_summary.md` | ETTh1 | 48 | ETTh1 h48 五模型正式结果表 |
| `formal_seed42_etth1_h96.md` | ETTh1 | 96 | ETTh1 h96 五模型正式结果表 |
| `formal_seed42_etth1_h168.md` | ETTh1 | 168 | ETTh1 h168 五模型正式结果表 |
| `formal_seed42_etth1_h336.md` | ETTh1 | 336 | ETTh1 h336 五模型正式结果表 |
| `formal_seed42_etth1.md` | ETTh1 | 全部 | ETTh1 多步长合并汇总（与 _all 有部分重叠） |
| `formal_seed42_etth1_all.md` | ETTh1 | 全部 | ETTh1 全五步长合并汇总 |
| `formal_seed42_etth1_auto_patchtst.md` | ETTh1 | 96, 336 | ETTh1 仅 Autoformer 与 PatchTST 两模型对比子集 |
| `formal_seed42_ettm1_h24.md` | ETTm1 | 24 | ETTm1 h24 五模型正式结果表 |
| `formal_seed42_ettm1_h48.md` | ETTm1 | 48 | ETTm1 h48 五模型正式结果表 |
| `formal_seed42_ettm1_h96.md` | ETTm1 | 96 | ETTm1 h96 五模型正式结果表 |
| `formal_seed42_ettm1_h168.md` | ETTm1 | 168 | ETTm1 h168 五模型正式结果表 |
| `formal_seed42_ettm1_h336.md` | ETTm1 | 336 | ETTm1 h336 五模型正式结果表 |
| `formal_seed42_ettm1_all.md` | ETTm1 | 全部 | ETTm1 全五步长合并汇总 |
| `formal_seed42_all.md` | ETTh1+ETTm1 | 全部 | 跨数据集总汇总 Markdown 表 |

**列说明**（所有正式 MD 文件格式统一）：dataset, horizon, model, run_tag, MSE, MAE, R2, MSE_target, MAE_target, R2_target, best_val_loss, best_val_r2, trained_epochs, seed, sample_limit, train/val/test_samples, model_params, train_time_seconds, device, data_dir

---

## ablation/ — 消融实验结果（Markdown 表格）

| 文件名 | 数据集 | 步长 | 说明 |
|--------|--------|------|------|
| `ablation_seed42_etth1_partial.md` | ETTh1 | 96, 336 | ETTh1 h96/h336 消融原始指标表（4 个消融变体） |
| `ablation_seed42_summary.md` | ETTh1+ETTm1 | 96, 336 | 完整 16 组消融实验对比表（含 delta MSE/MAE/R² 与百分比变化） |
| `ablation_seed42_etth1_comparison.md` | ETTh1 | 96 | ETTh1 h96 消融 vs 原模型中文对比表（含 MSE 变化百分比） |
| `ablation_seed42_vs_formal_comparison.md` | ETTh1+ETTm1 | 96, 336 | 消融变体 vs 原模型完整中文对比表（含 MSE/MAE/R² 差值与百分比变化） |

**消融变体说明**：
- `autoformer_no_decomp` — 关闭 Autoformer 序列分解，趋势分支置零
- `autoformer_no_autocorr` — 将 Auto-Correlation 替换为标准多头自注意力
- `patchtst_no_patch` — 逐时间点线性投影替代 Patch Embedding
- `patchtst_channel_mix` — 混合多变量输入替代 Channel Independence