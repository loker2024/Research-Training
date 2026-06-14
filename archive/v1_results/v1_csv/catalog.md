# v1_csv 目录配置清单

> 整理时间：2026-06-10
> 实验版本：formal_seed42 / ablation_seed42
> 预测步长：24, 48, 96, 168, 336
> 模型：LSTM, Transformer, Informer, Autoformer, PatchTST（正式实验）；消融变体（消融实验）

---

## formal/ — 正式核心实验结果

| 文件名 | 数据集 | 步长 | 说明 |
|--------|--------|------|------|
| `formal_seed42_etth1_h24_summary.csv` | ETTh1 | 24 | ETTh1 h24 五模型正式结果 |
| `formal_seed42_etth1_h48_summary.csv` | ETTh1 | 48 | ETTh1 h48 五模型正式结果 |
| `formal_seed42_etth1_h96.csv` | ETTh1 | 96 | ETTh1 h96 五模型正式结果 |
| `formal_seed42_etth1_h168.csv` | ETTh1 | 168 | ETTh1 h168 五模型正式结果 |
| `formal_seed42_etth1_h336.csv` | ETTh1 | 336 | ETTh1 h336 五模型正式结果 |
| `formal_seed42_etth1.csv` | ETTh1 | 全部 | ETTh1 多步长合并汇总（含 h24/h48，与 _all 有部分重叠） |
| `formal_seed42_etth1_all.csv` | ETTh1 | 全部 | ETTh1 全五步长合并汇总 |
| `formal_seed42_etth1_auto_patchtst.csv` | ETTh1 | 96, 336 | ETTh1 仅 Autoformer 与 PatchTST 两模型对比子集 |
| `formal_seed42_ettm1_h24.csv` | ETTm1 | 24 | ETTm1 h24 五模型正式结果 |
| `formal_seed42_ettm1_h48.csv` | ETTm1 | 48 | ETTm1 h48 五模型正式结果 |
| `formal_seed42_ettm1_h96.csv` | ETTm1 | 96 | ETTm1 h96 五模型正式结果 |
| `formal_seed42_ettm1_h168.csv` | ETTm1 | 168 | ETTm1 h168 五模型正式结果 |
| `formal_seed42_ettm1_h336.csv` | ETTm1 | 336 | ETTm1 h336 五模型正式结果 |
| `formal_seed42_ettm1_all.csv` | ETTm1 | 全部 | ETTm1 全五步长合并汇总 |
| `formal_seed42_all.csv` | ETTh1+ETTm1 | 全部 | 跨数据集总汇总（50 组实验） |
| `formal_seed42_mape.csv` | ETTh1+ETTm1 | 全部 | 从 50 组正式 summary JSON 提取的 MAPE/MAPE_target 明细 |
| `formal_seed42_mape_by_model.csv` | ETTh1+ETTm1 | 全部 | 按数据集与模型聚合的 MAPE/MAPE_target 均值、最小值和最大值 |

**列说明**：dataset, horizon, model, run_tag, MSE, MAE, R2, MSE_target, MAE_target, R2_target, best_val_loss, best_val_r2, trained_epochs, seed, sample_limit, train/val/test_samples, model_params, train_time_seconds, device, data_dir。MAPE 补充表另含 MAPE、MAPE_target 及按模型聚合后的 avg/min/max 字段。

---

## ablation/ — 消融实验结果

| 文件名 | 数据集 | 步长 | 说明 |
|--------|--------|------|------|
| `ablation_seed42_etth1_partial.csv` | ETTh1 | 96, 336 | ETTh1 h96/h336 消融原始指标（4 个消融变体） |
| `ablation_seed42_summary.csv` | ETTh1+ETTm1 | 96, 336 | 完整 16 组消融实验原始指标汇总 |
| `ablation_seed42_vs_formal_comparison.csv` | ETTh1+ETTm1 | 96, 336 | 消融变体 vs 原模型的逐组对比表（含 MSE/MAE/R² 差值与百分比变化） |

**ablation_partial / ablation_summary 列说明**：同正式实验列（dataset, horizon, model, run_tag, MSE, MAE, R2, ...），model 列为消融变体名称（如 autoformer_no_decomp）

**ablation_vs_formal_comparison 列说明**：dataset, horizon, base_model, ablation_model, ablation, base_MSE, ablation_MSE, delta_MSE, delta_MSE_pct, base_MAE, ablation_MAE, delta_MAE, base_R2, ablation_R2, delta_R2, base_params, ablation_params, base_train_time_seconds, ablation_train_time_seconds, base_trained_epochs, ablation_trained_epochs

**消融变体说明**：
- `autoformer_no_decomp` — 关闭 Autoformer 序列分解，趋势分支置零
- `autoformer_no_autocorr` — 将 Auto-Correlation 替换为标准多头自注意力
- `patchtst_no_patch` — 逐时间点线性投影替代 Patch Embedding
- `patchtst_channel_mix` — 混合多变量输入替代 Channel Independence

---

## figures/ — 可视化索引

| 文件名 | 说明 |
|--------|------|
| `prediction_samples_summary.csv` | 生成的预测/残差图索引：dataset, horizon, model, target_idx, prediction_shape, target_shape, prediction_figure, residual_figure |

---

## feature_mode/ — 单变量 vs 多变量对比实验

| 文件名 | 数据集 | 步长 | 说明 |
|--------|--------|------|------|
| `feature_mode_seed42_comparison.csv` | ETTh1+ETTm1 | 96, 336 | 代表性小样本单变量/多变量明细表（4 模型 × 2 数据集 × 2 步长 × 2 模式） |
| `feature_mode_seed42_comparison_delta.csv` | ETTh1+ETTm1 | 96, 336 | 单变量减多变量的目标列 MSE/MAE/R² 差值表 |
| `feature_mode_smoke_comparison.csv` | ETTh1 | 96 | LSTM 最小 smoke 明细表 |
| `feature_mode_smoke_comparison_delta.csv` | ETTh1 | 96 | LSTM 最小 smoke 差值表 |

**实验口径**：`univariate` 只输入目标列并预测目标列；`multivariate` 保留全部变量输入和输出，并以目标列指标进行对比。`feature_mode_seed42` 使用 `sample_limit=512` 和 `epochs=5`，适合快速复现实验流程；全量正式对比可将配置中的 `sample_limit` 改为 `0` 后重跑。

---

## ecl/ — ECL 高维快速实验

| 文件名 | 数据集 | 步长 | 说明 |
|--------|--------|------|------|
| `ecl_smoke_optv2_summary.csv` | ECL | 96 | 321 变量 ECL 高维快速验证汇总表，覆盖 Informer 与 PatchTST |

**实验口径**：使用 `data/processed_smoke/ECL`，保留 321 个变量，但每个 split 最多 256 个窗口；训练时 `sample_limit=64`、`epochs=1`、`batch_size=8`。该结果用于验证高维数据流程可跑通，并作为附录快速实验，不等同于 ECL 正式全量实验。
