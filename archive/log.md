# Archive 日志

> 本文件记录 `archive/` 目录下归档内容的来源、结构和注意事项。

---

## v1_results — 第一版正式实验结果归档

**归档时间**：2026-06-14
**来源分支/提交**：main（与 `docs/progress.md` 2026-06-14 之前记录对应）

### 归档范围

| 类别 | 内容 | 数量 |
|------|------|------|
| 正式核心实验 | ETTh1 / ETTm1 × 5 步长 × 5 模型（formal_seed42） | 50 组 |
| 消融实验 | ETTh1 / ETTm1 × 2 步长 × 4 消融变体（ablation_seed42） | 16 组 |
| 可视化图表 | 预测曲线、残差图、趋势图、消融差异图等 | 12 张 + manifest |
| CSV/MD 汇总 | 正式实验、消融对比、MAPE、单变量对比、ECL 快速验证等 | 约 40 个 |

### 模型列表

**正式实验（5 个）**：LSTM、Transformer、Informer、Autoformer、PatchTST

**消融变体（4 个）**：
- `autoformer_no_decomp` — 关闭序列分解，趋势分支置零
- `autoformer_no_autocorr` — Auto-Correlation 替换为标准多头自注意力
- `patchtst_no_patch` — 逐时间点线性投影替代 Patch Embedding
- `patchtst_channel_mix` — 混合多变量输入替代 Channel Independence

### 数据集与步长

| 数据集 | 变量数 | 频率 | 正式实验步长 | 消融实验步长 |
|--------|--------|------|--------------|--------------|
| ETTh1 | 7 | 小时 | 24, 48, 96, 168, 336 | 96, 336 |
| ETTm1 | 7 | 15 分钟 | 24, 48, 96, 168, 336 | 96, 336 |

### 实验类型

- **常规实验（formal_seed42）**：seed=42，全量数据，早停训练，结果目录 `h{horizon}/{dataset}/{model}/formal_seed42/`
- **消融实验（ablation_seed42）**：seed=42，全量数据，早停训练，结果目录 `h{horizon}/{dataset}/{ablation_model}/ablation_seed42/`

### seed

- 正式实验与消融实验均使用 **seed=42**

### 重要说明

1. **本结果为未调参版本**：LSTM、Transformer 使用默认超参（非搜索后的 Top1/Top2 配置），Informer、Autoformer、PatchTST 同样为默认配置，未做针对每个数据集的超参优化
2. **LSTM baseline 配置**：hidden_size=64, num_layers=2, lr=0.001（后续超参搜索发现 h256_l1_dp0.2_lr1e-3 更优，但未纳入本版结果）
3. **Transformer baseline 配置**：d_model=64, nhead=4, layers=2, ff=256, lr=0.001（后续搜索发现 d128_h8_lr5e-5 更优，但未纳入本版结果）
4. 归档文件编码为 UTF-8；原始 `log.md` 为 GBK 编码，本文件为 UTF-8 重写版

---

## 目录结构

```
archive/
├── log.md                          ← 本文件
└── v1_results/
    ├── figures/                    ← 可视化图表（12 张 PNG + manifest.json）
    │   ├── formal_metric_trends.png        — 五模型 MSE/MAE/R² 随步长变化趋势
    │   ├── formal_best_model_by_horizon.png — 各步长最优模型柱状图
    │   ├── formal_complexity_tradeoff.png   — 参数量 vs MSE 权衡图
    │   ├── ablation_delta_mse_pct.png       — 消融 MSE 百分比变化图
    │   ├── prediction_ETTh1_h96_patchtst.png    — ETTh1 h96 PatchTST 预测 vs 真实
    │   ├── prediction_ETTh1_h336_patchtst.png   — ETTh1 h336 PatchTST 预测 vs 真实
    │   ├── prediction_ETTm1_h96_autoformer.png  — ETTm1 h96 Autoformer 预测 vs 真实
    │   ├── prediction_ETTm1_h336_autoformer.png — ETTm1 h336 Autoformer 预测 vs 真实
    │   ├── residual_ETTh1_h96_patchtst.png       — ETTh1 h96 PatchTST 残差分析
    │   ├── residual_ETTh1_h336_patchtst.png      — ETTh1 h336 PatchTST 残差分析
    │   ├── residual_ETTm1_h96_autoformer.png     — ETTm1 h96 Autoformer 残差分析
    │   └── residual_ETTm1_h336_autoformer.png    — ETTm1 h336 Autoformer 残差分析
    ├── h{24,48,96,168,336}/         ← 按步长分类的结果详情
    │   ├── ETTh1/                   ← 每个数据集
    │   │   ├── lstm/formal_seed42/       — 含 _results.npy + _summary.json
    │   │   ├── transformer/formal_seed42/
    │   │   ├── informer/formal_seed42/
    │   │   ├── autoformer/formal_seed42/
    │   │   ├── patchtst/formal_seed42/
    │   │   ├── autoformer_no_decomp/ablation_seed42/    — 消融变体（仅 h96, h336）
    │   │   ├── autoformer_no_autocorr/ablation_seed42/
    │   │   ├── patchtst_no_patch/ablation_seed42/
    │   │   └── patchtst_channel_mix/ablation_seed42/
    │   └── ETTm1/                  ← 结构同 ETTh1
    ├── v1_csv/                      ← CSV 格式汇总表
    │   ├── formal/                  ← 正式实验汇总（每步长、每数据集、MAPE 等）
    │   ├── ablation/                ← 消融实验汇总与对比表
    │   ├── figures/                 ← 图表相关 CSV（预测样本摘要等）
    │   └── catalog.md               ← CSV 文件索引
    └── v1_md/                       ← Markdown 格式汇总表
        ├── formal/                  ← 正式实验 Markdown 汇总
        ├── ablation/                ← 消融实验 Markdown 汇总
        └── catalog.md               ← Markdown 文件索引
```

---

## 核心实验结论摘要

### 正式实验（formal_seed42）

**ETTh1**（5 个步长均 PatchTST 最优）：

| Horizon | 最低 MSE 模型 | MSE | R² |
|--------:|---------------|-----|-----|
| 24 | PatchTST | 0.380 | 0.703 |
| 48 | PatchTST | 0.420 | 0.671 |
| 96 | PatchTST | 0.483 | 0.621 |
| 168 | PatchTST | 0.514 | 0.597 |
| 336 | PatchTST | 0.594 | 0.533 |

**ETTm1**（5 个步长均 Autoformer 最优）：

| Horizon | 最低 MSE 模型 | MSE | R² |
|--------:|---------------|-----|-----|
| 24 | Autoformer | 0.307 | 0.759 |
| 48 | Autoformer | 0.447 | 0.649 |
| 96 | Autoformer | 0.461 | 0.638 |
| 168 | Autoformer | 0.509 | 0.599 |
| 336 | Autoformer | 0.555 | 0.563 |

### 消融实验（ablation_seed42）

| 消融方向 | 关键发现 |
|----------|---------|
| Autoformer 序列分解 | 移除后 MSE 上升约 37%~119%，贡献最显著 |
| Autoformer Auto-Correlation | ETTh1 上略低于轻量版，提示实现/超参仍有优化空间 |
| PatchTST Channel Independence | 混合通道后 MSE 上升约 85%~291%，Channel Independence 贡献最显著 |
| PatchTST Patching | h96 上移除后 MSE 上升约 8%，h336 上移除后仅上升 0.5% |

---

## 与主仓库结果的差异

归档后，主仓库 `results/` 目录可能继续演进（如增加调参后 LSTM/Transformer 结果、ECL 正式实验等），而 `archive/v1_results/` 保留归档时刻的快照，不再更新。