# 步骤 6：可视化与深入分析

## 图表索引

本步骤生成的图表统一保存在 `results/figures/`：

| 图表 | 路径 | 用途 |
| --- | --- | --- |
| 核心指标趋势 | `results/figures/formal_metric_trends.png` | 对比 5 个模型在 ETTh1/ETTm1 五个预测步长上的 MSE、MAE、R² |
| 各步长最优模型 | `results/figures/formal_best_model_by_horizon.png` | 展示每个数据集、每个 horizon 的最低 MSE 模型 |
| 复杂度对比 | `results/figures/formal_complexity_tradeoff.png` | 对比平均 MSE、参数量和训练耗时 |
| 消融影响 | `results/figures/ablation_delta_mse_pct.png` | 展示各消融变体相对原模型的 MSE 变化百分比 |
| ETTh1 h96 PatchTST 预测曲线 | `results/figures/prediction_ETTh1_h96_patchtst.png` | 目标列反归一化后的预测值与真实值 |
| ETTh1 h96 PatchTST 残差 | `results/figures/residual_ETTh1_h96_patchtst.png` | 预测误差随步长变化 |
| ETTh1 h336 PatchTST 预测曲线 | `results/figures/prediction_ETTh1_h336_patchtst.png` | 长步长目标列预测效果 |
| ETTh1 h336 PatchTST 残差 | `results/figures/residual_ETTh1_h336_patchtst.png` | 长步长残差分析 |
| ETTm1 h96 Autoformer 预测曲线 | `results/figures/prediction_ETTm1_h96_autoformer.png` | 目标列反归一化后的预测值与真实值 |
| ETTm1 h96 Autoformer 残差 | `results/figures/residual_ETTm1_h96_autoformer.png` | 预测误差随步长变化 |
| ETTm1 h336 Autoformer 预测曲线 | `results/figures/prediction_ETTm1_h336_autoformer.png` | 长步长目标列预测效果 |
| ETTm1 h336 Autoformer 残差 | `results/figures/residual_ETTm1_h336_autoformer.png` | 长步长残差分析 |

推理样本 shape 已写入 `results/figures/prediction_samples_summary.csv`，图表清单写入 `results/figures/manifest.json`。

## 核心实验结论

1. ETTh1 上 PatchTST 在五个预测步长中均取得最低 MSE，h24/h48/h96/h168/h336 的 MSE 分别为 0.380213、0.420251、0.483175、0.513911、0.594367。随着预测步长变长，MSE 逐步上升、R² 从 0.702517 下降到 0.532757，体现出长期预测误差累积。
2. ETTm1 上 Autoformer 在五个预测步长中均取得最低 MSE，h24/h48/h96/h168/h336 的 MSE 分别为 0.307059、0.447473、0.460524、0.508573、0.554604。Autoformer 与 PatchTST 在 ETTm1 上非常接近，但 Autoformer 整体略优。
3. 从 50 组正式结果的平均 MSE 看，PatchTST 最低，为 0.469335；Autoformer 次之，为 0.514711。两者明显优于 Informer、Transformer 和 LSTM。
4. Transformer、LSTM、Informer 在长步长上的退化更明显。以平均 MSE 计，Informer 为 0.817005，Transformer 为 0.917335，LSTM 为 0.946142；在 ETTh1 h168/h336 等长步长上，部分模型 R² 接近 0 或明显低于强模型。
5. 数据集差异明显：ETTh1 更适合 PatchTST，ETTm1 更适合 Autoformer。ETTm1 训练样本更多、频率更高，Autoformer 的序列分解对该数据集的周期/趋势结构更有帮助。

## 消融实验结论

1. PatchTST 的 Channel Independence 贡献最强。`patchtst_channel_mix` 平均使 MSE 上升 162.25%，其中 ETTh1 h96 上升 290.87%，说明混合通道会显著破坏 PatchTST 在多变量时序中的建模优势。
2. Autoformer 的序列分解模块非常关键。`autoformer_no_decomp` 平均使 MSE 上升 81.48%，在 ETTh1 h336 上升 118.90%，在 ETTm1 h336 上升 102.10%，证明趋势/季节分解对长步长预测很重要。
3. PatchTST 的 patching 机制整体有效但影响小于通道独立性。`patchtst_no_patch` 平均使 MSE 上升 8.54%；其中 ETTm1 h96 上升 19.33%，但 ETTh1 h336 仅上升 0.47%，提示 patch 长度和长步长配置仍有调参空间。
4. Auto-Correlation 的结论更复杂。`autoformer_no_autocorr` 在 ETTh1 上 MSE 反而下降约 9%，但在 ETTm1 上上升约 2% 到 4%。这说明当前轻量 Autoformer 实现或超参数可能没有完全发挥 Auto-Correlation 优势，后续可作为优化方向。

## 预测曲线与残差观察

1. 预测曲线使用已有 `formal_seed42` checkpoint 直接推理测试集首批样本，未重训模型；目标列已通过 `scaler.npz` 反归一化到原始量纲。
2. h96 图适合展示模型对局部波动的拟合能力；h336 图更适合展示长期预测中均值回归、相位偏移和误差累积现象。
3. 残差图可用于报告中的误差分析：短步长残差更集中，长步长残差更容易出现持续偏差，说明长预测窗口下模型会逐渐丢失局部细节。

## 后续报告建议

1. 报告主表使用 `results/formal_seed42_all.csv` 和 `results/ablation_seed42_vs_formal_comparison.csv`。
2. 正文图优先放 `formal_metric_trends.png`、`formal_best_model_by_horizon.png`、`ablation_delta_mse_pct.png` 和两组代表性预测/残差图。
3. 结论部分突出两条主线：PatchTST 在 ETTh1 上稳定领先，Autoformer 在 ETTm1 上整体最优；消融结果证明通道独立性和序列分解是最关键的结构贡献。
