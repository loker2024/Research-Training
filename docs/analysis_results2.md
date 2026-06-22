# results 2 完整矩阵结果分析

**分析时间**：2026-06-15 14:42 CST  
**数据来源**：`results 2/v2_csv/full_matrix/full_val_best_e50p10_tb_seed42_summary.csv`  
**运行标签**：`full_val_best_e50p10_tb_seed42`

## 1. 结果完整性

`results 2` 当前包含 50 个正式实验结果，已覆盖本项目当前采用的两个数据集：

- ETTh1：5 个预测步长 × 5 个模型 = 25 组
- ETTm1：5 个预测步长 × 5 个模型 = 25 组

运行状态文件 `results 2/run_state/full_val_best_e50p10_tb_seed42_state.json` 显示：

- `status = finished`
- `completed_before_start = 50`
- `pending_this_run = 0`
- `missing_data = 25`
- `error = None`

其中 `missing_data = 25` 来自 notebook 里仍保留的 ECL 计划项。按当前项目口径，ECL 已不再作为正式实验要求；因此 ETTh1 和 ETTm1 的 50 组结果已经构成本轮完整结果矩阵。

## 2. 总体结论

按 ETTh1 和 ETTm1 共 10 个任务的多变量 MSE 统计：

- PatchTST 在 9/10 个任务中取得最低 MSE，是 v2 的最稳定最优模型。
- Autoformer 在 ETTm1 h24 取得最低多变量 MSE，并且在多数任务中排名第二。
- Transformer、Informer、LSTM 与前两者存在明显差距。
- Autoformer 的验证集 loss 在 10/10 个任务中最低，但测试集 MSE 多数被 PatchTST 反超，说明 Autoformer 的验证集优势没有完全转化为测试集泛化优势。

按目标变量 MSE 统计：

- PatchTST 在 8/10 个任务中最优。
- Autoformer 在 ETTm1 h48 和 h336 的目标变量 MSE 最优。

## 3. 分数据集平均表现

| 数据集 | 模型 | Avg MSE | Avg MAE | Avg R2 | Target MSE | Target R2 | 总耗时(s) | 平均epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | patchtst | 0.4752 | 0.4684 | 0.6275 | 0.0821 | 0.4729 | 168.6 | 24.0 |
| ETTh1 | autoformer | 0.5462 | 0.5104 | 0.5718 | 0.1300 | 0.1660 | 264.1 | 18.4 |
| ETTh1 | transformer | 0.9428 | 0.7567 | 0.2610 | 0.5255 | -2.3769 | 349.6 | 33.6 |
| ETTh1 | informer | 1.0196 | 0.7721 | 0.2007 | 0.8066 | -4.1810 | 202.7 | 15.0 |
| ETTh1 | lstm | 1.1094 | 0.7730 | 0.1304 | 0.3693 | -1.3682 | 133.5 | 15.8 |
| ETTm1 | patchtst | 0.4500 | 0.4309 | 0.6458 | 0.0722 | 0.5380 | 417.9 | 14.6 |
| ETTm1 | autoformer | 0.4570 | 0.4455 | 0.6403 | 0.0597 | 0.6180 | 619.0 | 14.0 |
| ETTm1 | transformer | 0.7005 | 0.5913 | 0.4485 | 0.2218 | -0.4189 | 476.5 | 17.6 |
| ETTm1 | informer | 0.7606 | 0.5998 | 0.4013 | 0.2443 | -0.5626 | 468.6 | 12.0 |
| ETTm1 | lstm | 0.8648 | 0.6266 | 0.3194 | 0.1609 | -0.0291 | 277.6 | 12.2 |

### ETTh1

ETTh1 上 PatchTST 优势很稳：5 个 horizon 的多变量 MSE 和目标变量 MSE 全部最优。相比 Autoformer，PatchTST 的平均多变量 MSE 从 0.5462 降到 0.4752，相对下降约 13.0%；目标变量 MSE 从 0.1300 降到 0.0821，相对下降约 36.8%。

这说明在 ETTh1 小变量、小时级数据上，PatchTST 的 patch 建模和通道独立策略比 Autoformer 的分解结构更适合当前配置。

### ETTm1

ETTm1 上 PatchTST 和 Autoformer 更接近。PatchTST 的平均多变量 MSE 略优于 Autoformer：0.4500 对 0.4570；但 Autoformer 的平均目标变量 MSE 更低：0.0597 对 0.0722。

因此，如果报告以全部变量预测为主，ETTm1 仍可判定 PatchTST 略优；如果强调单一目标变量预测，ETTm1 应写成 Autoformer 与 PatchTST 互有胜负。

## 4. 各 horizon 最优模型

| 数据集 | Horizon | 多变量 MSE 最优 | MSE | Target MSE 最优 | Target MSE |
|---|---:|---|---:|---|---:|
| ETTh1 | 24 | patchtst | 0.3692 | patchtst | 0.0415 |
| ETTh1 | 48 | patchtst | 0.4152 | patchtst | 0.0574 |
| ETTh1 | 96 | patchtst | 0.4810 | patchtst | 0.0892 |
| ETTh1 | 168 | patchtst | 0.5261 | patchtst | 0.0952 |
| ETTh1 | 336 | patchtst | 0.5845 | patchtst | 0.1273 |
| ETTm1 | 24 | autoformer | 0.3030 | patchtst | 0.0175 |
| ETTm1 | 48 | patchtst | 0.4241 | autoformer | 0.0362 |
| ETTm1 | 96 | patchtst | 0.4757 | patchtst | 0.0439 |
| ETTm1 | 168 | patchtst | 0.4932 | patchtst | 0.0692 |
| ETTm1 | 336 | patchtst | 0.5442 | autoformer | 0.1201 |

## 5. 与 v1 正式结果对比

与 `archive/v1_results/experiments/formal_baseline/summaries/csv/formal_seed42_all.csv` 的 50 个同任务结果相比：

| 模型 | v1 Avg MSE | v2 Avg MSE | 变化 | v1 Target MSE | v2 Target MSE | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| patchtst | 0.4693 | 0.4626 | -1.4% | 0.0776 | 0.0772 | -0.6% |
| autoformer | 0.5147 | 0.5016 | -2.5% | 0.0894 | 0.0949 | +6.1% |
| transformer | 0.9173 | 0.8217 | -10.4% | 0.5464 | 0.3736 | -31.6% |
| informer | 0.8170 | 0.8901 | +8.9% | 0.5131 | 0.5254 | +2.4% |
| lstm | 0.9461 | 0.9871 | +4.3% | 0.3605 | 0.2651 | -26.5% |

整体看，v2 相比 v1 的平均多变量 MSE 几乎不变：0.7329 到 0.7326。但目标变量 MSE 从 0.3174 降到 0.2672，下降约 15.8%。

更重要的变化是胜出结构：

- v1：PatchTST 5 次最优，Autoformer 5 次最优
- v2：PatchTST 9 次最优，Autoformer 1 次最优

这说明 v2 的验证集最优配置和更长训练预算没有显著降低整体平均误差，但强化了 PatchTST 在跨 horizon 上的稳定优势。

## 6. 训练效率观察

从 ETTh1 和 ETTm1 的总训练耗时看：

- PatchTST 总耗时约 586.5 秒，平均 19.3 epoch，参数量约 113k。
- Autoformer 总耗时约 883.2 秒，平均 16.2 epoch，参数量约 111k。
- Transformer 总耗时约 826.0 秒，平均 25.6 epoch，参数量约 338k。
- LSTM 参数量最高，平均约 579k，但效果最弱。

PatchTST 在性能和效率之间的平衡最好：误差最低，同时耗时低于 Autoformer 和 Transformer。

## 7. 报告建议

可以在报告中采用如下表述：

> 在 ETTh1 和 ETTm1 的完整预测矩阵中，PatchTST 表现出最稳定的综合优势。在 10 个数据集-预测步长组合中，PatchTST 获得 9 个多变量 MSE 最优结果，并在 8 个目标变量 MSE 任务中最优。Autoformer 在验证集 loss 上表现最好，但其验证集优势未完全转化为测试集泛化优势。整体来看，PatchTST 更适合作为本实验的主模型结论，Autoformer 可作为强竞争基线；Transformer、Informer 和 LSTM 在当前配置下明显落后。

需要注意：

- 当前项目口径已收敛为 ETTh1 和 ETTm1 两个数据集，不再要求补齐 ECL 正式实验。
- ETTm1 的目标变量指标中 Autoformer 仍有优势，报告中不要简单写成 PatchTST 在所有指标上全面胜出。
- v2 相比 v1 的整体平均 MSE 改善有限，主要贡献是目标变量 MSE 改善和 PatchTST 稳定性增强。
