# v2_results：验证集最优配置完整矩阵

## 版本定位

v2 保存使用验证集最优配置重新训练的正式完整矩阵，运行标签为 `full_val_best_e50p10_tb_seed42`。训练预算为 `epochs=50`、`patience=10`，seed 为 42。

## 实验分类

| 实验目录 | 实验类型 | 数据范围 | 产物 |
|---|---|---|---|
| `experiments/validation_best_full_matrix/` | 调参后正式完整矩阵 | ETTh1、ETTm1 × 5 个步长 × 5 个模型，共 50 组 | 结果数组、JSON 指标、CSV/Markdown 汇总、运行状态和分析图 |

## 查找方式

- 单组实验：`experiments/validation_best_full_matrix/results/h{步长}/{数据集}/{模型}/full_val_best_e50p10_tb_seed42/`
- CSV 汇总：`experiments/validation_best_full_matrix/summaries/csv/`
- Markdown 汇总：`experiments/validation_best_full_matrix/summaries/md/`
- 图表：`experiments/validation_best_full_matrix/figures/`
- 运行状态：`experiments/validation_best_full_matrix/run_state/`
- 可视化 Notebook：`notebooks/visualization/v2_tuning_visualization.ipynb`

## 版本结论

在 ETTh1、ETTm1 与 5 个预测步长组成的 10 个任务中，PatchTST 获得 9 次最低 MSE，Autoformer 获得 1 次。
