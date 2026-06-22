# v4_results：重做消融与完整单/多变量对比

## 版本定位

v4 于 2026-06-22 从项目根目录的 `results_v4/` 归档而来，包含两个不同实验。它们已拆分到独立目录，避免消融产物与单/多变量产物混在一起。

## 实验分类

| 实验目录 | 实验类型 | 数据范围 | 运行标签 | 实验组数 |
|---|---|---|---|---:|
| `experiments/ablation_study/` | Autoformer/PatchTST 重做消融 | ETTh1、ETTm1 × h96/h336 × 2 个同批次基线和 4 个消融变体 | `ablation_rerun_seed42` | 24 |
| `experiments/univariate_multivariate_comparison/` | 完整单变量与多变量对比 | ETTh1、ETTm1 × 5 个步长 × 5 个模型 × 2 种特征模式 | `feature_mode_full_seed42` | 100 |

## 查找方式

### 消融实验

- 单组结果：`experiments/ablation_study/results/h{96,336}/{数据集}/{基线或消融模型}/ablation_rerun_seed42/`
- CSV/Markdown 汇总：`experiments/ablation_study/summaries/`
- 运行状态：`experiments/ablation_study/run_state/`
- 图表：`experiments/ablation_study/figures/`
- 可视化 Notebook：`notebooks/visualization/v4_ablation_visualization.ipynb`

### 单变量与多变量对比

- 单组结果：`experiments/univariate_multivariate_comparison/results/h{步长}/{数据集}/{特征模式}/{模型}/feature_mode_full_seed42/`
- CSV/Markdown 汇总：`experiments/univariate_multivariate_comparison/summaries/`
- 图表：`experiments/univariate_multivariate_comparison/figures/`
- 可视化 Notebook：`notebooks/visualization/v4_univariate_multivariate_visualization.ipynb`

## 版本结论

- 消融实验中，移除 Autoformer 序列分解后 MSE 平均上升约 66.2%；取消 PatchTST Channel Independence 后平均上升约 139.2%。
- 单/多变量的 50 个配对任务中，单变量取得 43 次较低目标变量 MSE；多变量取得的 7 次优势全部来自 PatchTST。
