# Archive 实验归档索引

> `archive/` 按“版本 → 实验类型 → 产物类型”组织。每个版本根目录的 `README.md` 记录该版实验范围、运行标签和主要结论。

## 四个版本总览

| 版本 | 实验目录 | 实验含义 | 运行标签 | 实验组数 |
|---|---|---|---|---:|
| v1 | `v1_results/experiments/formal_baseline/` | 未调参五模型正式基线 | `formal_seed42` | 50 |
| v2 | `v2_results/experiments/validation_best_full_matrix/` | 验证集最优配置五模型完整矩阵 | `full_val_best_e50p10_tb_seed42` | 50 |
| v3 | `v3_results/experiments/univariate_multivariate_comparison/` | h96/h336 单变量与多变量阶段性对比 | `feature_mode_full_seed42` | 40 |
| v4 | `v4_results/experiments/ablation_study/` | Autoformer/PatchTST 重做消融 | `ablation_rerun_seed42` | 24 |
| v4 | `v4_results/experiments/univariate_multivariate_comparison/` | 五步长单变量与多变量完整对比 | `feature_mode_full_seed42` | 100 |

## 版本关系

1. **v1：基线版**
   使用初始模型配置建立正式比较基线。当前只保留 50 组正式核心实验；旧 `ablation_seed42` 已清理。

2. **v2：调参后核心矩阵**
   使用验证集最优配置和统一训练预算重跑 50 组五模型矩阵，是正式模型性能比较的主要依据。

3. **v3：单/多变量阶段版**
   先完成 h96、h336 两个代表性步长，共 40 组，用于确认 feature mode 实验流程。

4. **v4：补充实验完成版**
   同时收录 24 组重做消融和 100 组完整单/多变量对比。原根目录 `results_v4/` 已整体迁入本版本。

## 统一目录规则

```text
archive/
├── log.md
├── v1_results/
│   ├── README.md
│   └── experiments/formal_baseline/
├── v2_results/
│   ├── README.md
│   └── experiments/validation_best_full_matrix/
├── v3_results/
│   ├── README.md
│   └── experiments/univariate_multivariate_comparison/
└── v4_results/
    ├── README.md
    └── experiments/
        ├── ablation_study/
        └── univariate_multivariate_comparison/
```

每个实验目录继续按以下产物类型分类：

- `results/`：单组实验的 `_summary.json` 与 `_results.npy`
- `summaries/csv/`：机器可读汇总和对比表
- `summaries/md/`：便于直接阅读的 Markdown 表
- `figures/`：可视化图表（该实验存在时）
- `run_state/`：断点续跑或完成状态（该实验存在时）

## 快速定位

- 比较未调参与调参后五模型表现：看 v1 的 `formal_baseline` 与 v2 的 `validation_best_full_matrix`
- 查看结构组件贡献：看 v4 的 `ablation_study`
- 查看单变量和多变量输入差异：优先看覆盖完整的 v4；v3 仅作为 h96/h336 阶段快照

## 可视化 Notebook

- `notebooks/visualization/v2_tuning_visualization.ipynb`：v2 调优结果，并与 v1 基线严格配对比较
- `notebooks/visualization/v4_ablation_visualization.ipynb`：v4 重做消融，不引用旧消融结果
- `notebooks/visualization/v4_univariate_multivariate_visualization.ipynb`：v4 五步长完整单/多变量分析

旧的 `v2_results_visualization.ipynb` 和 `v3_results_visualization.ipynb` 已由上述三个入口替代。
