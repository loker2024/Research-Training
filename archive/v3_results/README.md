# v3_results：早期单变量与多变量对比

## 版本定位

v3 是单变量/多变量正式对比的阶段性归档，运行标签为 `feature_mode_full_seed42`。这一版只完成 h96 和 h336，用来先验证两种输入模式的差异。

## 实验分类

| 实验目录 | 实验类型 | 数据范围 | 产物 |
|---|---|---|---|
| `experiments/univariate_multivariate_comparison/` | 单变量与多变量对比（阶段版） | ETTh1、ETTm1 × 2 个步长 × 5 个模型 × 2 种特征模式，共 40 组 | 结果数组、JSON 指标、CSV/Markdown 对比表 |

## 查找方式

- 单组实验：`experiments/univariate_multivariate_comparison/results/h{96,336}/{数据集}/{特征模式}/{模型}/feature_mode_full_seed42/`
- CSV 汇总：`experiments/univariate_multivariate_comparison/summaries/csv/`
- Markdown 汇总：`experiments/univariate_multivariate_comparison/summaries/md/`

> v3 是阶段性结果；覆盖全部 5 个步长的同类实验位于 v4。
