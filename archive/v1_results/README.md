# v1_results：未调参正式基线

## 版本定位

v1 保存项目第一版正式核心实验，运行标签为 `formal_seed42`。这一版使用五个模型的初始配置，主要作用是建立后续调参和结构分析的比较基线。

## 实验分类

| 实验目录 | 实验类型 | 数据范围 | 产物 |
|---|---|---|---|
| `experiments/formal_baseline/` | 未调参正式基线 | ETTh1、ETTm1 × 5 个步长 × 5 个模型，共 50 组 | 结果数组、JSON 指标、CSV/Markdown 汇总 |

## 查找方式

- 单组实验：`experiments/formal_baseline/results/h{步长}/{数据集}/{模型}/formal_seed42/`
- CSV 汇总：`experiments/formal_baseline/summaries/csv/`
- Markdown 汇总：`experiments/formal_baseline/summaries/md/`

> 旧版 v1 可视化图片和旧报告草稿已清理；当前报告写作优先引用 v2/v4 新图与最新汇总表。

## 版本结论

- ETTh1 的 5 个预测步长均由 PatchTST 取得最低 MSE。
- ETTm1 的 5 个预测步长均由 Autoformer 取得最低 MSE。
- 旧版 `ablation_seed42` 消融产物已经清理，因此 v1 当前只包含正式基线实验。
