# Data Directory

将 ECL、ETT、ILI 或 Traffic 等公开时间序列数据集的 CSV 文件放在本目录下。

第一版 notebook 默认不提交原始数据文件，原因是这些数据通常较大，并且可从公开来源重新下载。建议字段格式如下：

- 时间列：例如 `date`、`timestamp` 或 `time`。
- 目标列：例如 `OT`、`load`、`target`。
- 外生变量列：除时间列与目标列外的数值列。

notebook 会按时间顺序切分数据，并只用训练集统计量做标准化，避免数据泄露。

