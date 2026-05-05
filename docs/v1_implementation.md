# 第一版实现文档：长时序预测 PyTorch 实验骨架

## 1. 课题依据

结合仓库中的选题说明与研究报告标题，本课题关注 LSTM、Transformer、Informer、Autoformer 与 PatchTST 在长时序预测任务中的对比。报告要求覆盖多变量/单变量预测、不同预测步长、滑动窗口构造、时间顺序切分、MSE/MAE/MAPE 指标、预测曲线、残差分析、消融实验、复杂度分析与季节性讨论。

第一版代码优先解决“能规范跑通实验”的问题，因此实现 LSTM、Transformer 和 PatchTST-lite 三个代表模型，并把时序分解作为可开关模块预留出来。Informer 与 Autoformer 更适合作为第二版在同一训练框架中继续补充。

## 2. 第一版功能

- 数据输入：支持 CSV 数据，也支持无数据时使用合成季节性数据进行流程验证。
- 样本构造：使用 `seq_len` 历史窗口预测 `pred_len` 未来窗口。
- 数据划分：按时间顺序划分训练、验证、测试集，不随机打乱原始时间轴。
- 标准化：只在训练集上拟合均值和标准差，再应用到验证/测试集。
- 模型：
  - `LSTMForecaster`：RNN 基线，用最后时刻隐藏状态映射未来序列。
  - `TransformerForecaster`：标准编码器式 Transformer 基线。
  - `PatchTSTLite`：按变量切分 patch，使用 Transformer 编码 patch 序列，适合长输入窗口。
- 指标：MSE、MAE、MAPE。
- 可视化：预测值与真实值曲线、残差曲线。
- 消融：PatchTST-lite 支持 `use_decomposition` 开关，用于初步验证趋势/季节分解的影响。

## 3. 建议实验设置

| 场景 | `seq_len` | `pred_len` | 说明 |
| --- | ---: | ---: | --- |
| 短期预测 | 96 | 24 或 48 | 验证基础拟合能力 |
| 中期预测 | 96 或 168 | 96 | 对比模型稳定性 |
| 长期预测 | 336 | 168 或 336 | 观察误差累积和性能衰减 |

建议先用 ETT 或 ECL 数据验证流程，再扩展到 Traffic 或 ILI。每次实验固定随机种子，并记录模型参数量、训练耗时和显存占用。

## 4. 当前限制

- PatchTST-lite 是教学与第一版实验用实现，不等同于论文完整 PatchTST。
- 尚未实现 ProbSparse Attention、Auto-Correlation 等 Informer/Autoformer 核心结构。
- 未内置超参数搜索器，第一版建议手动网格尝试 `d_model`、`lr`、`batch_size`、`seq_len` 和 `pred_len`。

## 5. 下一版计划

1. 增加 Informer 的 ProbSparse Attention 与长序列 decoder。
2. 增加 Autoformer 的序列分解与 Auto-Correlation block。
3. 增加实验记录表，将多数据集、多预测步长结果自动汇总为 CSV。
4. 增加复杂度统计，包括参数量、单 epoch 时间、推理延迟和显存峰值。
5. 为报告生成预测图、残差图和消融结果表。

