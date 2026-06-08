# Mac 开发 + Windows CUDA 训练计划

## Summary

以 `docs/选题-时间序列预测.md` 为根本，核心实验覆盖 `24/48/96/168/336` 五个步长。Mac 只负责代码修改、形状验证、小样本 smoke test 和 notebook/脚本整理；正式训练全部留给 Windows CUDA 执行。

## Key Changes

- 重点优化除 Autoformer 外的两种变体：
  - `Informer`：修正/启用 distilling，简化长 horizon decoder，保证 ProbSparse 逻辑可复现。
  - `PatchTST`：去掉按变量 Python 循环，改为 batch-channel 合并计算；补充 Channel Mixing 消融版本。
- Autoformer 保持现有轻量版，只补必要兼容和“去分解模块”消融。
- 修正设备选择逻辑：
  - Mac 测试默认 `mps`，不做长训练。
  - Windows CUDA 训练默认 `cuda`。
  - 若设备不可用，自动回退并在结果中记录 `device`。
- 更新训练配置：
  - 核心 horizon 固定为 `[24, 48, 96, 168, 336]`。
  - 数据集优先 `ETTh1 / ETTm1 / ECL`。
  - 结果保存继续使用 `{dataset}_h{horizon}_{model}_results.npy`。

## Workflow

### Mac 阶段

- 修改 `Informer` 和 `PatchTST` 实现。
- 运行模型前向 shape test：5 个模型 × 5 个 horizon。
- 运行极小样本 `FAST_DEV_RUN=True`，只验证训练、预测、保存结果流程。
- 整理 Windows CUDA 训练用 notebook 或脚本，避免正式训练依赖手动改很多参数。

### Windows CUDA 阶段

- 重新生成或同步 `data/processed/`。
- 跑核心矩阵：`3 datasets × 5 horizons × 5 models = 75` 个正式结果。
- 跑消融实验：Informer、Autoformer、PatchTST 各至少一个关键组件消融。

### 分析阶段

- 汇总 MSE、MAE、MAPE、R2、训练耗时、参数量。
- 绘制预测值 vs 真实值、残差图、horizon 衰减曲线。
- 更新报告中的超参数经验、复杂度分析、季节波动讨论。

## Test Plan

### Mac 只验

- Python 语法与 notebook JSON 正常。
- 所有模型输出形状正确。
- `ETTh1` 小样本训练 1-3 epoch 能完成并保存结果。

### Windows CUDA 验

- 75 个核心结果文件生成完整。
- 每个结果包含配置、样本数、指标、训练耗时、最佳 epoch。
- 消融结果能与原模型形成可比表格。

## Assumptions

- Mac 不承担正式训练耗时任务。
- Windows CUDA 是正式实验结果来源。
- `24/48` 与 `96/168/336` 同等纳入核心报告。
- 每完成独立阶段都更新 `docs/progress.md`。
