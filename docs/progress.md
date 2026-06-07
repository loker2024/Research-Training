# 项目进度追踪

## 项目概述

基于 LSTM、Transformer、Informer、Autoformer、PatchTST 的长时序预测研究项目。

---

## 步骤 1：数据准备与预处理 ✅

**完成时间**：2026/06/07

**完成内容**：
1. ✅ 检查并安装依赖（torch, pandas, numpy, scikit-learn, matplotlib）
2. ✅ 从 GitHub 下载公开时间序列数据集（ETTh1、ETTm1、ECL）
3. ✅ 数据预处理：归一化（使用训练集统计量）、滑动窗口划分样本
4. ✅ 按时间顺序切分训练/验证/测试集，确保无数据泄露
5. ✅ 数据可视化：生成每个数据集的原始序列和归一化后序列图

**修改的文件**：
- `notebooks/data_preparation.ipynb` - 数据准备 notebook
- `data/processed/` - 生成的处理后数据目录

**数据集配置**：

| 数据集 | 变量数 | 频率 | 切分方式 | 训练/验证/测试样本数 (h336) |
|--------|--------|------|----------|----------------------------|
| ETTh1 | 7 | 小时 | 前12月/4月/4月 | 8,449 / 2,785 / 5,565 |
| ETTm1 | 7 | 15分钟 | 前12月/4月/4月 | 34,369 / 11,425 / 23,265 |
| ECL | 321 | 小时 | 70%/10%/20% | 18,221 / 2,535 / 4,929 |

**预测步长**：24、48、96、168、336（覆盖短中长期）
**回看窗口**：96

**测试结果**：
- ✅ 所有数据集下载成功
- ✅ 归一化参数保存正确
- ✅ 滑动窗口划分无 NaN/Inf 值
- ✅ 三张可视化图生成成功（ETTh1、ETTm1、ECL）

**下一步任务**：
1. 实现 LSTM 基线模型 ✅
2. 实现 Transformer 基线模型 ✅
3. 统一训练框架 ✅

---

## 步骤 2：基础模型实现 ✅

**完成时间**：2026/06/07

**完成内容**：
1. ✅ 实现 LSTM 基线模型
2. ✅ 实现 Transformer 基线模型
3. ✅ 统一训练框架（数据加载、训练循环、评估逻辑）

**修改的文件**：
- `models/__init__.py` - 模型库初始化
- `models/lstm.py` - LSTM 模型实现
- `models/transformer.py` - Transformer 模型实现
- `models/dataset.py` - 时序数据集加载器
- `models/trainer.py` - 训练框架
- `notebooks/train_baseline.ipynb` - 基础模型训练 notebook

**模型架构**：

| 模型 | 参数量 (ETTh1) | 特点 |
|------|---------------|------|
| LSTM | ~99,808 | 双层 LSTM，隐藏层 64 |
| Transformer | ~81,824 | 2 层，d_model=64，4 头注意力 |

**测试结果**：
- ✅ LSTM 模型测试通过（输入/输出形状正确）
- ✅ Transformer 模型测试通过（输入/输出形状正确）
- ✅ 所有预测步长测试通过（24/48/96/168/336）
- ✅ 设备切换测试通过（CPU/GPU）
- ✅ LSTM 训练示例运行成功（5 轮，验证损失 1.09）
- ✅ Transformer 训练示例运行成功（5 轮，验证损失 1.15）

**训练框架特性**：
- MSE 损失函数
- Adam 优化器
- ReduceLROnPlateau 学习率调度（factor=0.5, patience=5）
- 早停机制（patience=10）
- 模型保存与加载
- 评估指标计算（MSE、MAE、MAPE、R²）
- R² 决定系数（准确率）

**Baseline 训练结论（ETTh1, Horizon=24）**：

| 模型 | MSE | MAE | R² | 验证最佳 Epoch | 早停 |
|------|-----|-----|-----|---------------|------|
| LSTM | 1.246 | 0.831 | 0.025 | Epoch 3 | Epoch 23 |
| Transformer | 0.777 | 0.656 | 0.392 | Epoch 25 | Epoch 40 |

**主要发现**：
1. **LSTM 严重过拟合**：训练 R² 达 0.83，验证 R² 仅 0.33（最佳）→ 0.13（早停时），3 轮后验证损失持续上升
2. **Transformer 过拟合较轻**：训练 R² 0.75，验证 R² 0.47，收敛更稳定
3. **架构局限**：两个模型均只用最后时间步输出，信息压缩严重，预测能力受限
4. **Baseline 效果一般**：作为后续高效模型（Informer/Autoformer/PatchTST）的对比基准

---

## 步骤 3：高效变体模型实现 🔄

**开始时间**：2026/06/07

**完成内容**：
1. ✅ 实现 Informer 模型
2. ✅ 实现 Autoformer 模型
3. ✅ 实现 PatchTST 模型
4. ✅ 创建训练 notebook
5. ⏳ 等待运行训练实验

**修改的文件**：
- `models/informer.py` - Informer 模型实现
- `models/autoformer.py` - Autoformer 模型实现
- `models/patchtst.py` - PatchTST 模型实现
- `models/__init__.py` - 更新模型导出
- `notebooks/train_variants.ipynb` - 变体模型训练 notebook

**模型架构**：

| 模型 | 核心创新 | 参数量 (ETTh1) |
|------|----------|---------------|
| Informer | ProbSparse 注意力 O(L log L) | ~200k |
| Autoformer | 序列分解 + Auto-Correlation | ~250k |
| PatchTST | Patch 切分 + Channel Independence | ~150k |

**模型特点**：
- **Informer**：ProbSparse 注意力只计算 Top-K 重要 query，生成式解码器一次性输出所有预测步
- **Autoformer**：显式分解趋势和季节性，基于 FFT 的 Auto-Correlation 替代点积注意力
- **PatchTST**：将时间序列切分为 patches（类似 ViT），每个变量独立建模（Channel Independence）

**下一步任务**：
1. 在 ETTh1 上运行训练实验
2. 在 ETTm1 和 ECL 上运行训练实验
3. 测试不同预测步长（48, 96, 168, 336）
4. 与基线模型对比

---

## 资料整理：Transformer 高效变体原始论文 ✅

**完成时间**：2026/06/07 15:42

**完成内容**：
1. ✅ 下载 Informer 原始论文：Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting
2. ✅ 下载 Autoformer 原始论文：Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting
3. ✅ 下载 PatchTST 原始论文：A Time Series is Worth 64 Words: Long-term Forecasting with Transformers

**修改的文件**：
- `papers/Informer_Beyond_Efficient_Transformer_for_Long_Sequence_Time-Series_Forecasting.pdf`
- `papers/Autoformer_Decomposition_Transformers_with_Auto-Correlation_for_Long-Term_Series_Forecasting.pdf`
- `papers/PatchTST_A_Time_Series_is_Worth_64_Words_Long-term_Forecasting_with_Transformers.pdf`
- `docs/progress.md`

**测试结果**：
- ✅ 三个文件均下载成功
- ✅ 三个文件的文件头均为 `%PDF-`
- ✅ 文件大小校验正常：Informer 7.0 MB，Autoformer 11.6 MB，PatchTST 3.8 MB

**下一步任务**：
1. 阅读三篇论文并提取模型结构、核心公式和实验设置
2. 对照当前实现检查关键模块是否与论文描述一致

---

## 项目状态阅读与交接 ✅

**完成时间**：2026/06/07 16:26

**完成内容**：
1. ✅ 阅读项目说明、实施步骤、进度记录与数据说明
2. ✅ 梳理模型代码、训练框架、notebook 入口、检查点和已有结果
3. ✅ 检查 ETTh1、ETTm1、ECL 的预处理元信息
4. ✅ 运行模型前向形状 smoke test，确认 5 个模型统一输出 `(batch, horizon, features)`

**修改的文件**：
- `docs/progress.md` - 追加本次项目阅读与交接记录

**测试结果**：
- ✅ LSTM、Transformer、Informer、Autoformer、PatchTST 在输入 `(2, 96, 7)`、预测步长 24 下均输出 `(2, 24, 7)`
- ✅ 已读取 `results/ETTh1_h24_results.npy` 与 `results/ETTh1_h96_results.npy`
- ⚠️ 当前项目根目录未发现 `requirements.txt` 与 `myenv`

**下一步任务**：
1. 修正项目说明与实际实验配置不一致的问题（预测步长 96/168/336 vs 24/48/96/168/336）
2. 检查并修复 Transformer encoder 层复用同一实例的问题
3. 继续运行 ETTh1、ETTm1、ECL 的核心实验，并补齐结果记录

---

## Git 忽略规则与同步准备 ✅

**完成时间**：2026/06/07 16:33

**完成内容**：
1. ✅ 更新 `.gitignore`，忽略本地虚拟环境、IDE/助手配置、原始数据、预处理数据、训练权重、日志和结果目录
2. ✅ 更新 `README.md`，补充当前版本的项目范围、目录结构和运行顺序
3. ✅ 补齐 `requirements.txt`，加入当前训练框架使用的依赖
4. ✅ 更新 `data/README.md`，修正已删除文档链接

**修改的文件**：
- `.gitignore`
- `README.md`
- `requirements.txt`
- `data/README.md`
- `docs/progress.md`

**测试结果**：
- ✅ `data/*.csv`、`data/processed/`、`checkpoints/`、`runs/`、`results/`、`.vscode/`、`.claude/` 均被 Git 忽略
- ✅ 三个 notebook JSON 均可解析
- ✅ 5 个模型前向形状 smoke test 通过
- ✅ 待提交文件中未发现超过 100MB 的文件

**下一步任务**：
1. 提交当前整理后的项目文件
2. 推送到 GitHub 远程仓库

---

## Autoformer 轻量化与训练 notebook 重写 ✅

**完成时间**：2026/06/07 16:52

**完成内容**：
1. ✅ 重写 `models/autoformer.py`，保留序列分解与 FFT Auto-Correlation，但改为全局 Top-K lag + `torch.roll` 聚合，减少逐 batch/head gather 开销
2. ✅ 将 Autoformer 解码器改为 Direct-Forecast 时间投影，去掉 horizon 维度上的自注意力和交叉注意力堆叠
3. ✅ 重写 `notebooks/train_variants.ipynb`，默认只快速验证 Autoformer，并通过 `MODELS_TO_RUN` 控制是否训练 Informer / PatchTST
4. ✅ notebook 默认切换到项目要求的预测步长 `96/168/336`，修正 `ETTm1` 数据集名称大小写
5. ✅ 增加 `FAST_DEV_RUN`、`SAMPLE_LIMITS` 和按数据集设置 batch size，避免一次训练默认全量跑 100 轮

**修改的文件**：
- `models/autoformer.py` - 轻量 Direct-Forecast Autoformer 实现
- `notebooks/train_variants.ipynb` - 可控快速训练模板
- `docs/progress.md` - 追加本次优化记录

**测试结果**：
- ✅ `python -m py_compile models/autoformer.py models/__init__.py` 通过
- ✅ `notebooks/train_variants.ipynb` JSON 解析通过
- ✅ Autoformer 前向形状测试通过：`(2, 96, 7) -> (2, 96, 7)`
- ✅ ECL 规模前向形状测试通过：`(1, 96, 321) -> (1, 336, 321)`
- ✅ ETTh1 h96 小样本训练 smoke test 通过：CPU 上 128 个训练样本、64 个验证样本完成 1 轮，epoch 训练时间约 0.2s

**下一步任务**：
1. 用 `FAST_DEV_RUN=True` 依次跑通 ETTh1 的 `96/168/336`
2. 关闭快速验证后运行正式核心实验，并记录每个模型的训练耗时与指标
3. 若要跑 ECL 完整实验，优先处理当前压缩 `.npz` 文件过大的加载问题

---

## train_variants notebook 中文乱码修复 ✅

**完成时间**：2026/06/07 16:58

**完成内容**：
1. ✅ 修复 `notebooks/train_variants.ipynb` 中被写成 `?` 的中文 Markdown 标题和说明
2. ✅ 同步修复代码单元中的中文注释、输出提示、图表标题和标签

**修改的文件**：
- `notebooks/train_variants.ipynb` - 恢复中文文案
- `docs/progress.md` - 追加本次修复记录

**测试结果**：
- ✅ notebook JSON 可正常解析
- ✅ `rg "??" notebooks/train_variants.ipynb` 未发现连续问号乱码残留

**下一步任务**：
1. 在 Jupyter 中重新打开 `train_variants.ipynb` 检查显示效果
2. 继续执行 Autoformer 快速验证训练

---

## train_variants 结果保存逻辑修正 ✅

**完成时间**：2026/06/07 17:15

**完成内容**：
1. ✅ 修正多模型训练时只保存一个 `variant_results` 文件的问题
2. ✅ 改为按模型分别保存结果文件，命名格式为 `{DATASET}_h{HORIZON}_{model_name}_results.npy`
3. ✅ 清理 notebook 中残留的旧执行输出，避免继续显示 `h24_variant_results.npy`

**修改的文件**：
- `notebooks/train_variants.ipynb` - 调整结果保存单元
- `docs/progress.md` - 追加本次修复记录

**测试结果**：
- ✅ notebook JSON 可正常解析
- ✅ `rg "variant_results|variant_|变体模型结果|h24_variant" notebooks/train_variants.ipynb` 未发现旧命名残留

**下一步任务**：
1. 重新运行保存结果单元，确认生成以模型名命名的结果文件
2. 继续执行 Autoformer 快速验证训练

---

## train_variants 实验配置与结果字段补充 ✅

**完成时间**：2026/06/07 17:22

**完成内容**：
1. ✅ 将训练学习率提到配置单元，新增 `LEARNING_RATE`，避免在 `Trainer(..., lr=...)` 中硬编码
2. ✅ 每个模型结果文件新增训练配置字段：`epochs`、`trained_epochs`、`best_epoch`、`patience`、`batch_size`、`learning_rate`、`device`
3. ✅ 每个模型结果文件新增数据与模型字段：`train_samples`、`val_samples`、`test_samples`、`input_size`、`target_idx`、`model_params`
4. ✅ 每个模型结果文件新增训练摘要字段：`train_time_seconds`、`best_val_loss`、`best_val_r2`

**修改的文件**：
- `notebooks/train_variants.ipynb` - 更新配置、训练和结果保存单元
- `docs/progress.md` - 追加本次修复记录

**测试结果**：
- ✅ notebook JSON 可正常解析
- ✅ `Trainer(model, device=DEVICE, lr=LEARNING_RATE)` 已替代硬编码学习率
- ✅ 保存结果结构包含 epoch、学习率、样本量、模型参数量和最佳轮次等实验元信息

**下一步任务**：
1. 重新运行训练单元和保存单元，检查生成的模型结果文件字段
2. 继续执行 Autoformer 快速验证训练

---

## train_variants 批量实验循环改造 ✅

**完成时间**：2026/06/07 17:35

**完成内容**：
1. ✅ 将单变量 `DATASET`、`HORIZON` 改为列表配置 `DATASETS_TO_RUN`、`HORIZONS_TO_RUN`
2. ✅ 保留 `MODELS_TO_RUN` 列表配置，形成 `dataset × horizon × model` 的批量实验循环
3. ✅ 将数据加载封装为 `create_experiment_data(dataset_name, horizon)`，每个数据集和预测步长组合会重新加载对应数据
4. ✅ 将模型构建改为 `build_model(model_name, input_size, horizon)`，避免依赖全局 `HORIZON`
5. ✅ 训练完成后立即按 `{dataset}_h{horizon}_{model}_results.npy` 保存单模型结果，防止批量实验中途失败导致前面结果丢失
6. ✅ 清理 notebook 旧执行输出，避免旧单实验结果干扰当前批量模板

**修改的文件**：
- `notebooks/train_variants.ipynb` - 改造为批量实验模板
- `docs/progress.md` - 追加本次修复记录

**测试结果**：
- ✅ notebook JSON 可正常解析
- ✅ 所有代码单元可通过 Python 语法解析
- ✅ 导入、配置、数据函数和模型构建函数定义单元可执行
- ✅ 未发现 `DATASET = ...`、`HORIZON = ...` 单实验旧赋值残留
- ✅ `rg "??" notebooks/train_variants.ipynb` 未发现连续问号乱码残留

**下一步任务**：
1. 用默认 `DATASETS_TO_RUN=[DATASETS[0]]`、`HORIZONS_TO_RUN=[HORIZON_LIST[0]]` 先跑通单组合
2. 再逐步扩展为多个 horizon 或多个 dataset，避免一次性触发过大的 ECL 完整训练

---

## 待办事项

- [ ] 步骤 4：核心实验运行
- [ ] 步骤 5：消融实验
- [ ] 步骤 6：可视化与深入分析
- [ ] 步骤 7：撰写实验报告
