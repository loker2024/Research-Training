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

---

## 命令行实验入口与 ETTh1 快速验证 ✅

**完成时间**：2026/06/08 10:30

**完成内容**：
1. ✅ 修复 `TransformerModel` 多层 encoder 复用同一个 `TransformerEncoderLayer` 实例的问题，改为每层独立参数
2. ✅ 新增 `scripts/preprocess_data.py`，将数据预处理 notebook 的核心流程脚本化，支持按数据集和预测步长生成 `data/processed`
3. ✅ 新增 `scripts/run_experiments.py`，支持按 `dataset × horizon × model` 命令行运行实验，并保存 `.npy` 结果和 `.json` 摘要
4. ✅ 明确环境说明：Mac 使用 Conda 环境 `miniMac`，Windows 使用项目根目录下的 `myenv`
5. ✅ 恢复并保留 `myenv` 目录，供 Windows 环境使用

**修改的文件**：
- `models/transformer.py` - 修复 encoder 层参数共享问题
- `scripts/preprocess_data.py` - 新增命令行预处理脚本
- `scripts/run_experiments.py` - 新增命令行实验脚本
- `README.md` - 更新环境和脚本运行说明
- `AGENTS.md` - 更新跨平台环境说明
- `CLAUDE.md` - 更新跨平台环境说明
- `docs/progress.md` - 追加本次进度记录

**测试结果**：
- ✅ `/opt/miniconda3/envs/miniMac/bin/python -m py_compile models/transformer.py scripts/preprocess_data.py scripts/run_experiments.py` 通过
- ✅ Transformer 两层 encoder 实例和参数指针均不同
- ✅ `python scripts/preprocess_data.py --datasets ETTh1 --horizons 96` 生成或识别 ETTh1 h96 预处理数据
- ✅ ETTh1 h96 样本规模：train `(8449, 96, 7)`，val `(2689, 96, 7)`，test `(5709, 96, 7)`
- ✅ 使用 `miniMac` 跑通 Autoformer 快速验证：128 个训练/验证/测试样本，1 epoch，MPS 设备，训练耗时约 0.8s
- ✅ 结果保存至 `results/ETTh1_h96_autoformer_results.npy` 和 `results/ETTh1_h96_autoformer_summary.json`

**下一步任务**：
1. 用脚本继续补齐 ETTh1 的 `96/168/336` 核心实验
2. 再扩展到 ETTm1；ECL 完整实验需注意预处理文件体积和训练耗时
3. 汇总结果表，进入步骤 5 消融实验设计

---

## ETTh1 快速核心实验矩阵 ✅

**完成时间**：2026/06/08 10:37

**完成内容**：
1. ✅ 生成 ETTh1 的 `h168` 和 `h336` 预处理数据，补齐核心步长 `96/168/336`
2. ✅ 使用 `miniMac` 在 MPS 上运行 ETTh1 的 `3 个 horizon × 5 个模型` 快速实验矩阵
3. ✅ 每个组合使用 `sample_limit=128`、`epochs=1`，用于验证核心实验流程和结果文件结构
4. ✅ 新增 `scripts/summarize_results.py`，将 `results/*_summary.json` 汇总为 CSV 和 Markdown 表格
5. ✅ 生成 `results/ETTh1_quick_summary.csv` 与 `results/ETTh1_quick_summary.md`

**修改的文件**：
- `scripts/summarize_results.py` - 新增结果汇总脚本
- `README.md` - 补充汇总脚本运行说明
- `docs/progress.md` - 追加本次实验记录

**生成的本地结果文件**：
- `data/processed/ETTh1/h168/{train,val,test}.npz`
- `data/processed/ETTh1/h336/{train,val,test}.npz`
- `results/ETTh1_h{96,168,336}_{lstm,transformer,informer,autoformer,patchtst}_results.npy`
- `results/ETTh1_h{96,168,336}_{lstm,transformer,informer,autoformer,patchtst}_summary.json`
- `results/ETTh1_quick_summary.csv`
- `results/ETTh1_quick_summary.md`

**测试结果**：
- ✅ `python -m py_compile scripts/summarize_results.py scripts/run_experiments.py scripts/preprocess_data.py models/transformer.py` 通过
- ✅ ETTh1 h168 样本规模：train `(8377, 96, 7)`，val `(2617, 96, 7)`，test `(5637, 96, 7)`
- ✅ ETTh1 h336 样本规模：train `(8209, 96, 7)`，val `(2449, 96, 7)`，test `(5469, 96, 7)`
- ✅ 15 个快速实验组合全部完成并保存结果
- ✅ 汇总脚本不依赖 `tabulate`，可直接生成 Markdown 表

**快速实验摘要（非正式最终指标）**：

| Horizon | 最低测试 MSE 模型 | MSE | MAE | R² |
| --- | --- | ---: | ---: | ---: |
| 96 | PatchTST | 1.100747 | 0.699694 | 0.066844 |
| 168 | PatchTST | 1.168368 | 0.722849 | 0.037951 |
| 336 | LSTM | 1.050159 | 0.711227 | -0.015458 |

**下一步任务**：
1. 将快速实验扩展到 ETTm1 的 `96/168/336`
2. 选择合适 epoch 和 sample 设置，开始 ETTh1 正式核心实验
3. 为正式结果增加随机种子控制和重复实验统计

---

## ETTm1 快速核心实验矩阵 ✅

**完成时间**：2026/06/08 10:38

**完成内容**：
1. ✅ 生成 ETTm1 的 `h96/h168/h336` 预处理数据
2. ✅ 使用 `miniMac` 在 MPS 上运行 ETTm1 的 `3 个 horizon × 5 个模型` 快速实验矩阵
3. ✅ 每个组合使用 `sample_limit=128`、`epochs=1`，与 ETTh1 快速验证保持一致
4. ✅ 生成 `results/ETTm1_quick_summary.csv` 与 `results/ETTm1_quick_summary.md`
5. ✅ 生成 ETTh1 + ETTm1 合并汇总：`results/quick_summary.csv` 与 `results/quick_summary.md`

**修改的文件**：
- `docs/progress.md` - 追加本次实验记录

**生成的本地结果文件**：
- `data/processed/ETTm1/h{96,168,336}/{train,val,test}.npz`
- `results/ETTm1_h{96,168,336}_{lstm,transformer,informer,autoformer,patchtst}_results.npy`
- `results/ETTm1_h{96,168,336}_{lstm,transformer,informer,autoformer,patchtst}_summary.json`
- `results/ETTm1_quick_summary.csv`
- `results/ETTm1_quick_summary.md`
- `results/quick_summary.csv`
- `results/quick_summary.md`

**测试结果**：
- ✅ ETTm1 h96 样本规模：train `(34369, 96, 7)`，val `(11329, 96, 7)`，test `(23409, 96, 7)`
- ✅ ETTm1 h168 样本规模：train `(34297, 96, 7)`，val `(11257, 96, 7)`，test `(23337, 96, 7)`
- ✅ ETTm1 h336 样本规模：train `(34129, 96, 7)`，val `(11089, 96, 7)`，test `(23169, 96, 7)`
- ✅ 15 个 ETTm1 快速实验组合全部完成并保存结果
- ✅ 合并汇总表覆盖 ETTh1/ETTm1 的 30 个快速实验组合

**快速实验摘要（非正式最终指标）**：

| Dataset | Horizon | 最低测试 MSE 模型 | MSE | MAE | R² |
| --- | ---: | --- | ---: | ---: | ---: |
| ETTh1 | 96 | PatchTST | 1.100747 | 0.699694 | 0.066844 |
| ETTh1 | 168 | PatchTST | 1.168368 | 0.722849 | 0.037951 |
| ETTh1 | 336 | LSTM | 1.050159 | 0.711227 | -0.015458 |
| ETTm1 | 96 | LSTM | 1.192929 | 0.721122 | -0.047753 |
| ETTm1 | 168 | LSTM | 1.136494 | 0.725476 | -0.038627 |
| ETTm1 | 336 | LSTM | 1.178203 | 0.746393 | -0.025365 |

**下一步任务**：
1. 评估 ECL 预处理和快速实验的磁盘/内存成本，必要时先用更小 sample 或单模型 smoke test
2. 为正式核心实验加入固定随机种子、可复现实验配置文件和结果汇总标记
3. 开始 ETTh1/ETTm1 的正式多 epoch 核心实验

---

## Informer/PatchTST 变体优化与重训 ✅

**完成时间**：2026/06/08 10:44

**完成内容**：
1. ✅ 优化 Informer：保留 ProbSparse 编码器，将 horizon 维度上的自注意力/交叉注意力解码器替换为 Direct-Forecast 时间投影 + MLP refinement
2. ✅ 优化 PatchTST：将逐变量 Python 循环改为 `(B*C, L, 1)` 批量前向，保持 Channel Independence，同时显著降低 ECL 高维数据开销
3. ✅ 为实验脚本补充复现实验字段：`--seed`、`--run-tag`、`--data-dir`
4. ✅ 为预处理脚本补充 ECL smoke 能力：`--output-dir` 和 `--max-samples-per-split`
5. ✅ 为汇总脚本补充 `--run-tags` 过滤，避免优化后重训结果与旧结果混在一起
6. ✅ 使用优化后的 Informer/PatchTST 重新训练 ETTh1/ETTm1 的 `96/168/336` 快速矩阵，统一标记为 `run_tag=optv2`
7. ✅ 使用单独的 `data/processed_smoke` 跑通 ECL h96 的 Informer/PatchTST 高维 smoke test

**修改的文件**：
- `models/informer.py` - 轻量 Direct-Forecast Informer 解码器
- `models/patchtst.py` - 向量化 Channel Independence 前向
- `scripts/preprocess_data.py` - 增加输出目录和 split 样本上限参数
- `scripts/run_experiments.py` - 增加随机种子、run tag 和数据目录参数
- `scripts/summarize_results.py` - 增加 run tag 过滤和汇总字段
- `README.md` - 补充优化变体重训和 ECL smoke 命令示例
- `docs/progress.md` - 追加本次优化与训练记录

**生成的本地结果文件**：
- `results/ETTh1_h{96,168,336}_{informer,patchtst}_optv2_results.npy`
- `results/ETTh1_h{96,168,336}_{informer,patchtst}_optv2_summary.json`
- `results/ETTm1_h{96,168,336}_{informer,patchtst}_optv2_results.npy`
- `results/ETTm1_h{96,168,336}_{informer,patchtst}_optv2_summary.json`
- `results/optimized_variants_quick_summary.csv`
- `results/optimized_variants_quick_summary.md`
- `data/processed_smoke/ECL/h96/{train,val,test}.npz`
- `results/ECL_h96_{informer,patchtst}_ecl_smoke_optv2_results.npy`
- `results/ECL_h96_{informer,patchtst}_ecl_smoke_optv2_summary.json`
- `results/ECL_smoke_optv2_summary.csv`
- `results/ECL_smoke_optv2_summary.md`

**测试结果**：
- ✅ `python -m py_compile models/informer.py models/patchtst.py scripts/run_experiments.py scripts/summarize_results.py scripts/preprocess_data.py` 通过
- ✅ Informer 前向：`(2, 96, 7) -> (2, 96, 7)`，ECL 维度 `(1, 96, 321) -> (1, 336, 321)`
- ✅ PatchTST 前向：`(2, 96, 7) -> (2, 96, 7)`，ECL 维度 `(1, 96, 321) -> (1, 336, 321)`，ECL 前向约 0.02s
- ✅ ETTh1/ETTm1 的 12 个优化变体快速重训组合全部完成并保存结果
- ✅ ECL h96 smoke 预处理完成：train/val/test 均为 `(256, 96, 321)`，`data/processed_smoke` 占用约 159MB
- ✅ ECL h96 smoke 训练完成：Informer 测试 MSE `0.955738`，PatchTST 测试 MSE `0.745080`

**优化变体快速重训摘要（非正式最终指标）**：

| Dataset | Horizon | Model | MSE | MAE | R² |
| --- | ---: | --- | ---: | ---: | ---: |
| ETTh1 | 96 | Informer | 1.399250 | 0.837558 | -0.186211 |
| ETTh1 | 96 | PatchTST | 1.094928 | 0.692672 | 0.071777 |
| ETTh1 | 168 | Informer | 1.546414 | 0.873675 | -0.273338 |
| ETTh1 | 168 | PatchTST | 1.208124 | 0.746631 | 0.005215 |
| ETTh1 | 336 | Informer | 1.200505 | 0.772854 | -0.160836 |
| ETTh1 | 336 | PatchTST | 1.058842 | 0.707415 | -0.023854 |
| ETTm1 | 96 | Informer | 1.318025 | 0.769781 | -0.157625 |
| ETTm1 | 96 | PatchTST | 1.235166 | 0.715740 | -0.084850 |
| ETTm1 | 168 | Informer | 1.199626 | 0.741280 | -0.096322 |
| ETTm1 | 168 | PatchTST | 1.180158 | 0.726874 | -0.078531 |
| ETTm1 | 336 | Informer | 1.289300 | 0.803583 | -0.122050 |
| ETTm1 | 336 | PatchTST | 1.230023 | 0.751179 | -0.070462 |

**下一步任务**：
1. 基于 `run_tag` 和 `seed` 机制，制定正式核心实验配置（epochs、patience、sample/full 数据）
2. 对 ETTh1/ETTm1 先跑正式多 epoch 实验，避免 ECL 全量预处理过早占用大量磁盘
3. 根据正式结果进入步骤 5 消融实验

---

## 正式实验配置化入口 ✅

**完成时间**：2026/06/08 10:46

**完成内容**：
1. ✅ 为 `scripts/run_experiments.py` 增加 `--config` 参数，支持从 JSON 配置文件读取实验设置
2. ✅ 保留命令行覆盖能力，配置文件用于默认值，便于正式实验复现
3. ✅ 新增 `configs/core_experiment_smoke.json`，用于快速验证配置化入口
4. ✅ 新增 `configs/core_experiment_etth1_ettm1_formal.json`，作为 ETTh1/ETTm1 正式核心实验模板
5. ✅ 新增 `configs/ecl_smoke_optv2.json`，记录 ECL 高维 smoke test 配置
6. ✅ 使用 `configs/core_experiment_smoke.json` 跑通 ETTh1 h96 的 Informer/PatchTST 配置化 smoke test
7. ✅ 生成 `results/config_smoke_summary.csv` 与 `results/config_smoke_summary.md`

**修改的文件**：
- `scripts/run_experiments.py` - 新增 JSON 配置读取与默认值合并
- `configs/core_experiment_smoke.json` - 新增快速配置验证文件
- `configs/core_experiment_etth1_ettm1_formal.json` - 新增正式核心实验模板
- `configs/ecl_smoke_optv2.json` - 新增 ECL smoke 配置文件
- `README.md` - 补充配置文件运行示例
- `docs/progress.md` - 追加本次配置化记录

**测试结果**：
- ✅ `python -m py_compile scripts/run_experiments.py` 通过
- ✅ 三个 JSON 配置文件均可被 `python -m json.tool` 正常解析
- ✅ `python scripts/run_experiments.py --config configs/core_experiment_smoke.json` 跑通
- ✅ 配置化 smoke 结果保存为 `ETTh1_h96_informer_config_smoke_*` 与 `ETTh1_h96_patchtst_config_smoke_*`
- ✅ `results/config_smoke_summary.csv` 覆盖 2 个配置化 smoke 组合

**下一步任务**：
1. 根据 `configs/core_experiment_etth1_ettm1_formal.json` 运行 ETTh1/ETTm1 正式多 epoch 核心实验
2. 正式结果完成后，生成正式汇总表并与快速实验结果分开标记
3. 根据正式核心实验结果进入步骤 5 消融实验

---

## 24/48 预测步长补齐与五步长汇总 ✅

**完成时间**：2026/06/08 10:49

**完成内容**：
1. ✅ 修正实验覆盖范围，后续训练统一覆盖 `24/48/96/168/336`
2. ✅ 更新 `configs/core_experiment_etth1_ettm1_formal.json`，正式实验 horizon 改为 `24,48,96,168,336`
3. ✅ 更新 `configs/core_experiment_smoke.json`，用于验证短步长 h24 配置入口
4. ✅ 生成 ETTh1/ETTm1 的 `h24/h48` 预处理数据
5. ✅ 运行 ETTh1/ETTm1 的 `24/48 × 5 模型` 快速训练矩阵，标记为 `run_tag=quick5`
6. ✅ 为了口径一致，重新运行 ETTh1/ETTm1 的 `96/168/336 × 5 模型` 快速矩阵，同样标记为 `run_tag=quick5`
7. ✅ 生成完整五步长汇总：`results/quick5_all_horizons_summary.csv` 与 `results/quick5_all_horizons_summary.md`

**修改的文件**：
- `configs/core_experiment_etth1_ettm1_formal.json` - 正式实验补齐 24/48
- `configs/core_experiment_smoke.json` - smoke 配置切换到 h24
- `README.md` - 汇总与训练示例补齐五个 horizon
- `docs/progress.md` - 追加本次补齐记录

**生成的本地结果文件**：
- `data/processed/ETTh1/h24/{train,val,test}.npz`
- `data/processed/ETTh1/h48/{train,val,test}.npz`
- `data/processed/ETTm1/h24/{train,val,test}.npz`
- `data/processed/ETTm1/h48/{train,val,test}.npz`
- `results/ETTh1_h{24,48,96,168,336}_{lstm,transformer,informer,autoformer,patchtst}_quick5_results.npy`
- `results/ETTh1_h{24,48,96,168,336}_{lstm,transformer,informer,autoformer,patchtst}_quick5_summary.json`
- `results/ETTm1_h{24,48,96,168,336}_{lstm,transformer,informer,autoformer,patchtst}_quick5_results.npy`
- `results/ETTm1_h{24,48,96,168,336}_{lstm,transformer,informer,autoformer,patchtst}_quick5_summary.json`
- `results/quick5_all_horizons_summary.csv`
- `results/quick5_all_horizons_summary.md`

**测试结果**：
- ✅ ETTh1 h24 样本规模：train `(8521, 96, 7)`，val `(2761, 96, 7)`，test `(5781, 96, 7)`
- ✅ ETTh1 h48 样本规模：train `(8497, 96, 7)`，val `(2737, 96, 7)`，test `(5757, 96, 7)`
- ✅ ETTm1 h24 样本规模：train `(34441, 96, 7)`，val `(11401, 96, 7)`，test `(23481, 96, 7)`
- ✅ ETTm1 h48 样本规模：train `(34417, 96, 7)`，val `(11377, 96, 7)`，test `(23457, 96, 7)`
- ✅ `quick5_all_horizons_summary.csv` 覆盖 2 个数据集 × 5 个步长 × 5 个模型，共 50 个快速实验组合

**五步长 quick5 摘要（非正式最终指标）**：

| Dataset | Horizon | 最低测试 MSE 模型 | MSE | MAE | R² |
| --- | ---: | --- | ---: | ---: | ---: |
| ETTh1 | 24 | PatchTST | 1.018820 | 0.673929 | 0.127414 |
| ETTh1 | 48 | PatchTST | 1.035666 | 0.683290 | 0.088275 |
| ETTh1 | 96 | PatchTST | 1.094928 | 0.692672 | 0.071777 |
| ETTh1 | 168 | PatchTST | 1.208124 | 0.746631 | 0.005215 |
| ETTh1 | 336 | LSTM | 1.049870 | 0.709972 | -0.015179 |
| ETTm1 | 24 | LSTM | 1.443828 | 0.802862 | -0.026421 |
| ETTm1 | 48 | PatchTST | 1.549455 | 0.788071 | -0.050593 |
| ETTm1 | 96 | LSTM | 1.193375 | 0.722300 | -0.048144 |
| ETTm1 | 168 | LSTM | 1.138610 | 0.726439 | -0.040561 |
| ETTm1 | 336 | LSTM | 1.180346 | 0.746256 | -0.027229 |

**下一步任务**：
1. 后续正式训练必须覆盖 `24/48/96/168/336`
2. 运行五步长正式多 epoch 实验，并使用 `run_tag=formal_seed42` 与 quick5 结果区分
3. 正式结果完成后再进入步骤 5 消融实验

---

## results 分类目录整理 ✅

**完成时间**：2026/06/08

**完成内容**：
1. ✅ 新增 `scripts/organize_results.py`，将旧顶层结果文件整理成按步长分类视图
2. ✅ 生成按步长分类目录：`results/h{horizon}/{dataset}/{model}/{run_tag}/`
3. ✅ 按用户要求停用按模型分类视图，不再维护 `results/by_model`
4. ✅ 生成汇总表分类目录：`results/summaries/`
5. ✅ 生成结果索引文件：`results/RESULTS_INDEX.md`
6. ✅ 使用符号链接组织旧结果，原始顶层结果文件仍保留，避免重复占用磁盘
7. ✅ 新训练结果已改为生成时直接写入 `results/h{horizon}/{dataset}/{model}/{run_tag}/`

**修改的文件**：
- `scripts/organize_results.py` - 新增结果分类脚本
- `scripts/run_experiments.py` - 新结果生成时直接按步长分类落盘
- `scripts/summarize_results.py` - 递归读取分类结果，并跳过符号链接避免重复汇总
- `README.md` - 补充结果分类命令和目录结构
- `docs/progress.md` - 追加本次整理记录

**生成的本地结果目录**：
- `results/h24/`
- `results/h48/`
- `results/h96/`
- `results/h168/`
- `results/h336/`
- `results/summaries/`
- `results/RESULTS_INDEX.md`

**测试结果**：
- ✅ `python -m py_compile scripts/organize_results.py` 通过
- ✅ 已索引实验 summary：99 个
- ✅ `results/` 下包含 `h24/h48/h96/h168/h336`
- ✅ `results/by_model` 已移除
- ✅ `results/summaries` 下包含 17 个 CSV/Markdown 汇总文件链接
- ✅ 分类视图总计创建/更新 198 个旧结果文件链接
- ✅ 新生成的 `ETTh1 h24 LSTM formal_seed42` 结果直接保存到 `results/h24/ETTh1/lstm/formal_seed42/`

**下一步任务**：
1. 正式实验结果会自动进入 `results/h{horizon}/...`
2. 在正式汇总表中区分 `quick5`、`optv2`、`formal_seed42` 等 run tag
3. 继续执行五步长正式多 epoch 训练

---

## 训练续跑与跳过已有结果 ✅

**完成时间**：2026/06/08

**完成内容**：
1. ✅ 为 `scripts/run_experiments.py` 增加 `--skip-existing` 参数
2. ✅ 当目标 `_results.npy` 和 `_summary.json` 同时存在时，直接跳过对应实验组合
3. ✅ 三个配置文件均加入 `skip_existing: true`
4. ✅ 使用 `configs/core_experiment_smoke.json` 验证续跑：首次生成 h24 config smoke，第二次运行直接跳过
5. ✅ 重新运行 `scripts/organize_results.py --overwrite`，将新增结果同步到分类目录

**修改的文件**：
- `scripts/run_experiments.py` - 新增跳过已有结果逻辑
- `configs/core_experiment_smoke.json` - 启用 `skip_existing`
- `configs/core_experiment_etth1_ettm1_formal.json` - 启用 `skip_existing`
- `configs/ecl_smoke_optv2.json` - 启用 `skip_existing`
- `README.md` - 补充续跑命令示例
- `docs/progress.md` - 追加本次续跑机制记录

**测试结果**：
- ✅ `python -m py_compile scripts/run_experiments.py scripts/organize_results.py scripts/summarize_results.py` 通过
- ✅ 所有 `configs/*.json` 均可解析
- ✅ 重复运行 `python scripts/run_experiments.py --config configs/core_experiment_smoke.json` 时输出：
  - `跳过已存在结果: ETTh1_h24_informer_config_smoke`
  - `跳过已存在结果: ETTh1_h24_patchtst_config_smoke`
- ✅ 结果分类刷新后索引 summary 数量：98 个
- ✅ 分类视图结果链接数量：392 个
- ✅ 汇总表链接数量：15 个

**下一步任务**：
1. 运行 `configs/core_experiment_etth1_ettm1_formal.json` 开始正式五步长多 epoch 实验
2. 训练期间如中断，可直接重跑同一命令续跑
3. 正式结果完成后生成 `formal_seed42` 汇总表并进入消融实验

---

## ETTh1 h24 PatchTST 正式实验首块 ✅

**完成时间**：2026/06/08 10:56

**完成内容**：
1. ✅ 使用正式配置运行第一块多 epoch 实验：ETTh1 / h24 / PatchTST
2. ✅ 使用 `run_tag=formal_seed42`，与 quick5/config smoke 结果区分
3. ✅ 使用全量 ETTh1 h24 训练/验证/测试样本，`sample_limit=0`
4. ✅ 训练触发早停：上限 20 epoch，实际训练 11 epoch，最佳 epoch 为 6
5. ✅ 生成正式结果局部汇总：`results/formal_seed42_partial_summary.csv` 与 `.md`
6. ✅ 刷新结果分类目录，正式结果已进入按步长和按模型视图

**修改的文件**：
- `docs/progress.md` - 追加本次正式实验记录

**生成的本地结果文件**：
- `results/ETTh1_h24_patchtst_formal_seed42_results.npy`
- `results/ETTh1_h24_patchtst_formal_seed42_summary.json`
- `results/formal_seed42_partial_summary.csv`
- `results/formal_seed42_partial_summary.md`
- `results/by_horizon/h24/ETTh1/patchtst/formal_seed42/`
- `results/by_model/patchtst/ETTh1/h24/formal_seed42/`

**测试结果**：
- ✅ 训练样本：8521，验证样本：2761，测试样本：5781
- ✅ 训练设备：MPS
- ✅ 最佳验证损失：0.444811
- ✅ 测试 MSE：0.380213
- ✅ 测试 MAE：0.411152
- ✅ 测试 R²：0.702517
- ✅ 目标列 MSE：0.063568
- ✅ 目标列 MAE：0.193958
- ✅ 目标列 R²：0.589613
- ✅ `formal_seed42` 结果已通过 `organize_results.py` 分类

**下一步任务**：
1. 继续正式实验：ETTh1 h24 的 LSTM/Transformer/Informer/Autoformer
2. 再扩展到 ETTh1 的 h48/h96/h168/h336
3. ETTh1 完成后继续 ETTm1 的五步长正式实验

---

## ETTh1 h24 LSTM 正式实验与生成时分类 ✅

**完成时间**：2026/06/08

**完成内容**：
1. ✅ 按用户要求调整结果组织方式：不再需要按模型分类目录
2. ✅ `scripts/run_experiments.py` 改为结果生成时直接保存到 `results/h{horizon}/{dataset}/{model}/{run_tag}/`
3. ✅ `scripts/summarize_results.py` 改为递归扫描分类目录，并跳过符号链接避免重复统计
4. ✅ `scripts/organize_results.py` 改为只将旧顶层结果链接到 horizon-first 分类结构
5. ✅ 移除旧后处理视图目录 `results/by_model` 与 `results/by_horizon`
6. ✅ 使用新保存路径运行 ETTh1 / h24 / LSTM 正式实验
7. ✅ 更新 ETTh1 h24 formal 局部汇总，当前包含 LSTM 与 PatchTST 两个正式模型

**修改的文件**：
- `scripts/run_experiments.py` - 生成时按步长分类保存结果
- `scripts/summarize_results.py` - 递归汇总并跳过符号链接
- `scripts/organize_results.py` - 仅整理旧结果到 horizon-first 结构
- `README.md` - 更新结果目录说明
- `docs/progress.md` - 追加本次记录

**生成的本地结果文件**：
- `results/h24/ETTh1/lstm/formal_seed42/ETTh1_h24_lstm_formal_seed42_results.npy`
- `results/h24/ETTh1/lstm/formal_seed42/ETTh1_h24_lstm_formal_seed42_summary.json`
- `results/formal_seed42_etth1_h24_partial_summary.csv`
- `results/formal_seed42_etth1_h24_partial_summary.md`

**测试结果**：
- ✅ ETTh1 h24 LSTM 正式训练完成：上限 20 epoch，实际 11 epoch 后早停
- ✅ 最佳验证损失：0.879645
- ✅ 测试 MSE：0.870239
- ✅ 测试 MAE：0.664989
- ✅ 测试 R²：0.319114
- ✅ 新结果直接保存到 `results/h24/ETTh1/lstm/formal_seed42/`
- ✅ `results/by_model` 已移除
- ✅ 汇总表去重后只包含 LSTM 与 PatchTST 各一行

**下一步任务**：
1. 继续 ETTh1 h24 的 Transformer/Informer/Autoformer 正式实验
2. 每个正式结果都会直接落到 `results/h24/...`
3. ETTh1 h24 五模型完成后生成完整 h24 formal 汇总

---

## ETTh1 h24 五模型正式实验完成 ✅

**完成时间**：2026/06/08 15:01

**完成内容**：
1. ✅ 继续完成 ETTh1 / h24 / Transformer 正式实验
2. ✅ 继续完成 ETTh1 / h24 / Informer 正式实验
3. ✅ 继续完成 ETTh1 / h24 / Autoformer 正式实验
4. ✅ 结合已完成的 LSTM 与 PatchTST，ETTh1 h24 五模型正式结果已齐
5. ✅ 生成完整正式汇总：`results/formal_seed42_etth1_h24_summary.csv` 与 `.md`
6. ✅ 所有新结果均直接保存到 `results/h24/ETTh1/{model}/formal_seed42/`

**修改的文件**：
- `docs/progress.md` - 追加本次正式实验记录

**生成的本地结果文件**：
- `results/h24/ETTh1/transformer/formal_seed42/ETTh1_h24_transformer_formal_seed42_results.npy`
- `results/h24/ETTh1/transformer/formal_seed42/ETTh1_h24_transformer_formal_seed42_summary.json`
- `results/h24/ETTh1/informer/formal_seed42/ETTh1_h24_informer_formal_seed42_results.npy`
- `results/h24/ETTh1/informer/formal_seed42/ETTh1_h24_informer_formal_seed42_summary.json`
- `results/h24/ETTh1/autoformer/formal_seed42/ETTh1_h24_autoformer_formal_seed42_results.npy`
- `results/h24/ETTh1/autoformer/formal_seed42/ETTh1_h24_autoformer_formal_seed42_summary.json`
- `results/formal_seed42_etth1_h24_summary.csv`
- `results/formal_seed42_etth1_h24_summary.md`

**正式实验结果**：

| 模型 | MSE | MAE | R² | 最佳验证损失 | 训练轮数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PatchTST | 0.380213 | 0.411152 | 0.702517 | 0.444811 | 11 |
| Autoformer | 0.444251 | 0.446119 | 0.652413 | 0.389005 | 10 |
| Informer | 0.624358 | 0.549096 | 0.511494 | 0.526139 | 7 |
| Transformer | 0.743416 | 0.642608 | 0.418342 | 0.627699 | 8 |
| LSTM | 0.870239 | 0.664989 | 0.319114 | 0.879645 | 11 |

**测试结果**：
- ✅ ETTh1 h24 正式汇总覆盖 5 个模型
- ✅ 当前 ETTh1 h24 最低 MSE 模型：PatchTST
- ✅ 当前 ETTh1 h24 最高 R² 模型：PatchTST
- ✅ `results/h24/ETTh1/` 下五个模型均有 `formal_seed42` summary

**下一步任务**：
1. 继续 ETTh1 h48 五模型正式实验
2. 生成 `formal_seed42_etth1_h48_summary.csv/md`
3. 按顺序推进 ETTh1 h96/h168/h336

---

## results 目录临时结果清理 ✅

**完成时间**：2026/06/09 11:28

**完成内容**：
1. ✅ 清理 `results/` 中 quick、quick5、smoke、optv2、default 与 partial 等临时/过渡结果
2. ✅ 移除整理脚本生成的旧符号链接视图与空目录
3. ✅ 保留并重建 ETTh1 h24/h48 五模型 `formal_seed42` 正式结果
4. ✅ 重新生成 h24/h48 正式汇总表
5. ✅ 更新 `results/RESULTS_INDEX.md` 为当前保留范围

**修改的文件**：
- `results/RESULTS_INDEX.md` - 更新结果索引与保留范围说明
- `docs/progress.md` - 追加本次清理记录

**保留的本地结果范围**：
- `results/h24/ETTh1/{model}/formal_seed42/` - h24 五模型正式结果
- `results/h48/ETTh1/{model}/formal_seed42/` - h48 五模型正式结果
- `results/formal_seed42_etth1_h24_summary.csv`
- `results/formal_seed42_etth1_h24_summary.md`
- `results/formal_seed42_etth1_h48_summary.csv`
- `results/formal_seed42_etth1_h48_summary.md`

**测试结果**：
- ✅ `find results -type l` 确认符号链接数量为 0
- ✅ `find results -type d -empty` 确认空目录数量为 0
- ✅ `find results -type f` 确认当前保留 25 个结果相关文件
- ✅ h24 正式汇总覆盖 5 个模型
- ✅ h48 正式汇总覆盖 5 个模型

**下一步任务**：
1. 继续 ETTh1 h96/h168/h336 正式实验
2. 后续结果统一保存到 `results/h{horizon}/ETTh1/{model}/formal_seed42/`

---

## results Git 追踪同步 ✅

**完成时间**：2026/06/09 11:34

**完成内容**：
1. ✅ 确认 `.gitignore` 未再忽略 `results/`，结果目录可被 Git 正常追踪
2. ✅ 将清理后的正式 ETTh1 seed-42 结果范围同步为当前追踪对象：`h24/h48 × 5 个模型`
3. ✅ 保留聚合汇总文件：`formal_seed42_etth1_h24_summary.*`、`formal_seed42_etth1_h48_summary.*`
4. ✅ 为 `results/**/*.npy` 增加 `.gitignore` 例外，保证正式结果数组与 JSON 摘要一起被追踪
5. ✅ 更新 README 中的结果目录说明，明确正式结果纳入追踪、临时 quick/smoke/optv2 结果不保留

**修改的文件**：
- `.gitignore` - 允许追踪 `results/` 下的正式 `.npy` 结果
- `README.md` - 同步 results 追踪说明
- `results/` - 纳入正式结果与结果索引
- `docs/progress.md` - 追加本次同步记录

**测试结果**：
- ✅ `git check-ignore` 未命中 `results/RESULTS_INDEX.md` 和正式结果文件
- ✅ 正式 `.npy` 结果文件已进入 Git 暂存区
- ✅ `results/` 当前体积约 100K，适合提交到版本库

**下一步任务**：
1. 提交并推送当前 results 追踪同步变更
