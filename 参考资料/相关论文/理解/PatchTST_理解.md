# PatchTST: A Time Series is Worth 64 Words — Long-term Forecasting with Transformers

## 一、30 秒速览

* **一句话概括**：PatchTST 将时间序列分割成固定长度的 Patch（类似 ViT 对图像的处理），配合通道独立策略，让 Transformer 在长序列预测上达到了新的 state-of-the-art，同时参数量更少、训练更快。
* **研究问题**：现有 Transformer 时序预测方法（Informer、Autoformer 等）将每个时间步作为一个 token，导致：(1) 序列过长时注意力计算昂贵；(2) 相邻时间步高度冗余，单步 token 缺乏局部语义；(3) 多变量间强行建立交叉注意力实际上带来了噪声而非信息增益。
* **核心方法**：(1) **Patching**：将连续时间步打成分组（Patch），每个 Patch 作为一个 token，大幅缩短序列长度并丰富局部语义；(2) **Channel Independence**：每个变量独立建模，不做跨变量注意力。
* **主要结果**：在 8 个数据集上全面超越 Informer、Autoformer、FEDformer 等 Transformer 变体，在多数设置下也超越 DLinear；MSE 平均降低 20-30%；同时参数量仅为 Autoformer 的约 1/3。
* **最大创新**：将 ViT 的 Patch 思想引入时序预测，配合通道独立，从"逐点建模"转向"局部块建模"，这是范式级别的思路转换。
* **最大局限**：通道独立假设各变量之间无交互，在变量强耦合的数据上可能损失信息；Patch 大小需要手动设定，对不同频率的数据需要调参；未充分利用多变量间的交叉信息。
* **阅读建议**：⭐⭐⭐⭐⭐ 强烈推荐。这是 2023 年 ICLR 的重要工作，Patching + Channel Independence 的组合简单而有效，对后续研究影响深远（TSMixer、iTransformer 等都从中借鉴）。

## 二、论文基本信息

* **作者**：Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam
* **机构**：IBM Research
* **发表时间**：2022 年 11 月（arXiv），2023 年 5 月（ICLR 2023）
* **会议或期刊**：ICLR 2023（顶级会议）
* **论文链接**：https://arxiv.org/abs/2211.14730
* **官方代码**：https://github.com/yuqie98/PatchTST （约 2.6k Stars）
* **项目主页**：无独立项目主页
* **引用数**：约 544+（OpenAlex 统计，2026年初估计已被引用更多）

## 三、研究背景

### 3.1 论文研究什么

长序列时序预测（LSTF）——与 Informer、Autoformer 相同的问题设定，但 PatchTST 从一个完全不同的角度审视问题：不是去改进注意力机制（ProbSparse、Auto-Correlation），而是改进 Transformer 的**输入表示**和**建模策略**。

### 3.2 为什么这个问题重要

- 前代 Transformer 变体（Informer、Autoformer、FEDformer）都专注于改进注意力机制本身，但忽视了输入表示的问题——逐点作为 token 可能不是最优的。
- DLinear 等简单线性模型竟然在部分数据集上超越复杂 Transformer，说明 Transformer 在时序预测上的潜力尚未被充分挖掘。
- 通道维度的建模策略（多变量交互 vs 独立建模）鲜少被系统研究。

### 3.3 现有方法有什么不足

| 方法类型 | 问题 |
|---------|------|
| **Informer** | 逐点 token + 稀疏注意力：单步 token 缺乏局部语义，ProbSparse 丢弃部分查询可能丢失有效信息 |
| **Autoformer** | 逐点 token + 分解 + 自相关：分解思想好但输入表示仍为逐点，相邻点冗余 |
| **FEDformer** | 频域注意力：有效但增加了架构复杂度 |
| **DLinear** | 简单线性模型即可超越 Transformer，说明复杂注意力可能是过度设计 |

**核心洞察**：问题不在于注意力机制不好，而在于 Transformer 处理时序的**输入粒度**不对——逐点 token 让模型看到了太多冗余的"像素级"细节，缺少"语义级"的局部模式。

## 四、核心方法

### 4.1 整体思路

PatchTST 的核心思想来自 Vision Transformer (ViT)：图像中把 16×16 的 patch 作为 token，而不是单个像素。类比到时间序列：把连续 P 个时间步作为一个 Patch 作为 token，而不是单个时间步。

加上**通道独立**策略（每个变量独立通过同一个 Transformer），构成了 PatchTST 的两大支柱。

### 4.2 模型输入与输出

- **输入**：多元时间序列 X ∈ ℝ^{B×C×L}（Batch×Channels×Length），每个变量独立处理为 xᵢ ∈ ℝ^{B×L}
- **Patch 分割后**：每个变量序列被分割为 N = ⌊(L - P)/S⌋ + 1 个 Patch，每个 Patch 长度 P，步长 S
- **输出**：预测序列 Ŷ ∈ ℝ^{B×H×C}（H 个预测步长，C 个变量分别预测后拼接）

### 4.3 模型结构

```
多元时间序列 X ∈ ℝ^{B×C×L}
    │
    ├── 对每个变量 c 独立处理 ────────────────────────────┐
    │                                                      │
    │   单变量序列 x_c ∈ ℝ^{L}                              │
    │       │                                               │
    │       ▼                                               │
    │   ┌──────────────────────┐                           │
    │   │  Patching             │                           │
    │   │  将 L 个时间步分割为    │                           │
    │   │  N 个 Patch           │                           │
    │   │  (每个 Patch P 个步长) │                           │
    │   └──────────────────────┘                           │
    │       │ N 个 Patch (token)                             │
    │       ▼                                               │
    │   ┌──────────────────────┐                           │
    │   │  Linear Projection   │                           │
    │   │  将每个 Patch 投影到   │                           │
    │   │  d_model 维空间       │                           │
    │   └──────────────────────┘                           │
    │       │                                               │
    │       ▼                                               │
    │   ┌──────────────────────┐                           │
    │   │  + 可学习位置编码      │                           │
    │   └──────────────────────┘                           │
    │       │                                               │
    │       ▼                                               │
    │   ┌──────────────────────┐                           │
    │   │  Transformer Encoder │                           │
    │   │  (标准自注意力)        │                           │
    │   │  ×L 层               │                           │
    │   └──────────────────────┘                           │
    │       │                                               │
    │       ▼                                               │
    │   ┌──────────────────────┐                           │
    │   │  Linear Head          │                           │
    │   │  输出 H 个预测步长     │                           │
    │   └──────────────────────┘                           │
    │       │                                               │
    │       ▼                                               │
    │   Ŷ_c ∈ ℝ^{H} (变量 c 的预测)                         │
    │                                                      │
    ◄──────────────────────────────────────────────────────┘
    │
    ▼
  拼合所有变量 → Ŷ ∈ ℝ^{B×H×C}
```

### 4.4 核心创新模块

#### 4.4.1 Patching

**直觉**：单个时间步就像图像的单个像素——信息量太少、相邻点高度冗余。把连续 P 个时间步打包成一个 Patch，就像 ViT 把 16×16 像素打包成一个 token，让每个 token 包含局部语义信息。

**实现关键**：
- Patch 长度 P：每个 Patch 包含 P 个时间步
- 步长 S：Patch 移动的步幅。S < P 时 Patch 之间有重叠（类似卷积的感受野重叠）
- Patch 数量 N = ⌊(L - P)/S⌋ + 1（若使用填充可能有微调）

**效果**：
- 序列长度从 L 缩短为 N ≈ L/P（或 L/S），注意力计算量从 O(L²) 降至 O(N²) ≈ O((L/P)²)。
- 每个 Patch 包含 P 步的局部模式，比单步 token 有更丰富的语义。
- 重叠 Patch（S < P）进一步增强局部连续性。

#### 4.4.2 Channel Independence

**直觉**：多变量间强行建立交叉注意力（如标准 Transformer 的多变量注意力），往往会引入噪声——因为不同变量的模式差异很大（如温度和湿度），强行让它们互相注意反而干扰学习。

**实现**：每个变量独立通过同一个 Transformer Encoder（权重共享），不做任何跨变量交互。预测时各变量分别输出，最后拼合。

**效果**：
- 注意力序列长度从 C×L 降至 L（每个变量独立处理），大幅降低计算量。
- 避免了跨变量噪声。
- 权重共享确保模型从所有变量中学习通用模式。

**关键发现**：论文通过实验证明，通道独立不仅在预测精度上优于通道混合，而且参数量更少、训练更快。

#### 4.4.3 仅编码器架构

与 Informer/Autoformer 不同，PatchTST 只使用 Transformer Encoder，不需要 Decoder：
- 编码器输出 N 个 token 的表示。
- 将 N 个 token 的表示 Flatten 或 Mean Pooling 后，通过一个线性层直接映射到 H 个预测步长。
- 这种设计更简单，且避免了 Decoder 的误差累积问题。

### 4.5 完整数据流程

1. 输入多元时间序列 X ∈ ℝ^{B×C×L}，分解为 C 个单变量序列。
2. 每个单变量序列 x_c ∈ ℝ^{L} 经过 Patching：分割为 N 个 Patch，每个 Patch 长度 P。
3. 每个 Patch 通过线性投影映射到 d<sub>model</sub> 维空间，并加上可学习位置编码。
4. 投影后的 N 个 token 送入 L 层标准 Transformer Encoder（Multi-Head Self-Attention + FFN）。
5. 编码器输出 N 个 token 表示，Flatten 后通过线性层映射到 H 个预测值。
6. C 个变量的预测结果拼合为最终输出 Ŷ ∈ ℝ^{B×H×C}。

## 五、关键公式

### 公式 1：Patch 分割

$$\mathbf{X}_p^{(c)} = [\mathbf{x}_1^{(c)}, \mathbf{x}_2^{(c)}, \ldots, \mathbf{x}_N^{(c)}]$$

其中 xᵢ^{(c)} ∈ ℝ^{P} 是第 i 个 Patch，包含第 (i-1)×S+1 到 (i-1)×S+P 个时间步。

* **公式作用**：将长度为 L 的单变量序列分割为 N 个长度为 P 的 Patch。
* **直觉理解**：就像把一段长文本分成 N 个短语，每个短语包含 P 个字。每个短语比单个字包含更多语义信息。
* **符号含义**：
  - C：变量数
  - L：输入序列长度
  - P：Patch 长度（每个 Patch 包含的时间步数）
  - S：步长（相邻 Patch 之间的间隔，S ≤ P，S < P 时有重叠）
  - N：Patch 数量，N ≈ (L - P) / S + 1
* **设计原因**：逐点 token 让注意力看到的是"雪花"而非"模式"；Patch 让每个 token 包含 P 步的局部模式，大幅减少序列长度的同时丰富每个 token 的语义。

### 公式 2：线性投影与位置编码

$$\mathbf{Z}_0^{(c)} = [\mathbf{x}_1^{(c)} \mathbf{W}_p + \mathbf{b}_p, \mathbf{x}_2^{(c)} \mathbf{W}_p + \mathbf{b}_p, \ldots, \mathbf{x}_N^{(c)} \mathbf{W}_p + \mathbf{b}_p] + \mathbf{P}_{pos}$$

* **公式作用**：将每个 Patch 通过线性投影映射到 d<sub>model</sub> 维空间，并加上可学习的位置编码。
* **直觉理解**：与 ViT 和 NLP 中的做法完全相同——线性投影将局部窗口映射到高维空间，位置编码告诉模型每个 Patch 在时间轴上的位置。
* **符号含义**：
  - W<sub>p</sub> ∈ ℝ^{P × d<sub>model</sub>}：线性投影矩阵
  - b<sub>p</sub> ∈ ℝ^{d<sub>model</sub>}：偏置
  - P<sub>pos</sub>：可学习位置编码，维度 (N, d<sub>model</sub>)
* **设计原因**：可学习的位置编码比固定正弦编码更灵活，能适应不同数据的周期性模式。

### 公式 3：通道独立的 Transformer 编码

$$\hat{\mathbf{y}}^{(c)} = \text{Linear}(\text{Flatten}(\text{TransformerEncoder}(\mathbf{Z}_0^{(c)})))$$

* **公式作用**：对每个变量 c 独立地通过 Transformer Encoder + 线性头得到预测结果。
* **直觉理解**：每个变量的序列独立走过"Patch → 投影 → Transformer → 线性头"的完整流程，变量之间没有任何信息交换。
* **设计原因**：论文通过实验证明，通道独立（CI）比通道混合（CD）效果更好。原因是：(1) 不同变量的模式差异大，混合注意力引入噪声；(2) CI 让注意力只需关注单变量的时间模式，更简洁高效；(3) 参数共享从所有变量中学习通用时序模式。

## 六、实验设计

* **数据集**：
  - ETTh1/ETTh2（小时级，7 变量）
  - ETTm1/ETTm2（15 分钟级，7 变量）
  - Weather（小时级，21 变量）
  - ECL（小时级，321 变量）
  - Traffic（小时级，862 变量）
  - Exchange（日级，8 变量）
  - Illness（周级，7 变量）
- **任务**：多变量预测，预测步长 24-720（根据数据集不同）
- **基线方法**：Informer、Autoformer、FEDformer、LogTrans、Reformer、Pyraformer、DLinear、Scaleformer 等
- **评价指标**：MSE、MAE
- **实现细节**：
  - 优化器：AdamW
  - 学习率：论文未明确指定具体值，代码默认使用学习率调度
  - Batch Size：论文未明确说明
  - 训练 Epoch：论文未明确说明（代码默认使用 early stopping）
  - Patch 长度 P：16（ETT 数据集）、24（Weather）、32（ECL/Traffic）
  - 步长 S：通常设为 P/2（即 50% 重叠）
  - 编码器层数：3
  - 注意力头数：4-8
  - d_model：16-128（因数据集而异，Patch 设计使得 d_model 可以更小）
  - Dropout：0.1
- **硬件环境**：论文未明确说明

## 七、实验结果

### 7.1 主结果

**PatchTST vs 主要基线（MSE 相对提升）**：

| 数据集 | 预测步长 | vs Informer | vs Autoformer | vs FEDformer | vs DLinear |
|--------|---------|------------|--------------|-------------|-----------|
| ETTh1 | 720 | ~-35% | ~-25% | ~-20% | ~-15% |
| ETTm1 | 720 | ~-30% | ~-20% | ~-15% | ~-10% |
| Weather | 720 | ~-25% | ~-15% | ~-10% | 数据相当 |
| ECL | 336 | ~-20% | ~-10% | ~-5% | ~-5% |
| Traffic | 336 | ~-15% | ~-5% | 数据相当 | 数据相当 |

> 注：以上为论文表格近似值，原文以具体数值为准。

**关键发现**：
- PatchTST 在 8 个数据集上全面超越所有 Transformer 基线。
- 与 DLinear（简单线性模型）的比较中，PatchTST 在大多数设置上也占优，但在 Weather 短步长上两者差距很小。
- 通道独立策略在所有数据集上都优于通道混合。
- Patch 设计允许使用更小的 d_model（16-128 vs 传统 512），参数量仅约 Autoformer 的 1/3。

### 7.2 消融实验

| 变体 | 说明 | 效果 |
|------|------|------|
| 完整 PatchTST（Patch + CI） | — | ✅ 最优 |
| 去掉 Patch（逐点 token） | Patch → 逐点 | 性能显著下降 + 计算量增大 |
| 去掉 CI（改用通道混合） | CI → CD | 性能下降（尤其在多变量数据上） |
| Patch + CD | 保留 Patch 但通道混合 | 比完整 PatchTST 差，但比逐点+CI 好 |
| 不同 Patch 大小 P | P=8, 16, 32, 64 | P=16 左右通常最优（数据集相关） |
| 不同步长 S | S=P（无重叠）vs S=P/2（重叠） | 重叠 (S<P) 通常更好 |

**关键结论**：
- Patch 和 CI 各自独立有贡献，组合后效果最好。
- Patch 大小存在甜点值：太小（如 P=4）语义不够丰富，太大（如 P=128）丢失过多细节。
- 重叠 Patch 比无重叠 Patch 略好，但重叠对计算量影响不大。

### 7.3 可视化结果

论文提供了注意力图可视化：
- PatchTST 的注意力图呈现明显的周期性块状模式（整个 Patch 作为一个单元被注意），对比逐点注意力的碎片化模式。
- 不同层的注意力关注不同尺度的模式：浅层关注局部周期，深层关注全局趋势。

### 7.4 自监督学习结果

论文还探索了自监督预训练：
- 使用 Masked Patch Reconstruction（类似 MAE）对 PatchTST 进行预训练。
- 预训练后在下游预测任务上 finetune，在标注数据少时（如 5% 训练数据）提升显著（MSE 降低 10-20%）。

### 7.5 结果是否支持作者结论

**支持的部分**：
- Patch + CI 组合在所有数据集上全面达到 SOTA，说服力强。
- 消融实验清晰展示了 Patch 和 CI 各自的独立贡献。
- 与 DLinear 的比较是诚实的——PatchTST 不是在所有设置上都大幅领先，但在大多数长步长设置上确实更好。

**需要警惕的部分**：
- 通道独立在强耦合变量数据上可能不是最优解。
- 自监督部分是额外贡献，但与主实验的预测任务存在不同的评估标准。

## 八、审稿人视角

* **创新性**：⭐⭐⭐⭐ 高。Patching 思想虽然来自 ViT，但将其有效应用于时序预测并配合通道独立的设计是重要的方法论贡献。单独看 Patch 或 CI 都不算完全原创，但组合后的效果和理论分析有新意。
* **实验充分性**：⭐⭐⭐⭐⭐ 非常充分。8 个数据集 + 多种步长 + 详细消融 + 自监督实验 + 可视化分析，覆盖面和深度都很好。
* **对比公平性**：⭐⭐⭐⭐ 较好。基线包含了当时所有主流方法，包括 DLinear。但部分基线的超参数可能采用原始论文设置，未充分调优。
* **统计可靠性**：⭐⭐⭐⭐ 较好。报告了多次运行均值和标准差。
* **潜在问题**：
  - 通道独立假设变量间无交互，这在某些强耦合数据上可能次优（如温度和露点温度）。
  - Patch 大小 P 需要手动选择，不同频率/周期的数据最优 P 不同，论文未提供自动选择机制。
  - 在某些短步长设置上与 DLinear 差距很小（甚至偶尔略差），说明简单模型的竞争力。
* **论文局限**：
  - 仅验证了编码器架构（无解码器），是否适用于需要更精细输出控制的任务（如概率预测）有待探索。
  - 自监督部分与主实验的关联可以更紧密。
  - 未讨论与多变量联合建模方法的互补可能（如 patch + cross-channel attention）。
* **总体评价**：⭐⭐⭐⭐⭐ 优秀工作。简单、有效、有理论支撑。Patching + CI 的设计直接挑战了前代 Transformer 变体的核心假设（复杂注意力 + 通道混合），实验结果扎实。

## 九、复现指南

### 9.1 官方资源

- **代码仓库**：https://github.com/yuqie98/PatchTST （约 2.6k Stars，440 Forks）
- **框架**：PyTorch
- **许可证**：论文未明确说明

### 9.2 环境与依赖

- Python 3.8+
- PyTorch 1.12+
- 其他依赖：einops（用于张量操作）、pandas, numpy, scipy, matplotlib
- 代码仓库包含 `requirements.txt`

### 9.3 数据准备

- 数据集可从 HuggingFace 或论文提供的链接下载。
- 将数据放入 `datasets/` 目录，格式与 Informer/Autoformer 仓库兼容。

### 9.4 训练步骤

```bash
# 示例：在 ETTh1 上训练，预测步长 336
python run.py --model PatchTST \
  --data ETTh1 \
  --pred_len 336 \
  --seq_len 96 \
  --patch_len 16 \
  --stride 8 \
  --d_model 128 \
  --n_heads 4 \
  --e_layers 3 \
  --channel_independent True
```

### 9.5 评价步骤

- 训练后自动在测试集上评估 MSE 和 MAE。
- 支持多种子运行取均值。

### 9.6 关键超参数

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| seq_len | 96（ETT）/ 336（Weather 等） | 输入序列长度 |
| pred_len | 24-720 | 预测步长 |
| patch_len | 16（ETT）/ 24-32（其他） | Patch 长度 |
| stride | patch_len / 2 | Patch 步长（通常为 P 的一半） |
| d_model | 16-128 | 模型维度（因数据集而异） |
| n_heads | 4-8 | 注意力头数 |
| e_layers | 3 | 编码器层数 |
| d_ff | 128-512 | 前馈层维度 |
| dropout | 0.1 | Dropout 率 |
| channel_independent | True | 是否通道独立（默认 True） |
| revin | True | 是否使用 Reversible Instance Normalization |

### 9.7 论文未说明的信息

- ❌ 精确的学习率值（代码使用学习率调度器）
- ❌ Batch Size 具体值
- ❌ 训练总 Epoch 数
- ❌ 学习率调度策略细节
- ❌ 权重初始化方法
- ❌ 精确的训练/验证/测试切分比例
- ❌ Patch 大小的自动选择策略
- ❌ 是否使用了权重衰减

### 9.8 可能踩坑点

1. **Patch 大小选择**：P 是关键超参数。建议从 P=16 开始，对于 15 分钟数据尝试 P=32-64，对于日数据尝试 P=8-16。步长 S 通常设为 P/2。
2. **通道独立与数据集的关系**：在变量数少（如 ETT 7 变量）的数据集上效果提升有限，在变量数多（如 ECL 321 变量）的数据集上提升显著。
3. **Reversible Instance Normalization（RevIN）**：代码中默认使用 RevIN（论文中提到但未重点强调）。不使用 RevIN 性能会明显下降。
4. **序列长度与 Patch 大小的匹配**：seq_len 应该能被 patch_len 整除（或加上填充后整除），否则可能产生边界问题。
5. **d_model 可以很小**：因为 Patch 将 P 步压缩到一个 token，d_model 不需要像传统 Transformer 那么大（512），16-128 通常足够。
6. **内存效率**：通道独立意味着 C 个序列独立通过模型，总计算量为 C × 单变量计算量。对于高维数据（如 Traffic 862 变量），训练时间会较长但不会 OOM。

### 9.9 最小复现方案

1. 克隆仓库 `https://github.com/yuqie98/PatchTST`
2. 下载 ETTh1 数据集，放入 `datasets/` 目录
3. 安装 PyTorch 和基础依赖（`pip install -r requirements.txt`）
4. 运行：
   ```bash
   python run.py --model PatchTST --data ETTh1 --pred_len 336 \
     --seq_len 96 --patch_len 16 --stride 8
   ```
5. 观察输出的 MSE/MAE，与论文 Table 1 对比

### 9.10 复现成功标准

- 在 ETTh1 上 pred_len=336 时 MSE 达到论文报告值的 ±10% 范围内
- 在 Weather 上 pred_len=720 时 MSE 达到论文报告值的 ±10% 范围内
- 通道独立版本应优于通道混合版本
- Patch 版本应优于逐点 token 版本

## 十、最终总结

* **这篇论文真正的新意是什么**：
  1. **Patching for Time Series**：将 ViT 的 Patch 思想引入时序预测，解决了"逐点 token 缺乏局部语义"的根本问题。这不是简单的工程技巧，而是对 Transformer 处理时序数据的输入粒度的重新思考。
  2. **Channel Independence**：用实验证明"不建跨变量注意力反而更好"，挑战了多变量 Transformer 的设计直觉。

* **最值得学习的部分**：
  1. Patch 的理论分析：为什么 Patching 能同时提升精度和效率（减少序列长度 + 丰富语义 + 减少噪声）。
  2. 通道独立的消融实验设计：清晰展示了 CI vs CD 在不同维度数据上的对比。
  3. 整体方法论："先确认问题在哪里（逐点 token + 通道混合），再提出简洁的解决方案（Patch + CI）"——简洁而有效的思路。

* **最值得怀疑的部分**：
  1. 通道独立在强物理耦合变量上是否真的更好？一些后续工作（如 iTransformer）表明，对变量嵌入而非时间序列嵌入可能更优。
  2. Patch 大小的选择缺乏自动机制，依赖人工调参。
  3. 在短步长预测（如 24 步）上，PatchTST 相对 DLinear 的优势很小，甚至偶尔逊于简单线性模型。

* **适合哪些人阅读**：
  - 时序预测方向的研究生（理解为什么 Patching 有效是基本素养）
  - Transformer 架构设计的研究者（Patching 是 Transformer 应用于时序的关键技巧）
  - 需要在实际项目中使用时序预测的工程师（PatchTST 架构简洁，易于实现和调试）

* **是否值得精读**：⭐⭐⭐⭐⭐ 强烈推荐。这篇论文的 Patching + CI 思想已被广泛采纳，理解它对理解整个领域的发展方向至关重要。

* **推荐继续阅读**：
  1. DLinear（"Transformers are vulnerable to simple linear models"的挑战性工作）
  2. iTransformer（反转维度：对变量而非时间做注意力，PatchTST 的"对立面"探索）
  3. TimesNet（多周期建模思路，与 PatchTST 的 Patch 思想互补）
  4. ViT 原始论文（理解 Patch 思想的来源）