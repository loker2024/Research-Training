# 基于 LSTM 与 Transformer 变体的长时序预测实验研究

作者：待补充  
单位：待补充

## 摘要

长时序预测需要在较长历史窗口内识别趋势、周期和局部波动，并在多步预测中控制误差累积。针对这一问题，本文基于统一的数据预处理、训练和评估框架，对 LSTM、标准 Transformer、Informer、Autoformer 与 PatchTST 五类模型进行对比实验。实验使用 ETTh1 与 ETTm1 两个电力变压器温度数据集，统一采用 96 步回看窗口，并设置 24、48、96、168 和 336 五个预测步长。核心实验共完成 50 组正式训练，评价指标包括 MSE、MAE 和 R2；同时围绕 Autoformer 的序列分解与自相关机制、PatchTST 的 patching 与通道独立建模设计 16 组消融实验。结果表明，PatchTST 在 ETTh1 的五个预测步长上均取得最低 MSE，Autoformer 在 ETTm1 的五个预测步长上均取得最低 MSE。跨数据集和步长平均后，PatchTST 的 MSE 最低，为 0.469335，Autoformer 次之，为 0.514711。消融结果进一步显示，移除 PatchTST 的通道独立建模会使平均 MSE 上升 162.25%，移除 Autoformer 的序列分解模块会使平均 MSE 上升 81.48%。这些结果说明，面向长时序预测的结构设计需要同时关注局部片段建模、变量间干扰控制以及趋势和周期分解。本文的结论限定于当前轻量实现、单随机种子和 ETTh1/ETTm1 主实验范围，ECL 高维数据仍需在后续实验中补充验证。

**关键词**：长时序预测；LSTM；Transformer；Informer；Autoformer；PatchTST；消融实验

## 1 引言

时间序列预测广泛存在于电力负荷、交通流量、工业监测和气象分析等任务中。与短期预测相比，长时序预测需要模型在更长的预测窗口内保持对趋势变化、周期结构和局部扰动的识别能力。随着预测步长增加，模型误差往往会逐步累积，预测曲线也更容易出现相位偏移、过度平滑或远期细节丢失。因此，长时序预测不仅要求模型具备序列建模能力，还要求模型能够在计算效率和长期依赖建模之间取得平衡。

传统循环神经网络能够按时间顺序处理历史观测，其中 LSTM 通过门控结构缓解普通 RNN 的梯度衰减问题。然而，当预测窗口较长时，循环结构的并行效率和长距离信息保留能力仍受到限制。Transformer 通过自注意力机制直接建模任意时间步之间的依赖关系，提升了并行能力和全局依赖表达能力，但标准自注意力的计算复杂度随序列长度二次增长，直接用于长时序预测时会带来计算开销和泛化退化问题。

为缓解上述问题，近年来出现了多种面向长时序预测的 Transformer 变体。Informer 通过 ProbSparse Attention 和序列蒸馏降低长序列注意力计算成本；Autoformer 将时间序列分解为趋势项和季节项，并用 Auto-Correlation 捕捉周期依赖；PatchTST 则将连续时间点切分为 patch，并采用通道独立建模减少多变量间的干扰。已有研究表明，这些结构能够在不同数据集上改善长期预测表现，但在一个统一轻量实验框架下，它们相对于 LSTM 和标准 Transformer 的差异、误差随步长变化的规律以及关键模块的贡献仍需要结合项目实验进行验证。

本文围绕以下问题展开研究：第一，在统一训练设置下，LSTM、Transformer 及三种高效 Transformer 变体在长时序预测中表现如何；第二，预测步长从 24 增加到 336 时，不同模型的误差如何变化；第三，Autoformer 的序列分解和 Auto-Correlation、PatchTST 的 patching 和通道独立建模是否真正贡献了性能提升；第四，模型预测精度、参数量和训练耗时之间是否存在可观察的权衡关系。

本文的贡献主要包括三点。首先，完成 ETTh1 与 ETTm1 上 2 个数据集、5 个预测步长、5 个模型的统一对比实验，形成 50 组正式结果。其次，围绕 Autoformer 和 PatchTST 的核心结构完成 16 组消融实验，定量分析关键模块对预测性能的影响。最后，结合指标趋势、复杂度图、预测曲线和残差图，对长期预测中的误差累积、季节波动拟合和模型适用边界进行讨论。

## 2 相关工作

### 2.1 循环神经网络与标准 Transformer

LSTM 是长短期记忆网络的经典结构，通过输入门、遗忘门和输出门控制历史状态的保留与更新。在时间序列预测任务中，LSTM 常被用作稳定基线，因为它能够显式处理序列顺序，并在中短期依赖建模中具有较好的可解释性。然而，LSTM 的递归计算使其并行效率受限，且在较长预测步长下容易出现误差累积。

Transformer 以自注意力机制替代递归结构，使模型能够并行处理序列并直接捕捉长距离依赖。对于长度为 L 的序列，标准自注意力需要计算 L 乘 L 的注意力矩阵，因此当输入序列较长或模型层数增加时，计算和内存成本都会显著上升。在本项目中，标准 Transformer 作为强基线，用于衡量直接使用自注意力结构进行长时序预测的表现。

### 2.2 长时序预测 Transformer 变体

Informer 针对长序列预测中的注意力开销提出 ProbSparse Attention。其核心思想是只对少数重要 query 计算完整注意力，其余位置使用近似表示，从而将注意力计算从二次复杂度降低到近似 O(L log L)。Informer 还引入 self-attention distilling 压缩中间序列长度，并使用一次性生成式解码器输出完整预测窗口。

Autoformer 从时间序列的趋势性和周期性出发，引入序列分解模块，将输入拆分为趋势项和季节项，再通过 Auto-Correlation 机制寻找周期延迟并聚合相似子序列。与逐点注意力相比，Auto-Correlation 更强调序列级周期结构，因此适合存在明显周期或季节模式的数据。

PatchTST 借鉴视觉 Transformer 中 patch 的思想，将连续时间点切分为局部片段后再输入 Transformer。这样既减少 token 数量，也提高每个 token 的局部语义密度。同时，PatchTST 采用通道独立策略，即对每个变量分别建模并共享参数，避免不同变量分布差异导致的干扰。本文的消融实验重点检验 patching 和通道独立建模在当前数据集上的贡献。

## 3 方法

### 3.1 任务定义

设多变量时间序列为 \(X = \{x_1, x_2, ..., x_T\}\)，其中每个时间步 \(x_t \in \mathbb{R}^{C}\)，C 为变量数。给定长度为 L 的历史窗口 \(X_{t-L+1:t}\)，模型需要预测未来 H 个时间步：

\[
\hat{Y}_{t+1:t+H} = f(X_{t-L+1:t})
\]

本文统一设置回看窗口 \(L=96\)，预测步长 \(H \in \{24, 48, 96, 168, 336\}\)。所有模型输出形状统一为 `(batch, horizon, features)`，从而保证训练、评估和可视化流程一致。

### 3.2 模型设置

本文比较五类模型：

| 模型 | 主要机制 | 在本文中的角色 |
| --- | --- | --- |
| LSTM | 门控循环结构 | RNN 基线 |
| Transformer | 标准多头自注意力 | 标准注意力基线 |
| Informer | ProbSparse Attention 与序列蒸馏 | 高效注意力变体 |
| Autoformer | 序列分解与 Auto-Correlation | 趋势和周期建模变体 |
| PatchTST | Patching 与通道独立建模 | 局部片段和变量独立变体 |

所有模型均使用项目中的统一训练框架。LSTM 和 Transformer 用于提供基础对照，Informer、Autoformer 和 PatchTST 用于验证面向长时序预测的结构改进是否带来实际收益。

### 3.3 消融设计

为分析关键模块贡献，本文围绕 Autoformer 和 PatchTST 设计四个消融变体：

| 消融模型 | 对应原模型 | 被检验模块 | 消融方式 |
| --- | --- | --- | --- |
| `autoformer_no_decomp` | Autoformer | Series Decomposition | 关闭分解，趋势分支置零 |
| `autoformer_no_autocorr` | Autoformer | Auto-Correlation | 替换为标准多头自注意力 |
| `patchtst_no_patch` | PatchTST | Patching | 使用逐时间点线性投影替代 patch embedding |
| `patchtst_channel_mix` | PatchTST | Channel Independence | 混合多变量输入，取消通道独立建模 |

消融实验覆盖 ETTh1、ETTm1 两个数据集，以及 h96 和 h336 两个代表性预测步长，共 16 组正式实验。

## 4 实验设置

### 4.1 数据集

实验使用 ETTh1 和 ETTm1 两个公开电力变压器温度数据集作为主结果来源。项目中保留了 ECL 数据集，但当前完整正式结果矩阵主要覆盖 ETTh1 和 ETTm1；ECL 高维数据适合作为后续附录或扩展实验。

| 数据集 | 变量数 | 频率 | 切分方式 | 说明 |
| --- | ---: | --- | --- | --- |
| ETTh1 | 7 | 小时 | 前 12 月 / 4 月 / 4 月 | 小时级电力变压器温度数据 |
| ETTm1 | 7 | 15 分钟 | 前 12 月 / 4 月 / 4 月 | 15 分钟级电力变压器温度数据 |

数据按时间顺序划分训练集、验证集和测试集，避免未来信息泄露。标准化参数仅由训练集统计量得到，并应用于验证集和测试集。

### 4.2 训练与评价

正式核心实验配置如下：

| 配置项 | 取值 |
| --- | --- |
| 回看窗口 | 96 |
| 预测步长 | 24、48、96、168、336 |
| 训练轮数上限 | 20 |
| early stopping patience | 5 |
| batch size | 32 |
| learning rate | 0.001 |
| weight decay | 0.00001 |
| 随机种子 | 42 |
| 正式 run tag | `formal_seed42` |
| 消融 run tag | `ablation_seed42` |
| 评价指标 | MSE、MAE、R2 |

实验结果来自 `results/formal_seed42_all.csv` 和 `results/ablation_seed42_vs_formal_comparison.csv`。预测曲线和残差图由已有 checkpoint 对测试集首批样本推理得到，不重新训练模型。

## 5 实验结果与分析

### 5.1 核心指标趋势

图 1 展示了五类模型在 ETTh1 和 ETTm1 上随预测步长变化的 MSE、MAE 和 R2 趋势。总体上，随着预测步长从 24 增加到 336，模型误差普遍升高，R2 整体下降，说明长期预测中的误差累积是稳定存在的现象。

![核心指标趋势](../results/figures/formal_metric_trends.png)

**图 1：核心实验指标趋势。** 图中比较了 5 个模型在 2 个数据集和 5 个预测步长上的 MSE、MAE 与 R2。

### 5.2 不同数据集上的最优模型

ETTh1 上 PatchTST 在所有预测步长中均取得最低 MSE。其 h24、h48、h96、h168 和 h336 的 MSE 分别为 0.380213、0.420251、0.483175、0.513911 和 0.594367。虽然 MSE 随预测步长增加而上升，但 PatchTST 始终保持相对优势，说明 patch 化输入与通道独立建模对小时级 ETT 数据较为有效。

ETTm1 上 Autoformer 在所有预测步长中均取得最低 MSE。其 h24、h48、h96、h168 和 h336 的 MSE 分别为 0.307059、0.447473、0.460524、0.508573 和 0.554604。ETTm1 频率更高、样本量更大，Autoformer 的序列分解和周期建模可能更容易捕捉 15 分钟级数据中的局部周期和趋势变化。

| 数据集 | 预测步长 | 最优模型 | MSE | MAE | R2 |
| --- | ---: | --- | ---: | ---: | ---: |
| ETTh1 | 24 | PatchTST | 0.380213 | 0.411152 | 0.702517 |
| ETTh1 | 48 | PatchTST | 0.420251 | 0.432552 | 0.670760 |
| ETTh1 | 96 | PatchTST | 0.483175 | 0.472237 | 0.621377 |
| ETTh1 | 168 | PatchTST | 0.513911 | 0.489040 | 0.597451 |
| ETTh1 | 336 | PatchTST | 0.594367 | 0.545011 | 0.532757 |
| ETTm1 | 24 | Autoformer | 0.307059 | 0.351648 | 0.758935 |
| ETTm1 | 48 | Autoformer | 0.447473 | 0.439475 | 0.648585 |
| ETTm1 | 96 | Autoformer | 0.460524 | 0.452287 | 0.637520 |
| ETTm1 | 168 | Autoformer | 0.508573 | 0.480255 | 0.599255 |
| ETTm1 | 336 | Autoformer | 0.554604 | 0.507802 | 0.562683 |

![各预测步长最优模型](../results/figures/formal_best_model_by_horizon.png)

**图 2：各数据集和预测步长的最优模型。** ETTh1 的最优模型均为 PatchTST，ETTm1 的最优模型均为 Autoformer。

### 5.3 模型平均表现与复杂度权衡

跨 2 个数据集和 5 个预测步长取平均后，PatchTST 的平均 MSE 最低，为 0.469335；Autoformer 次之，为 0.514711。Informer、Transformer 和 LSTM 的平均 MSE 分别为 0.817005、0.917335 和 0.946142，明显弱于 PatchTST 和 Autoformer。标准 Transformer 的平均参数量最高，但性能并未优于面向时序结构设计的变体，说明直接套用标准注意力结构并不能充分适应当前长时序预测任务。

| 模型 | 平均 MSE | 平均 MAE | 平均 R2 | 平均参数量 | 平均训练时间(s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PatchTST | 0.469335 | 0.451390 | 0.631359 | 112140.8 | 197.319 |
| Autoformer | 0.514711 | 0.485138 | 0.595797 | 110688.6 | 263.691 |
| Informer | 0.817005 | 0.644982 | 0.358376 | 110131.8 | 299.006 |
| Transformer | 0.917335 | 0.691747 | 0.279686 | 132768.0 | 225.111 |
| LSTM | 0.946142 | 0.702745 | 0.257014 | 117280.0 | 128.940 |

![复杂度对比](../results/figures/formal_complexity_tradeoff.png)

**图 3：模型复杂度与性能权衡。** PatchTST 在平均误差和训练耗时之间取得较好平衡；LSTM 训练较快但误差较高。

### 5.4 预测曲线与残差分析

预测曲线进一步展示了模型在具体样本上的行为。ETTh1 h96 的 PatchTST 能够捕捉目标序列的主要波动趋势，但在局部突变处存在平滑化倾向。ETTh1 h336 中，这种现象更明显，模型倾向于输出较平滑的远期趋势，对远期细节变化的拟合能力减弱。

![ETTh1 h96 PatchTST 预测曲线](../results/figures/prediction_ETTh1_h96_patchtst.png)

**图 4：ETTh1 h96 PatchTST 预测曲线。** 模型能够拟合主要趋势，但局部突变处仍存在偏差。

![ETTh1 h336 PatchTST 残差](../results/figures/residual_ETTh1_h336_patchtst.png)

**图 5：ETTh1 h336 PatchTST 残差。** 长预测窗口下残差更容易出现持续偏正或偏负，体现出误差累积和相位偏移。

ETTm1 上 Autoformer 表现最优。h96 预测曲线显示其能够较好跟随高频数据中的周期波动；但在 h336 上，残差仍会随预测窗口拉长而扩大。这说明序列分解能够提升高频数据上的整体性能，但不能完全消除远期预测中的不确定性。

![ETTm1 h96 Autoformer 预测曲线](../results/figures/prediction_ETTm1_h96_autoformer.png)

**图 6：ETTm1 h96 Autoformer 预测曲线。** Autoformer 对高频数据的主要周期变化有较好拟合。

![ETTm1 h336 Autoformer 残差](../results/figures/residual_ETTm1_h336_autoformer.png)

**图 7：ETTm1 h336 Autoformer 残差。** 即使在最优模型上，长步长预测仍存在远期误差扩大的问题。

## 6 消融实验

消融实验结果表明，PatchTST 的通道独立建模和 Autoformer 的序列分解模块是当前实验中贡献最显著的结构。将 PatchTST 改为通道混合后，平均 MSE 上升 162.25%，平均 R2 下降 0.664538；移除 Autoformer 的序列分解后，平均 MSE 上升 81.48%，平均 R2 下降 0.386341。

| 消融项 | 平均 MSE 变化 | 平均 MSE 变化率 | 平均 R2 变化 |
| --- | ---: | ---: | ---: |
| 移除 PatchTST 通道独立建模 | +0.845824 | +162.25% | -0.664538 |
| 移除 Autoformer 序列分解 | +0.491294 | +81.48% | -0.386341 |
| 移除 PatchTST patching | +0.041929 | +8.54% | -0.032984 |
| 将 Auto-Correlation 替换为标准注意力 | -0.022206 | -3.15% | +0.017405 |

![消融影响](../results/figures/ablation_delta_mse_pct.png)

**图 8：消融实验的 MSE 变化率。** 通道独立建模和序列分解是最关键的性能来源。

PatchTST 的通道独立建模对性能影响最大。该结果说明，在当前 7 变量 ETT 数据中，直接混合多变量输入可能引入变量间分布差异和噪声干扰；相反，通道独立建模通过共享参数但分变量处理的方式保留了单变量模式。

Autoformer 的序列分解同样关键。移除分解后，模型需要直接在原始序列上学习趋势和周期结构，导致 h96 和 h336 上的误差均明显增加。这与 Autoformer 的设计动机一致，即长时序预测中的趋势和季节结构应被显式建模。

Auto-Correlation 的消融结果较复杂。将其替换为标准注意力后，ETTm1 上 MSE 略有上升，但 ETTh1 上反而略有下降，使平均 MSE 变化为负。这一现象并不能证明 Auto-Correlation 无效，而更可能说明当前项目中的轻量 Autoformer 实现、训练轮数或超参数尚未完全发挥 Auto-Correlation 的优势。后续可围绕分解窗口、模型宽度、top-k lag 和训练时长继续调参。

## 7 讨论

### 7.1 数据集差异

ETTh1 与 ETTm1 的最优模型不同，说明长时序预测模型的适用性与数据频率、样本规模和周期结构有关。ETTh1 为小时级数据，PatchTST 的 patch 化输入能够较好保留局部片段模式，并减少注意力 token 数量。ETTm1 为 15 分钟级数据，样本更多、周期更密集，Autoformer 的趋势和季节分解更容易发挥作用。

### 7.2 长步长误差累积

从 h24 到 h336，最优模型的 MSE 仍持续上升，说明长步长预测的误差累积不是单个模型结构即可完全解决的问题。预测曲线和残差图显示，模型对平滑周期变化拟合较好，但对突变、相位变化和远期细节更敏感。随着预测窗口变长，模型更倾向于输出平均化或平滑化趋势，这会降低远期局部波动的拟合质量。

### 7.3 超参数与复现实验经验

本项目使用统一的 96 步回看窗口，使不同模型在相同历史信息下比较。`epochs=20` 和 `patience=5` 在当前规模下能够较快得到可复现实验结果，但也可能限制部分模型的充分收敛。实验中使用 `run_tag` 区分 `formal_seed42`、`ablation_seed42` 和临时 smoke 结果，有助于避免结果覆盖和汇总混入。Windows 环境下还需要确认 CUDA Python 环境，避免误用 CPU 解释器造成训练时间异常。

### 7.4 局限性

本文结果存在三个主要边界。第一，当前正式主实验只覆盖 ETTh1 和 ETTm1，ECL 高维数据尚未形成完整主结果矩阵，因此不能直接推广到高维电力负荷预测场景。第二，所有正式结果基于单随机种子 42，尚未报告多随机种子的均值和方差。第三，当前模型实现为课程项目中的轻量版本，与原论文完整配置可能存在差异，因此本文更适合说明统一实现下的相对趋势，而不是复现原论文最优指标。

## 8 结论

本文基于统一训练和评估框架，对 LSTM、Transformer、Informer、Autoformer 与 PatchTST 在长时序预测任务中的表现进行了系统比较。实验结果显示，PatchTST 在 ETTh1 的五个预测步长上均取得最低 MSE，Autoformer 在 ETTm1 的五个预测步长上均取得最低 MSE。跨数据集平均后，PatchTST 取得最低平均 MSE，Autoformer 次之，二者明显优于 LSTM、标准 Transformer 和 Informer。

消融实验进一步表明，PatchTST 的通道独立建模和 Autoformer 的序列分解模块对性能提升贡献最大。预测曲线和残差图说明，即使最优模型能够较好捕捉主要趋势和周期，长步长预测仍存在远期误差累积、局部突变拟合不足和预测平滑化等问题。后续工作可从三个方向推进：补充 ECL 高维正式实验，进行多随机种子重复验证，以及围绕 patch 长度、分解窗口、模型宽度和 Auto-Correlation 配置开展系统调参。

## 参考文献

[1] Hochreiter S, Schmidhuber J. Long Short-Term Memory. Neural Computation, 1997.  
[2] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. NeurIPS, 2017.  
[3] Zhou H, Zhang S, Peng J, et al. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. AAAI, 2021.  
[4] Wu H, Xu J, Wang J, Long M. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. NeurIPS, 2021.  
[5] Nie Y, Nguyen N H, Sinthong P, Kalagnanam J. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. ICLR, 2023.  
[6] 项目文档与实验结果：`docs/model_introduction.md`、`docs/analysis_step6.md`、`results/formal_seed42_all.csv`、`results/ablation_seed42_vs_formal_comparison.csv`。

## 附录 A 术语与缩写

| 术语 | 含义 | 本文使用方式 |
| --- | --- | --- |
| LSTM | Long Short-Term Memory | 首次全称，后续使用 LSTM |
| Transformer | 基于自注意力的序列模型 | 指标准 Transformer 基线 |
| Informer | 使用 ProbSparse Attention 的长序列预测模型 | 高效注意力变体 |
| Autoformer | 使用序列分解和 Auto-Correlation 的预测模型 | 趋势和周期建模变体 |
| PatchTST | 使用 patching 和通道独立建模的预测模型 | 局部片段建模变体 |
| MSE | Mean Squared Error | 主要误差指标，越低越好 |
| MAE | Mean Absolute Error | 辅助误差指标，越低越好 |
| R2 | Coefficient of Determination | 拟合优度指标，越高越好 |
| h24/h48/h96/h168/h336 | 预测步长 | 分别表示预测未来 24/48/96/168/336 个时间步 |

## 附录 B 主要结果文件

- `results/formal_seed42_all.csv`
- `results/formal_seed42_all.md`
- `results/ablation_seed42_summary.csv`
- `results/ablation_seed42_summary.md`
- `results/ablation_seed42_vs_formal_comparison.csv`
- `results/ablation_seed42_vs_formal_comparison.md`
- `results/figures/manifest.json`
- `results/figures/prediction_samples_summary.csv`
