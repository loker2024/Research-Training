# 高效时序预测模型详解

## 目录

- [1. Informer (AAAI 2021)](#1-informer)
- [2. Autoformer (NeurIPS 2021)](#2-autoformer)
- [3. PatchTST (ICLR 2023)](#3-patchtst)
- [4. 三者对比总结](#4-对比总结)

---

## 1. Informer

> **论文**: Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI 2021 Best Paper)

### 1.1 背景问题

标准 Transformer 用于时序预测有两个瓶颈：

1. **二次复杂度问题**：自注意力机制需要计算 $QK^T$，复杂度 $O(L^2)$，当序列长度 $L$ 较大时计算开销巨大
2. **内存瓶颈**：堆叠多层 Transformer 会导致内存占用激增，难以处理长序列
3. **逐步解码低效**：推理时需要逐步生成预测，速度慢

### 1.2 核心创新

#### (1) ProbSparse 注意力机制

**关键观察**：在时序预测中，注意力分布通常是**稀疏的**——只有少数 Query 对应的时间步真正重要，大部分注意力权重接近均匀分布。

**数学定义**：

标准自注意力：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q, K, V \in \mathbb{R}^{L \times d}$，注意力矩阵为 $L \times L$。

**稀疏性度量**：

定义第 $i$ 个 Query 的"活跃度"：
$$M(q_i, K) = \max_j \frac{q_i k_j^T}{\sqrt{d}} - \frac{1}{L} \sum_{j=1}^{L} \frac{q_i k_j^T}{\sqrt{d}}$$

$M$ 值越大，说明该 Query 的注意力分布越不均匀（越重要）。

**ProbSparse 策略**：

1. 计算所有 Query 的 $M$ 值
2. 只选择 Top-$u$ 个最重要的 Query（$u = c \cdot \ln L$，$c$ 是常数）
3. 其余 Query 的注意力输出用均值 $\frac{1}{L}\sum V$ 代替

```
标准注意力: 所有 Query × 所有 Key → O(L²)
ProbSparse: Top-u Query × 所有 Key → O(L log L)
```

#### (2) Self-attention Distilling

随着层数增加，注意力特征图中会出现**冗余的相似性**。Informer 引入逐层压缩机制：

```
Layer 1: [x1, x2, x3, x4, x5, x6, x7, x8]  (L 个 token)
            ↓  Conv1D + MaxPool + ELU
Layer 2: [h1, h2, h3, h4]                     (L/2 个 token)
            ↓  Conv1D + MaxPool + ELU
Layer 3: [g1, g2]                             (L/4 个 token)
```

数学表达：
$$X_{l+1} = \text{MaxPool}(\text{ELU}(\text{Conv1d}(X_l)))$$

这类似于 CNN 中的下采样，逐层减少序列长度，同时提取高层特征。

#### (3) Generative Decoder

传统方法逐步生成预测（自回归），Informer 一次性生成整个预测序列：

```
输入: [t-L+1, ..., t]  (L 长度的编码器输出)
输出: [t+1, ..., t+H]  (H 长度的预测序列)
```

实现方式：
1. 将目标序列初始化为零（或均值）
2. 与编码器输出拼接
3. 通过解码器一次性生成所有预测步

### 1.3 模型架构

```
┌─────────────────────────────────────────────────────────┐
│                      Informer                            │
├─────────────────────────────────────────────────────────┤
│  输入序列 [x₁, x₂, ..., xₗ]                             │
│         ↓                                               │
│  ┌─────────────────┐                                    │
│  │  Encoding Block  │ × N 层                            │
│  │  - ProbSparse    │                                   │
│  │    Self-Attn     │                                   │
│  │  - Distilling    │                                   │
│  └────────┬────────┘                                    │
│           ↓                                             │
│  ┌─────────────────┐    ┌─────────────────┐            │
│  │  Decoding Block  │ ←─│  Cross Attention │            │
│  │  - Self-Attn     │    │  (Encoder-Decoder)│           │
│  │  - FFN           │    └─────────────────┘            │
│  └────────┬────────┘                                    │
│           ↓                                             │
│  输出序列 [ŷ₁, ŷ₂, ..., ŷₕ]                             │
└─────────────────────────────────────────────────────────┘
```

### 1.4 复杂度分析

| 模块 | 标准 Transformer | Informer |
|------|-----------------|----------|
| Self-Attention | $O(L^2 d)$ | $O(L \log L \cdot d)$ |
| 空间复杂度 | $O(L^2)$ | $O(L \log L)$ |
| 解码 | $O(H \cdot L \cdot d)$ | $O(H \cdot d)$ |

### 1.5 关键实现细节

```python
# ProbSparse 注意力核心逻辑
def prob_sparse_attention(Q, K, V, top_u):
    """
    Q: (B, H, L_Q, D)  - Query
    K: (B, H, L_K, D)  - Key
    V: (B, H, L_V, D)  - Value
    top_u: int - 选择的 Query 数量
    """
    # 1. 计采样部分 Key 计算 M 值
    U = c * ln(L_Q)  # 采样数量
    index = random_sample(K, U)
    K_sample = K[:, :, index, :]

    # 2. 计算 M 值 = max - mean
    Q_K = torch.matmul(Q, K_sample.transpose(-2, -1)) / sqrt(d)
    M = Q_K.max(dim=-1) - Q_K.mean(dim=-1)

    # 3. 选择 Top-u 个 Query
    top_index = M.topk(top_u).indices

    # 4. 只对 Top-u Query 计算完整注意力
    Q_top = Q[:, :, top_index, :]
    attn = softmax(Q_top @ K^T / sqrt(d)) @ V

    # 5. 其余位置用均值填充
    output = mean(V).expand(...)
    output[:, :, top_index, :] = attn

    return output
```

---

## 2. Autoformer

> **论文**: Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting (NeurIPS 2021)

### 2.1 背景问题

时序数据通常具有以下特征：
1. **趋势性**（Trend）：长期上升或下降
2. **周期性**（Seasonal）：重复的模式（如 24 小时用电高峰）
3. **随机性**（Residual）：噪声

标准 Transformer 直接对原始序列建模，难以捕捉这些结构。

### 2.2 核心创新

#### (1) 序列分解模块（Series Decomposition）

采用经典的时间序列分解思想，将输入分解为趋势项和周期项：

```
输入 X → [移动平均] → 趋势项 Trend (T)
       → [减去趋势] → 周期项 Seasonal (S)
```

数学表达：
$$X = T + S$$

实现方式：
$$T = \text{AvgPool}(\text{Padding}(X))$$
$$S = X - T$$

其中 AvgPool 使用 kernel_size = 25（可调节），padding 保持序列长度不变。

#### (2) 自相关机制（Auto-Correlation）

**核心思想**：用**序列相关性**替代点积注意力，天然捕捉周期性模式。

**自相关函数**：
$$R(\tau) = \sum_{t} X_t \cdot X_{t-\tau}$$

$R(\tau)$ 表示序列与自身延迟 $\tau$ 步的相关性。峰值对应的 $\tau$ 就是主要周期。

**与标准注意力的区别**：

```
标准注意力:
  相似度 = dot(q_i, k_j)  → 逐点比较
  聚合 = softmax(相似度) × v_j → 加权求和

自相关:
  相似度 = corr(X, lag(X, τ))  → 整体序列的周期相关性
  聚合 = 相似度 × lag(V, τ)    → 按周期延迟聚合
```

**Top-K 周期选择**：

1. 计算自相关值 $R(\tau_1), R(\tau_2), ...$
2. 选择 Top-K 个最显著的周期
3. 对应的延迟序列加权聚合

```python
def auto_correlation(X, V, top_k):
    """
    X: (B, L, D) - 用于计算相关性
    V: (B, L, D) - 用于聚合
    """
    # 1. FFT 计算自相关（频域乘法 = 时域卷积）
    X_fft = torch.fft.rfft(X, dim=1)
    R = torch.fft.irfft(X_fft * torch.conj(X_fft), dim=1)

    # 2. 选择 Top-K 周期
    top_k_values, top_k_indices = R.topk(top_k, dim=1)

    # 3. 按延迟聚合
    V_shifted = torch.roll(V, shifts=tau, dims=1)
    output = sum(weight * V_shifted for tau in top_k_indices)

    return output
```

**复杂度**：$O(L \log L)$（FFT 计算）

#### (3) 两阶段聚合

Autoformer 的注意力分为两步：

```
Step 1: 周期内聚合（Intra-period）
  - 对每个周期内的元素做加权求和
  - 类似于局部注意力

Step 2: 周期间交互（Inter-period）
  - 不同周期之间交换信息
  - 类似于全局注意力
```

### 2.3 模型架构

```
┌──────────────────────────────────────────────────────────────┐
│                        Autoformer                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  输入: X (原始序列)                                           │
│         ↓                                                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  Auto-Correlation Block                  │ │
│  │                                                          │ │
│  │    X ──→ [Series Decompose] ──→ Seasonal (S), Trend (T) │ │
│  │              ↓                                           │ │
│  │    S ──→ [Auto-Correlation] ──→ S'                      │ │
│  │              ↓                                           │ │
│  │    S' + T ──→ [聚合] ──→ 输出                           │ │
│  └─────────────────────────────────────────────────────────┘ │
│         ↓  × N 层                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  Decoder (同样结构)                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│         ↓                                                     │
│  输出: Trend_final + Seasonal_final                           │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

每个 Block 的详细流程：

```
输入: X
  │
  ├──→ [Series Decompose] ──→ Trend₁, Seasonal₁
  │
  ├──→ [Auto-Correlation(Seasonal₁)] ──→ Seasonal_out
  │
  ├──→ [Series Decompose(Seasonal_out + Trend₁)] ──→ Trend₂, Seasonal₂
  │
  └──→ 输出: Seasonal₂ (传递到下一层)
```

### 2.4 与标准 Transformer 的对比

| 组件 | 标准 Transformer | Autoformer |
|------|-----------------|------------|
| 相似度计算 | 点积 $QK^T$ | 自相关 $R(\tau)$ |
| 聚合方式 | 加权求和 | 延迟聚合 |
| 序列结构 | 不分解 | 分解为 Trend + Seasonal |
| 复杂度 | $O(L^2)$ | $O(L \log L)$ |
| 周期建模 | 弱 | 强 |

### 2.5 关键实现细节

```python
class AutoCorrelation(nn.Module):
    def __init__(self, top_k=3):
        super().__init__()
        self.top_k = top_k

    def forward(self, queries, keys, values):
        B, L, D = queries.shape

        # 1. FFT 计算自相关
        q_fft = torch.fft.rfft(queries, dim=1)
        k_fft = torch.fft.rfft(keys, dim=1)
        # 频域点积 = 时域卷积
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), dim=1)

        # 2. 选择 Top-K 周期
        top_k_corr, top_k_indices = torch.topk(corr, self.top_k, dim=1)

        # 3. 按延迟聚合 Value
        output = torch.zeros_like(values)
        for i in range(self.top_k):
            tau = top_k_indices[:, i, :].unsqueeze(-1)  # 延迟量
            weight = top_k_corr[:, i, :].unsqueeze(-1)  # 权重
            # 循环移位
            shifted = torch.roll(values, shifts=tau.item(), dims=1)
            output += weight * shifted

        return output / self.top_k


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

    def forward(self, x):
        # x: (B, L, D)
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend
```

---

## 3. PatchTST

> **论文**: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (ICLR 2023)

### 3.1 背景问题

1. **计算效率**：直接对原始序列做注意力，token 数 = 序列长度，计算量大
2. **信息密度低**：每个 token 只代表一个时间点，语义信息少
3. **多变量建模**：标准方法将所有变量拼接成一个长序列，变量间相互干扰

### 3.2 核心创新

#### (1) Patching（切片）

借鉴 Vision Transformer (ViT) 的思想，将时间序列切成**不重叠的 Patch**：

```
原始序列 (L=96):
[x1, x2, x3, x4, x5, x6, x7, x8, ..., x96]

切成 Patch (patch_len=16, stride=16):
[P1: x1-x16]
[P2: x17-x32]
[P3: x33-x48]
[P4: x49-x64]
[P5: x65-x80]
[P6: x81-x96]

→ 6 个 token，每个 token 是 16 维向量
```

**参数设置**：
- `patch_len`: 每个 patch 的长度，推荐 16
- `stride`: 滑动步长，通常等于 patch_len（不重叠）

**好处**：

1. **降维**：token 数从 $L$ 降到 $L/P$（$P$ 是 patch 长度），注意力计算量减少 $P^2$ 倍
2. **信息密度高**：每个 token 包含局部时序模式（如一天的变化趋势）
3. **类似 CNN 感受野**：每个 patch 类似于卷积核覆盖的区域

数学表达：
$$Z = \text{Patch}(X) \in \mathbb{R}^{N \times P}$$

其中 $N = L/P$ 是 patch 数量，$P$ 是 patch 长度。

然后通过线性投影：
$$Z_{\text{embed}} = Z \cdot W_{\text{proj}} + b \in \mathbb{R}^{N \times d_{\text{model}}}$$

#### (2) Channel Independence（通道独立）

**传统方法**（Channel Mixing）：
```
输入: [变量1, 变量2, ..., 变量C] → 拼接成一个长序列 → Transformer
```
问题：不同变量的量纲、分布、模式差异大，混合建模会相互干扰。

**Channel Independence**：
```
变量1 → 独立的 Transformer → 预测变量1
变量2 → 独立的 Transformer → 预测变量2
...
变量C → 独立的 Transformer → 预测变量C
```

每个变量共享同一个 Transformer 权重，但独立处理。

**为什么有效**：
1. 避免不同变量间的噪声干扰
2. 共享参数，不会增加参数量
3. 实验证明效果优于 Channel Mixing

#### (3) 可选：Channel Mixing（对比实验用）

也提供 Channel Mixing 版本用于消融实验：
- 将所有变量的 patch 拼接在一起
- 让 Transformer 学习变量间关系

### 3.3 模型架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         PatchTST                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  输入: X ∈ R^{B × L × C}  (B=batch, L=序列长度, C=变量数)        │
│         ↓                                                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Channel Independence                            │ │
│  │  每个变量独立处理: X_c ∈ R^{B × L}                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│         ↓                                                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Patching + Embedding                            │ │
│  │  X_c → [P1, P2, ..., P_N] → Linear → Z ∈ R^{B × N × d}    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│         ↓                                                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Transformer Encoder                             │ │
│  │  + Positional Encoding                                       │ │
│  │  + Multi-Head Self-Attention × L 层                         │ │
│  │  + FFN                                                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│         ↓                                                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Flatten + Linear                                │ │
│  │  Z ∈ R^{B × N × d} → Flatten → Linear → Ŷ ∈ R^{B × H}    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│         ↓                                                         │
│  输出: 所有变量的预测拼接                                         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.4 详细流程示例

以 ETTh1 数据集为例（7 变量，序列长度 96）：

```
输入: X ∈ R^{B × 96 × 7}

Step 1: Channel Independence
  变量1: X₁ ∈ R^{B × 96}
  变量2: X₂ ∈ R^{B × 96}
  ...
  变量7: X₇ ∈ R^{B × 96}

Step 2: Patching (patch_len=16, stride=16)
  X₁ → [P1, P2, P3, P4, P5, P6] ∈ R^{B × 6 × 16}
  → 6 个 patch，每个 16 维

Step 3: Linear Projection
  Z₁ = [P1, ..., P6] · W + b ∈ R^{B × 6 × d_model}

Step 4: Positional Encoding
  Z₁ = Z₁ + PE ∈ R^{B × 6 × d_model}

Step 5: Transformer Encoder
  Z₁' = TransformerEncoder(Z₁) ∈ R^{B × 6 × d_model}

Step 6: Flatten + Predict
  Flatten: Z₁' ∈ R^{B × 6*d_model}
  Linear: Ŷ₁ ∈ R^{B × H}  (H 是预测步长)

Step 7: 合并所有变量
  Ŷ = [Ŷ₁, Ŷ₂, ..., Ŷ₇] ∈ R^{B × H × 7}
```

### 3.5 与标准 Transformer 的对比

| 组件 | 标准 Transformer | PatchTST |
|------|-----------------|----------|
| Token 定义 | 单个时间点 | Patch（多个时间点） |
| Token 数量 | L | L/P |
| 注意力复杂度 | $O(L^2)$ | $O((L/P)^2)$ |
| 输入维度 | C 维（混合） | 1 维（独立） |
| 局部模式 | 弱 | 强（patch 保留） |

### 3.6 关键实现细节

```python
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x):
        """
        x: (B, L, 1) - 单变量序列
        """
        B, L, _ = x.shape

        # 切 patch: unfold 操作
        # x: (B, L) → (B, N, patch_len)
        x = x.squeeze(-1)  # (B, L)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # patches: (B, N, patch_len)

        # 线性投影
        z = self.projection(patches)  # (B, N, d_model)

        return z


class PatchTST(nn.Module):
    def __init__(self, input_size, seq_len, pred_len, patch_len=16,
                 d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()

        self.input_size = input_size
        self.pred_len = pred_len
        self.patch_len = patch_len

        # Patch 数量
        self.num_patches = (seq_len - patch_len) // (seq_len // (seq_len // patch_len)) + 1

        # 每个变量独立的 Patch 嵌入
        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            stride=patch_len,  # 不重叠
            d_model=d_model
        )

        # 位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model)
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出投影
        self.head = nn.Linear(d_model * self.num_patches, pred_len)

    def forward(self, x):
        """
        x: (B, L, C) - 多变量序列
        """
        B, L, C = x.shape

        # Channel Independence: 每个变量独立处理
        outputs = []
        for c in range(C):
            # 提取单变量: (B, L, 1)
            x_c = x[:, :, c:c+1]

            # Patch 嵌入: (B, N, d_model)
            z = self.patch_embedding(x_c)

            # 加位置编码
            z = z + self.pos_embedding

            # Transformer
            z = self.transformer(z)  # (B, N, d_model)

            # Flatten + 预测
            z = z.reshape(B, -1)  # (B, N*d_model)
            pred = self.head(z)  # (B, pred_len)

            outputs.append(pred)

        # 合并所有变量: (B, pred_len, C)
        output = torch.stack(outputs, dim=-1)

        return output
```

### 3.7 超参数推荐

根据原论文实验：

| 数据集 | patch_len | stride | d_model | nhead | num_layers |
|--------|-----------|--------|---------|-------|------------|
| ETTh1/ETTm1 | 16 | 16 | 128 | 8 | 3 |
| ECL | 16 | 16 | 128 | 16 | 3 |
| Traffic | 16 | 16 | 128 | 16 | 3 |

---

## 4. 对比总结

### 4.1 核心思想对比

| 模型 | 核心思想 | 解决的问题 |
|------|---------|-----------|
| **Informer** | 稀疏注意力 + 层级压缩 | 长序列计算效率 |
| **Autoformer** | 自相关 + 序列分解 | 周期性建模 |
| **PatchTST** | 切 Patch + 通道独立 | 降维 + 局部模式 |

### 4.2 技术细节对比

| 特性 | Informer | Autoformer | PatchTST |
|------|----------|------------|----------|
| 注意力机制 | ProbSparse | Auto-Correlation | 标准 Self-Attention |
| 复杂度 | $O(L \log L)$ | $O(L \log L)$ | $O((L/P)^2)$ |
| 序列分解 | ✗ | ✓ | ✗ |
| Patching | ✗ | ✗ | ✓ |
| Channel 策略 | Mixing | Mixing | Independence |
| 解码方式 | 一步生成 | 一步生成 | 一步生成 |

### 4.3 适用场景

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 超长序列（L>1000） | Informer | ProbSparse 效率最高 |
| 有明显周期的数据 | Autoformer | 自相关机制天然捕捉周期 |
| 多变量预测 | PatchTST | Channel Independence 避免干扰 |
| 计算资源有限 | PatchTST | Patch 降维效果最好 |

### 4.4 实验表现（ETTh1 数据集，Horizon=96）

参考各论文报告的结果：

| 模型 | MSE | MAE |
|------|-----|-----|
| Transformer (baseline) | ~0.8 | ~0.7 |
| Informer | 0.098 | 0.248 |
| Autoformer | 0.097 | 0.245 |
| PatchTST | 0.071 | 0.203 |

> PatchTST 通常在大多数数据集上取得最优或接近最优的结果。

---

## 5. 实现计划

### 5.1 实现顺序

1. **Informer** → 2. **Autoformer** → 3. **PatchTST**

理由：
- Informer 的编码器结构与现有 Transformer baseline 最接近，改造成本最低
- Autoformer 引入新的注意力机制，需要额外模块
- PatchTST 的 Patch 机制和 Channel Independence 与前两者差异较大

### 5.2 文件结构

```
models/
├── __init__.py
├── lstm.py
├── transformer.py
├── informer.py        # 新增
├── autoformer.py      # 新增
├── patchtst.py        # 新增
├── dataset.py
└── trainer.py
```

### 5.3 统一接口

所有模型保持相同的输入输出接口：

```python
class BaseModel(nn.Module):
    def __init__(self, input_size, horizon, ...):
        ...

    def forward(self, x):
        """
        Args:
            x: (batch, lookback, features)
        Returns:
            output: (batch, horizon, features)
        """
        ...
```

---

## 参考文献

1. Zhou, H., et al. "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting." AAAI 2021.
2. Wu, H., et al. "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting." NeurIPS 2021.
3. Nie, Y., et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023.
