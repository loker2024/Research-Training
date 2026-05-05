# PROGRESS.md

本文件记录 Research-Training 项目的工作日志、阶段成果、提交记录和后续待办。

## 2026-05-05

### 1. 初始化课题代码第一版

完成内容：

- 阅读并整理课题方向：长时序预测，关注 LSTM、Transformer、Informer、Autoformer、PatchTST。
- 新增第一版 PyTorch notebook：`notebooks/long_term_forecasting_v1.ipynb`。
- notebook 包含以下模块：
  - 全局配置 `ExperimentConfig`。
  - 合成数据生成，用于无真实数据时验证流程。
  - CSV 数据读取。
  - 时间顺序切分训练集、验证集、测试集。
  - 只使用训练集拟合标准化参数。
  - 滑动窗口样本构造。
  - LSTM 基线模型。
  - Transformer 基线模型。
  - PatchTST-lite 模型。
  - 移动平均时序分解模块。
  - MSE、MAE、MAPE 指标。
  - 训练循环、验证循环、测试评估。
  - 预测曲线和残差图。
  - 分解消融实验入口。
- 新增第一版实现文档：`docs/v1_implementation.md`。
- 新增项目说明：`README.md`。
- 新增依赖文件：`requirements.txt`。
- 新增数据目录说明：`data/README.md`。
- 新增 `.gitignore`，忽略数据文件、训练输出、checkpoint 和 notebook 临时文件。

验证情况：

- notebook JSON 可正常解析。
- notebook 代码语法检查通过。
- 当时系统 Python 未安装 `torch`，因此未执行真实训练烟测。

提交记录：

```text
3316e12 Add first PyTorch forecasting notebook
```

### 2. 增加数据集说明文档

完成内容：

- 新增 `docs/datasets.md`。
- 介绍 ETT、ECL、ILI、Traffic 数据集的背景、字段、适用场景和注意事项。
- 说明初学者推荐使用顺序：
  1. ETTh1
  2. ETTm1
  3. ECL
  4. ILI
  5. Traffic
- 说明时间序列数据不能随机划分，必须按时间顺序切分。
- 说明 `pred_len` 在小时级和 15 分钟级数据中的含义。
- 在 `README.md` 和 `data/README.md` 中加入数据集文档入口。

提交记录：

```text
76fab7d Add dataset guide documentation
```

### 3. 下载公开数据集到本地

完成内容：

- 从 Hugging Face `AutonLab/Timeseries-PILE/forecasting/autoformer` 下载公开时间序列数据集。
- 保存到本地 `data/` 目录。
- 按课题习惯将部分文件重命名为更直观的名称。

当前本地数据文件：

| 文件 | 来源文件 | 约大小 | Git 状态 |
| --- | --- | ---: | --- |
| `data/ETTh1.csv` | `ETTh1.csv` | 2.5 MB | ignored |
| `data/ETTh2.csv` | `ETTh2.csv` | 2.3 MB | ignored |
| `data/ETTm1.csv` | `ETTm1.csv` | 9.9 MB | ignored |
| `data/ETTm2.csv` | `ETTm2.csv` | 9.2 MB | ignored |
| `data/ECL.csv` | `electricity.csv` | 91 MB | ignored |
| `data/ILI.csv` | `national_illness.csv` | 65 KB | ignored |
| `data/Traffic.csv` | `traffic.csv` | 130 MB | ignored |

确认事项：

- 数据文件已下载完成。
- `git status --short --ignored data` 显示这些 CSV 为 `!!`，说明它们被 `.gitignore` 忽略。
- 数据集没有上传到 GitHub。

### 4. 增加项目级 AI 协作记忆

完成内容：

- 新增 `AGENTS.md`。
- 记录项目背景、目录约定、Git 与数据约定、文档约定、技术偏好、模型路线、数据集使用建议、运行验证流程和协作注意事项。
- 明确规定：
  - 文档默认使用 Markdown。
  - 数据集不上传 GitHub。
  - notebook 代码需要详细中文注释。
  - 数据切分必须按时间顺序。
  - 标准化只能使用训练集统计量。

提交记录：

```text
7a8a2a4 Add project agent memory
```

## 当前项目状态

远程仓库：

```text
https://github.com/loker2024/Research-Training.git
```

当前分支：

```text
main
```

当前已提交文档：

- `README.md`
- `AGENTS.md`
- `PROGRESS.md`
- `docs/v1_implementation.md`
- `docs/datasets.md`
- `data/README.md`
- `选题-时间序列预测.md`

当前已提交 notebook：

- `notebooks/long_term_forecasting_v1.ipynb`

当前本地存在但不提交的数据：

- `data/ETTh1.csv`
- `data/ETTh2.csv`
- `data/ETTm1.csv`
- `data/ETTm2.csv`
- `data/ECL.csv`
- `data/ILI.csv`
- `data/Traffic.csv`

## 待办事项

优先级较高：

1. 安装依赖，尤其是 `torch`。
2. 运行 notebook 默认合成数据实验，确认训练循环和图表输出正常。
3. 切换到 `data/ETTh1.csv`，跑通真实数据第一轮实验。
4. 记录 ETTh1 上 LSTM、Transformer、PatchTST-lite 的 MSE、MAE、MAPE。
5. 将结果表和关键图保存到 `outputs/`，但不提交大文件。

中期任务：

1. 增加多预测步长实验循环，例如 24、48、96、168、336。
2. 增加多模型自动对比表。
3. 增加 ECL、ILI、Traffic 的实验配置说明。
4. 增加参数量、训练时间和推理时间统计。
5. 将稳定代码逐步抽取到 `src/`，保留 notebook 作为实验入口。

后续模型扩展：

1. 实现 Informer 的核心 ProbSparse Attention。
2. 实现 Autoformer 的序列分解与 Auto-Correlation block。
3. 对比当前 PatchTST-lite 与更接近论文版本的 PatchTST。
4. 增加外生变量融合实验。
5. 增加概率预测或置信区间输出。

## 维护规则

- 每完成一次明确工作，都在本文件追加日志。
- 日志按日期倒序或顺序均可，但同一天内保持从早到晚。
- 每条日志尽量包含完成内容、验证情况、提交记录和遗留问题。
- 不记录大段命令输出，只记录关键结论。
- 数据文件、训练输出和模型权重不进入 Git。

