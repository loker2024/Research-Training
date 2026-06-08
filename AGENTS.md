# 项目说明

## 项目概述

基于 LSTM、Transformer、Informer、Autoformer、PatchTST 的长时序预测研究项目。

## 数据集配置

**使用 3 个数据集**（精简版，兼顾不同频率和变量规模）：

| 数据集   | 变量数 | 频率   | 切分方式        |
| ----- | --- | ---- | ----------- |
| ETTh1 | 7   | 小时   | 前12月/4月/4月  |
| ETTm1 | 7   | 15分钟 | 前12月/4月/4月  |
| ECL   | 321 | 小时   | 70%/10%/20% |

**预测步长**：96、168、336（覆盖中长期）
**回看窗口**：96

> 原始 CSV 文件（data/\*.csv）保留全部 7 个数据集，仅处理以上 3 个。

## 环境配置

**本地虚拟环境**：

- Mac: Conda 环境 `miniMac`
- Windows: 项目根目录下的 `myenv`

激活虚拟环境：

```bash
# Windows
myenv\Scripts\activate

# Mac
conda activate miniMac
```

## 依赖安装

```bash
pip install -r requirements.txt
```

## 项目结构

```
Research-Training/
├── data/           # 数据集目录
├── docs/           # 文档
├── notebooks/      # Jupyter 笔记本
├── myenv/          # 本地虚拟环境 (已 gitignore)
└── 项目步骤.md     # 实施步骤说明
```

## 运行说明

1. 激活对应平台的虚拟环境（Mac: `miniMac`，Windows: `myenv`）
2. 启动 Jupyter: `jupyter notebook`
3. 按照 `项目步骤.md` 中的顺序执行实验



## Progress tracking

每完成一个独立步骤，都要更新 `docs/progress.md`，记录：

- 完成时间
- 完成内容
- 修改的文件
- 测试结果
- 下一步任务
