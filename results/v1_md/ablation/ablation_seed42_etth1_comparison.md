# ETTh1 消融实验对比表

## h96 对比

| 模型 | 类型 | MSE | MAE | R² | MSE 变化 |
|------|------|----:|----:|---:|---------:|
| Autoformer | 正式 | 0.601495 | 0.541733 | 0.528660 | — |
| autoformer_no_decomp | 消融 | 1.011054 | 0.783477 | 0.207723 | +68.1% ⬆ |
| autoformer_no_autocorr | 消融 | 0.542886 | 0.514618 | 0.574587 | -9.7% ⬇ |
| PatchTST | 正式 | 0.483175 | 0.472237 | 0.621377 | — |
| patchtst_no_patch | 消融 | 0.519393 | 0.496346 | 0.592996 | +7.5% ⬆ |
| patchtst_channel_mix | 消融 | 1.888603 | 1.028806 | -0.479937 | +290.8% ⬆ |

## h336 对比

| 模型 | 类型 | MSE | MAE | R² | MSE 变化 |
|------|------|----:|----:|---:|---------:|
| Autoformer | 正式 | 0.689396 | 0.598789 | 0.458053 | — |
| autoformer_no_decomp | 消融 | 1.509056 | 0.933451 | -0.186297 | +118.9% ⬆ |
| autoformer_no_autocorr | 消融 | 0.628062 | 0.573486 | 0.506269 | -8.9% ⬇ |
| PatchTST | 正式 | 0.594367 | 0.545011 | 0.532757 | — |
| patchtst_no_patch | 消融 | 0.597170 | 0.551790 | 0.530554 | +0.5% ≈ |
| patchtst_channel_mix | 消融 | 1.505545 | 0.964119 | -0.183537 | +153.4% ⬆ |

## 关键发现

1. **Series Decomposition 对 Autoformer 至关重要**：去除分解后 MSE 上升 68~119%，R² 大幅下降
2. **Auto-Correlation 对 Autoformer 效果存疑**：替换为标准 MHA 后 MSE 反而下降 9%，说明在当前轻量实现中 Auto-Correlation 可能不是关键
3. **Channel Independence 对 PatchTST 至关重要**：混合变量后 MSE 上升 153~291%，R² 转负
4. **Patching 对 PatchTST 影响较小**：去除 patch 后 MSE 仅上升 0.5~7.5%，效果基本持平

> 注：ETTm1 消融实验尚未完成（10/16），完整对比表待补充。
