# Pure forward inference benchmark

## 计时口径

- 只测 `torch.inference_mode()` 下的 `model(x)` 前向传播。
- 不包含 DataLoader、数据搬运、反归一化、指标计算或结果保存时间。
- 使用 `docs/best_model_params.md` 记录的 v2 调优后结构参数。
- 权重使用随机初始化；这不影响模型结构的前向计算量，但不代表某个具体 checkpoint 的预测精度。
- 每个配置先 warmup，再重复计时，并在 CUDA/MPS 上做同步。

## 输出文件

- 明细 CSV：`archive/v2_results/experiments/validation_best_full_matrix/summaries/csv/pure_forward_inference_benchmark.csv`
- 按模型汇总 CSV：`archive/v2_results/experiments/validation_best_full_matrix/summaries/csv/pure_forward_inference_benchmark_by_model.csv`
- 按模型/步长汇总 CSV：`archive/v2_results/experiments/validation_best_full_matrix/summaries/csv/pure_forward_inference_benchmark_by_model_horizon.csv`
- 元数据 JSON：`archive/v2_results/experiments/validation_best_full_matrix/summaries/csv/pure_forward_inference_benchmark_metadata.json`

## Batch size = 1：五模型平均单次 forward 延迟

| model | n_tasks | mean_latency_median_ms | mean_latency_per_sample_ms | model_params | device |
| --- | --- | --- | --- | --- | --- |
| autoformer | 10 | 7.2089 | 7.2089 | 89271 | mps |
| informer | 10 | 2.6577 | 2.6577 | 182175 | mps |
| lstm | 10 | 2.2747 | 2.2747 | 380328 | mps |
| patchtst | 10 | 1.3669 | 1.3669 | 77488 | mps |
| transformer | 10 | 1.4791 | 1.4791 | 238376 | mps |

## Batch size = 128：五模型平均吞吐

| model | n_tasks | mean_latency_median_ms | mean_latency_per_sample_ms | mean_throughput_samples_per_sec | model_params | device |
| --- | --- | --- | --- | --- | --- | --- |
| autoformer | 10 | 15.1731 | 0.1185 | 8506.2526 | 89271 | mps |
| informer | 10 | 32.2407 | 0.2519 | 4029.3919 | 182175 | mps |
| lstm | 10 | 13.6797 | 0.1069 | 9370.6290 | 380328 | mps |
| patchtst | 10 | 5.3294 | 0.0416 | 24101.3813 | 77488 | mps |
| transformer | 10 | 15.2688 | 0.1193 | 8395.2298 | 238376 | mps |

## 明细预览

| dataset | horizon | model | batch_size | latency_median_ms | latency_p95_ms | throughput_samples_per_sec |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 24 | lstm | 1 | 1.6164 | 6.5600 | 618.6524 |
| ETTh1 | 24 | lstm | 128 | 13.3518 | 15.0836 | 9586.6987 |
| ETTh1 | 24 | transformer | 1 | 2.0690 | 10.8772 | 483.3205 |
| ETTh1 | 24 | transformer | 128 | 14.2855 | 16.8686 | 8960.1607 |
| ETTh1 | 24 | informer | 1 | 5.8856 | 13.4488 | 169.9067 |
| ETTh1 | 24 | informer | 128 | 27.2789 | 32.3651 | 4692.2720 |
| ETTh1 | 24 | autoformer | 1 | 8.1983 | 30.2569 | 121.9760 |
| ETTh1 | 24 | autoformer | 128 | 13.5055 | 15.6705 | 9477.5911 |
| ETTh1 | 24 | patchtst | 1 | 1.1217 | 5.3377 | 891.5302 |
| ETTh1 | 24 | patchtst | 128 | 4.7944 | 9.2758 | 26697.8364 |
| ETTh1 | 48 | lstm | 1 | 2.5445 | 8.2381 | 393.0014 |
| ETTh1 | 48 | lstm | 128 | 12.4751 | 15.2367 | 10260.4014 |
