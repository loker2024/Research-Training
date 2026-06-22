# 五个模型最优参数记录

选择口径：基于 `ETTh1 h96` 超参数搜索结果，按验证集 `best_val_loss` 最低选择。

| 模型 | best_val_loss | 结构参数 | 训练参数 |
| --- | ---: | --- | --- |
| LSTM | 0.889822 | `hidden_size=256`, `num_layers=1`, `dropout=0.1` | `epochs=50`, `patience=10`, `batch_size=128`, `lr=0.001`, `weight_decay=0.0` |
| Transformer | 0.905789 | `d_model=128`, `nhead=4`, `num_layers=2`, `dim_feedforward=128`, `dropout=0.1` | `epochs=50`, `patience=10`, `batch_size=128`, `lr=0.0001`, `weight_decay=0.0` |
| Informer | 0.818046 | `d_model=64`, `n_heads=4`, `n_encoder_layers=2`, `n_decoder_layers=2`, `d_ff=256`, `factor=3`, `dropout=0.1` | `epochs=50`, `patience=10`, `batch_size=128`, `lr=0.001`, `weight_decay=0.00001` |
| Autoformer | 0.666360 | `d_model=64`, `n_heads=4`, `n_encoder_layers=2`, `n_decoder_layers=1`, `d_ff=128`, `factor=3`, `dropout=0.1`, `kernel_size=25` | `epochs=50`, `patience=10`, `batch_size=128`, `lr=0.001`, `weight_decay=0.00001` |
| PatchTST | 0.680877 | `d_model=64`, `n_heads=8`, `n_layers=2`, `d_ff=128`, `patch_len=32`, `stride=8`, `dropout=0.1` | `epochs=50`, `patience=10`, `batch_size=128`, `lr=0.001`, `weight_decay=0.00001` |

## 来源文件

| 模型 | 来源 |
| --- | --- |
| LSTM | `test_results/h96/ETTh1/lstm/ETTh1_h96_lstm_h256_l1_dp01_lr0.001_wd0.0_summary.json` |
| Transformer | `test_results/h96/ETTh1/transformer/ETTh1_h96_transformer_d128_h4_l2_ff128_dp01_lr0.0001_wd0.0_summary.json` |
| Informer | `test_results/h96/ETTh1/informer/ETTh1_h96_informer_d64_h4_enc2_dec2_ff256_fac3_dp01_summary.json` |
| Autoformer | `test_results/h96/ETTh1/autoformer/ETTh1_h96_autoformer_d64_h4_enc2_dec1_ff128_fac3_ks25_summary.json` |
| PatchTST | `test_results/h96/ETTh1/patchtst/ETTh1_h96_patchtst_d64_h8_l2_ff128_pl32_st8_dp01_summary.json` |
