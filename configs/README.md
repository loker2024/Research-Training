# Config Index

This directory keeps reusable experiment configurations for the current
ETTh1/ETTm1 forecasting scope. One-off recovery configs should not stay here
after the corresponding run has been completed.

## Core Experiments

| File | Purpose |
| --- | --- |
| `core_experiment_etth1_ettm1_formal.json` | Formal 50-run matrix: ETTh1/ETTm1 x 5 horizons x 5 models. |
| `core_experiment_smoke.json` | Small CLI smoke test for the config-driven runner. |

## Analysis Experiments

| File | Purpose |
| --- | --- |
| `univariate_multivariate_comparison.json` | Full univariate-vs-multivariate comparison matrix. |
| `ablation_rerun_etth1_ettm1.json` | Fair 24-run ablation rerun with Autoformer/PatchTST baselines and four variants in the same batch. |

## Hyperparameter Search

| File | Purpose |
| --- | --- |
| `lstm_search.json` | LSTM grid search on ETTh1 h96. |
| `transformer_search.json` | Transformer grid search on ETTh1 h96. |
| `informer_search.json` | Informer grid search on ETTh1 h96. |
| `autoformer_search.json` | Autoformer grid search on ETTh1 h96. |
| `patchtst_search.json` | PatchTST grid search on ETTh1 h96. |

## Historical Reproduction

| File | Purpose |
| --- | --- |
| `lstm_top1.json` | Historical LSTM Top-1 validation-loss config registered as `lstm_top1`. |
| `transformer_top1.json` | Historical Transformer Top-1 validation-loss config registered as `transformer_top1`. |

Notes:

- Current formal reports use ETTh1 and ETTm1 only.
- `sample_limit=0` means full data.
- Keep temporary continuation configs out of this directory once the interrupted
  run is finished.
