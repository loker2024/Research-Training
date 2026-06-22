"""Run reproducible forecasting experiments from the command line."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from models import (  # noqa: E402
    AutoformerModel,
    AutoformerNoAutocorr,
    AutoformerNoDecomp,
    InformerModel,
    LSTMModel,
    PatchTSTChannelMix,
    PatchTSTModel,
    PatchTSTNoPatch,
    TimeSeriesDataset,
    Trainer,
    TransformerModel,
)

# 消融实验只改变被检验模块，其余结构参数继承 ETTh1 h96 验证集最优配置。
ABLATION_TUNED_MODEL_CONFIGS = {
    "autoformer": {
        "d_model": 64,
        "n_heads": 4,
        "n_encoder_layers": 2,
        "n_decoder_layers": 1,
        "d_ff": 128,
        "factor": 3,
        "dropout": 0.1,
        "kernel_size": 25,
    },
    "patchtst": {
        "d_model": 64,
        "n_heads": 8,
        "n_layers": 2,
        "d_ff": 128,
        "patch_len": 32,
        "stride": 8,
        "dropout": 0.1,
    },
}

ABLATION_TUNED_TRAINING_CONFIG = {
    "epochs": 50,
    "patience": 10,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 0.00001,
}

ABLATION_MODEL_CONFIGS = {
    "autoformer_ablation_base": dict(ABLATION_TUNED_MODEL_CONFIGS["autoformer"]),
    "autoformer_no_decomp": dict(ABLATION_TUNED_MODEL_CONFIGS["autoformer"]),
    "autoformer_no_autocorr": dict(ABLATION_TUNED_MODEL_CONFIGS["autoformer"]),
    "patchtst_ablation_base": dict(ABLATION_TUNED_MODEL_CONFIGS["patchtst"]),
    "patchtst_no_patch": dict(ABLATION_TUNED_MODEL_CONFIGS["patchtst"]),
    "patchtst_channel_mix": dict(ABLATION_TUNED_MODEL_CONFIGS["patchtst"]),
}


MODEL_BUILDERS = {
    "lstm": lambda input_size, horizon: LSTMModel(
        input_size=input_size, hidden_size=64, num_layers=2, dropout=0.1, horizon=horizon
    ),
    "lstm_baseline": lambda input_size, horizon: LSTMModel(
        input_size=input_size, hidden_size=256, num_layers=2, dropout=0.2, horizon=horizon
    ),
    "lstm_top1": lambda input_size, horizon: LSTMModel(
        input_size=input_size, hidden_size=256, num_layers=1, dropout=0.2, horizon=horizon
    ),
    "transformer": lambda input_size, horizon: TransformerModel(
        input_size=input_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        horizon=horizon,
    ),
    "informer": lambda input_size, horizon: InformerModel(
        input_size=input_size,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=1,
        d_ff=128,
        dropout=0.1,
        horizon=horizon,
    ),
    "autoformer": lambda input_size, horizon: AutoformerModel(
        input_size=input_size,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=1,
        d_ff=128,
        dropout=0.1,
        horizon=horizon,
    ),
    "patchtst": lambda input_size, horizon: PatchTSTModel(
        input_size=input_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        horizon=horizon,
    ),
    # --- 重做消融实验基线：与各消融变体使用同一批次和调优后结构参数 ---
    "autoformer_ablation_base": lambda input_size, horizon: AutoformerModel(
        input_size=input_size,
        horizon=horizon,
        **ABLATION_MODEL_CONFIGS["autoformer_ablation_base"],
    ),
    "patchtst_ablation_base": lambda input_size, horizon: PatchTSTModel(
        input_size=input_size,
        horizon=horizon,
        **ABLATION_MODEL_CONFIGS["patchtst_ablation_base"],
    ),
    "transformer_top1": lambda input_size, horizon: TransformerModel(
        input_size=input_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        horizon=horizon,
    ),
    # --- 消融变体 ---
    "autoformer_no_decomp": lambda input_size, horizon: AutoformerNoDecomp(
        input_size=input_size,
        horizon=horizon,
        **ABLATION_MODEL_CONFIGS["autoformer_no_decomp"],
    ),
    "autoformer_no_autocorr": lambda input_size, horizon: AutoformerNoAutocorr(
        input_size=input_size,
        horizon=horizon,
        **ABLATION_MODEL_CONFIGS["autoformer_no_autocorr"],
    ),
    "patchtst_no_patch": lambda input_size, horizon: PatchTSTNoPatch(
        input_size=input_size,
        horizon=horizon,
        **ABLATION_MODEL_CONFIGS["patchtst_no_patch"],
    ),
    "patchtst_channel_mix": lambda input_size, horizon: PatchTSTChannelMix(
        input_size=input_size,
        horizon=horizon,
        **ABLATION_MODEL_CONFIGS["patchtst_channel_mix"],
    ),
}


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def maybe_subset(dataset, limit: int | None):
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def create_loaders(
    dataset_name: str,
    horizon: int,
    batch_size: int,
    sample_limit: int | None,
    num_workers: int,
    data_dir: Path,
):
    train_dataset = TimeSeriesDataset(data_dir, dataset_name, horizon, "train")
    val_dataset = TimeSeriesDataset(data_dir, dataset_name, horizon, "val")
    test_dataset = TimeSeriesDataset(data_dir, dataset_name, horizon, "test")

    train_dataset = maybe_subset(train_dataset, sample_limit)
    val_dataset = maybe_subset(val_dataset, sample_limit)
    test_dataset = maybe_subset(test_dataset, sample_limit)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    return json.loads(config_path.read_text(encoding="utf-8"))


def get_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="Run forecasting experiments")
    parser.add_argument("--config", default="")
    parser.add_argument("--datasets", default=defaults.get("datasets", "ETTh1"))
    parser.add_argument("--horizons", default=defaults.get("horizons", "96"))
    parser.add_argument("--models", default=defaults.get("models", "autoformer"))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 1))
    parser.add_argument("--patience", type=int, default=defaults.get("patience", 3))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 32))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 1e-3))
    parser.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 1e-5))
    parser.add_argument("--device", default=defaults.get("device", "auto"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--run-tag", default=defaults.get("run_tag", ""))
    parser.add_argument("--data-dir", default=defaults.get("data_dir", str(ROOT / "data" / "processed")))
    parser.add_argument("--sample-limit", type=int, default=defaults.get("sample_limit", 128))
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 0))
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=defaults.get("skip_existing", False),
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        default=defaults.get("no_tensorboard", False),
    )
    return parser


def run_one(args, dataset_name: str, horizon: int, model_name: str) -> dict:
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    tag = f"_{args.run_tag}" if args.run_tag else ""
    run_name = f"{dataset_name}_h{horizon}_{model_name}{tag}"
    run_tag_dir = args.run_tag or "default"
    output_dir = ROOT / "results" / f"h{horizon}" / dataset_name / model_name / run_tag_dir
    result_path = output_dir / f"{run_name}_results.npy"
    summary_path = output_dir / f"{run_name}_summary.json"

    if args.skip_existing and result_path.exists() and summary_path.exists():
        print(f"跳过已存在结果: {run_name}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    train_loader, val_loader, test_loader = create_loaders(
        dataset_name,
        horizon,
        args.batch_size,
        args.sample_limit,
        args.num_workers,
        data_dir,
    )
    base_dataset = train_loader.dataset.dataset if isinstance(train_loader.dataset, Subset) else train_loader.dataset
    input_size = base_dataset.input_size
    target_idx = base_dataset.target_idx

    model = MODEL_BUILDERS[model_name](input_size, horizon)
    trainer = Trainer(model, device=args.device, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed)

    started = time.time()
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        patience=args.patience,
        save_dir=str(ROOT / "checkpoints"),
        model_name=run_name,
        log_dir=None if args.no_tensorboard else str(ROOT / "runs"),
    )
    train_seconds = time.time() - started
    predictions, targets = trainer.predict(test_loader)
    metrics = trainer.compute_metrics(predictions, targets, target_idx=target_idx)

    result = {
        "dataset": dataset_name,
        "horizon": horizon,
        "model": model_name,
        "epochs": args.epochs,
        "trained_epochs": len(history["train_losses"]),
        "best_epoch": int(np.argmin(history["val_losses"]) + 1),
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "device": trainer.device,
        "seed": args.seed,
        "run_tag": args.run_tag,
        "data_dir": str(data_dir),
        "sample_limit": args.sample_limit,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "input_size": input_size,
        "target_idx": target_idx,
        "model_config": ABLATION_MODEL_CONFIGS.get(model_name),
        "model_params": count_parameters(model),
        "train_time_seconds": train_seconds,
        "history": history,
        "metrics": metrics,
        "best_val_loss": history["best_val_loss"],
        "best_val_r2": history["best_val_r2"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_path, result, allow_pickle=True)
    summary_path.write_text(
        json.dumps(to_jsonable({k: v for k, v in result.items() if k != "history"}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved result: {result_path}")
    return result


def main() -> None:
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", default="")
    probe_args, remaining = config_probe.parse_known_args()
    config = load_config(probe_args.config)
    parser = get_parser(config)
    args = parser.parse_args(["--config", probe_args.config] + remaining)

    datasets = parse_csv_list(args.datasets)
    horizons = parse_int_list(args.horizons)
    models = parse_csv_list(args.models)
    unknown_models = sorted(set(models) - set(MODEL_BUILDERS))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")

    for dataset_name in datasets:
        for horizon in horizons:
            for model_name in models:
                run_one(args, dataset_name, horizon, model_name)


if __name__ == "__main__":
    main()
