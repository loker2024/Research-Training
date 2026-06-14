"""Run and summarize univariate-vs-multivariate forecasting comparisons.

Univariate mode uses only the target column as input and predicts only the
target column. Multivariate mode keeps the original all-variable input/output.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import TimeSeriesDataset, Trainer  # noqa: E402
from scripts.run_experiments import (  # noqa: E402
    MODEL_BUILDERS,
    count_parameters,
    parse_csv_list,
    parse_int_list,
    to_jsonable,
)


DEFAULT_MODELS = "lstm,transformer,autoformer,patchtst"
RESULT_ROOT = ROOT / "results" / "univariate_multivariate"
CSV_DIR = ROOT / "results" / "v1_csv" / "feature_mode"
MD_DIR = ROOT / "results" / "v1_md" / "feature_mode"


class FeatureModeDataset(Dataset):
    """Dataset view for univariate or multivariate comparison."""

    def __init__(self, data_dir: Path, dataset_name: str, horizon: int, split: str, mode: str):
        if mode not in {"univariate", "multivariate"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.base = TimeSeriesDataset(data_dir, dataset_name, horizon, split)
        self.mode = mode
        self.dataset_name = dataset_name
        self.horizon = horizon
        self.split = split
        self.original_input_size = self.base.input_size
        self.original_target_idx = self.base.target_idx

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.mode == "univariate":
            target_idx = self.original_target_idx
            return x[:, target_idx : target_idx + 1], y[:, target_idx : target_idx + 1]
        return x, y

    @property
    def input_size(self) -> int:
        return 1 if self.mode == "univariate" else self.base.input_size

    @property
    def target_idx(self) -> int:
        return 0 if self.mode == "univariate" else self.base.target_idx


def maybe_subset(dataset: Dataset, limit: int | None):
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def unwrap_dataset(dataset: Dataset) -> FeatureModeDataset:
    if isinstance(dataset, Subset):
        return dataset.dataset
    return dataset


def create_loaders(args, dataset_name: str, horizon: int, mode: str):
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    train_dataset = FeatureModeDataset(data_dir, dataset_name, horizon, "train", mode)
    val_dataset = FeatureModeDataset(data_dir, dataset_name, horizon, "val", mode)
    test_dataset = FeatureModeDataset(data_dir, dataset_name, horizon, "test", mode)

    train_dataset = maybe_subset(train_dataset, args.sample_limit)
    val_dataset = maybe_subset(val_dataset, args.sample_limit)
    test_dataset = maybe_subset(test_dataset, args.sample_limit)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    )


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
    parser = argparse.ArgumentParser(description="Run univariate-vs-multivariate experiments")
    parser.add_argument("--config", default="")
    parser.add_argument("--datasets", default=defaults.get("datasets", "ETTh1,ETTm1"))
    parser.add_argument("--horizons", default=defaults.get("horizons", "96,336"))
    parser.add_argument("--models", default=defaults.get("models", DEFAULT_MODELS))
    parser.add_argument("--modes", default=defaults.get("modes", "univariate,multivariate"))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 5))
    parser.add_argument("--patience", type=int, default=defaults.get("patience", 3))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 32))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 1e-3))
    parser.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 1e-5))
    parser.add_argument("--device", default=defaults.get("device", "auto"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--run-tag", default=defaults.get("run_tag", "feature_mode_seed42"))
    parser.add_argument("--data-dir", default=defaults.get("data_dir", str(ROOT / "data" / "processed")))
    parser.add_argument("--sample-limit", type=int, default=defaults.get("sample_limit", 512))
    parser.add_argument("--num-workers", type=int, default=defaults.get("num_workers", 0))
    parser.add_argument("--skip-existing", action="store_true", default=defaults.get("skip_existing", False))
    parser.add_argument("--no-tensorboard", action="store_true", default=defaults.get("no_tensorboard", True))
    return parser


def summary_paths(run_tag: str) -> tuple[Path, Path]:
    safe_tag = run_tag or "default"
    return (
        CSV_DIR / f"{safe_tag}_comparison.csv",
        MD_DIR / f"{safe_tag}_comparison.md",
    )


def run_one(args, dataset_name: str, horizon: int, model_name: str, mode: str) -> dict:
    set_seed(args.seed)
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {model_name}")

    run_tag = args.run_tag or "default"
    run_name = f"{dataset_name}_h{horizon}_{model_name}_{mode}_{run_tag}"
    output_dir = RESULT_ROOT / f"h{horizon}" / dataset_name / mode / model_name / run_tag
    result_path = output_dir / f"{run_name}_results.npy"
    summary_path = output_dir / f"{run_name}_summary.json"

    if args.skip_existing and result_path.exists() and summary_path.exists():
        print(f"跳过已存在结果: {run_name}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    train_loader, val_loader, test_loader = create_loaders(args, dataset_name, horizon, mode)
    train_dataset = unwrap_dataset(train_loader.dataset)
    input_size = train_dataset.input_size
    target_idx = train_dataset.target_idx

    model = MODEL_BUILDERS[model_name](input_size, horizon)
    trainer = Trainer(model, device=args.device, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed)

    started = time.time()
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=args.epochs,
        patience=args.patience,
        save_dir=str(ROOT / "checkpoints" / "univariate_multivariate"),
        model_name=run_name,
        log_dir=None if args.no_tensorboard else str(ROOT / "runs" / "univariate_multivariate"),
    )
    train_seconds = time.time() - started

    predictions, targets = trainer.predict(test_loader)
    metrics = trainer.compute_metrics(predictions, targets, target_idx=target_idx)

    result = {
        "dataset": dataset_name,
        "horizon": horizon,
        "model": model_name,
        "feature_mode": mode,
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
        "data_dir": str(Path(args.data_dir)),
        "sample_limit": args.sample_limit,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "input_size": input_size,
        "original_input_size": train_dataset.original_input_size,
        "target_idx": target_idx,
        "original_target_idx": train_dataset.original_target_idx,
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
        json.dumps(to_jsonable({key: value for key, value in result.items() if key != "history"}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved result: {result_path.relative_to(ROOT)}")
    return result


def flatten_summary(summary: dict) -> dict[str, object]:
    metrics = summary["metrics"]
    return {
        "dataset": summary["dataset"],
        "horizon": summary["horizon"],
        "model": summary["model"],
        "feature_mode": summary["feature_mode"],
        "run_tag": summary.get("run_tag", ""),
        "MSE": metrics.get("MSE"),
        "MAE": metrics.get("MAE"),
        "MAPE": metrics.get("MAPE"),
        "R2": metrics.get("R2"),
        "MSE_target": metrics.get("MSE_target"),
        "MAE_target": metrics.get("MAE_target"),
        "MAPE_target": metrics.get("MAPE_target"),
        "R2_target": metrics.get("R2_target"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_val_r2": summary.get("best_val_r2"),
        "trained_epochs": summary.get("trained_epochs"),
        "seed": summary.get("seed"),
        "sample_limit": summary.get("sample_limit"),
        "train_samples": summary.get("train_samples"),
        "val_samples": summary.get("val_samples"),
        "test_samples": summary.get("test_samples"),
        "input_size": summary.get("input_size"),
        "model_params": summary.get("model_params"),
        "train_time_seconds": summary.get("train_time_seconds"),
        "device": summary.get("device"),
    }


def build_delta_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset"], row["horizon"], row["model"])][str(row["feature_mode"])] = row

    delta_rows: list[dict[str, object]] = []
    for (dataset, horizon, model), modes in sorted(grouped.items()):
        if "univariate" not in modes or "multivariate" not in modes:
            continue
        uni = modes["univariate"]
        multi = modes["multivariate"]
        delta_rows.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "model": model,
                "univariate_MSE_target": uni["MSE_target"],
                "multivariate_MSE_target": multi["MSE_target"],
                "delta_MSE_target": float(uni["MSE_target"]) - float(multi["MSE_target"]),
                "delta_MSE_target_pct": (
                    (float(uni["MSE_target"]) - float(multi["MSE_target"]))
                    / (float(multi["MSE_target"]) + 1e-8)
                    * 100
                ),
                "univariate_MAE_target": uni["MAE_target"],
                "multivariate_MAE_target": multi["MAE_target"],
                "delta_MAE_target": float(uni["MAE_target"]) - float(multi["MAE_target"]),
                "univariate_R2_target": uni["R2_target"],
                "multivariate_R2_target": multi["R2_target"],
                "delta_R2_target": float(uni["R2_target"]) - float(multi["R2_target"]),
            }
        )
    return delta_rows


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_table(rows: list[dict[str, object]], csv_path: Path, md_path: Path) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row[column]) for column in columns) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_run(run_tag: str) -> tuple[Path, Path, Path, Path]:
    summary_paths_found = sorted((RESULT_ROOT).rglob(f"*_{run_tag}_summary.json"))
    rows = [
        flatten_summary(json.loads(path.read_text(encoding="utf-8")))
        for path in summary_paths_found
    ]
    rows = sorted(rows, key=lambda row: (row["dataset"], int(row["horizon"]), row["model"], row["feature_mode"]))
    csv_path, md_path = summary_paths(run_tag)
    write_table(rows, csv_path, md_path)

    delta_rows = build_delta_rows(rows)
    delta_csv = csv_path.with_name(csv_path.stem + "_delta.csv")
    delta_md = md_path.with_name(md_path.stem + "_delta.md")
    write_table(delta_rows, delta_csv, delta_md)

    print(f"Saved comparison: {csv_path.relative_to(ROOT)}")
    print(f"Saved comparison: {md_path.relative_to(ROOT)}")
    print(f"Saved delta: {delta_csv.relative_to(ROOT)}")
    print(f"Saved delta: {delta_md.relative_to(ROOT)}")
    print(f"Rows: detail={len(rows)}, delta={len(delta_rows)}")
    return csv_path, md_path, delta_csv, delta_md


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
    modes = parse_csv_list(args.modes)
    unknown_modes = sorted(set(modes) - {"univariate", "multivariate"})
    if unknown_modes:
        raise ValueError(f"Unknown feature modes: {unknown_modes}")

    for dataset_name in datasets:
        for horizon in horizons:
            for model_name in models:
                for mode in modes:
                    run_one(args, dataset_name, horizon, model_name, mode)

    summarize_run(args.run_tag or "default")


if __name__ == "__main__":
    main()
