"""LSTM hyperparameter tuning via grid search.

Usage:
    python scripts/tune_lstm.py
    python scripts/tune_lstm.py --config configs/lstm_search.json
    python scripts/tune_lstm.py --datasets ETTh1 --horizons 96 --dry-run

Results are saved to:
    test_results/h{horizon}/{dataset}/lstm/{run_name}_results.npy
    test_results/h{horizon}/{dataset}/lstm/{run_name}_summary.json
    test_results/h{horizon}/{dataset}/lstm/lstm_search_summary.csv
    test_results/h{horizon}/{dataset}/lstm/lstm_search_summary.md
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import LSTMModel, TimeSeriesDataset, Trainer  # noqa: E402
from models.trainer import resolve_device  # noqa: E402

# ---------------------------------------------------------------------------
# Default search space
# ---------------------------------------------------------------------------
DEFAULT_SEARCH_SPACE: dict[str, list] = {
    "hidden_size":    [128, 256],
    "num_layers":     [1, 2],
    "dropout":        [0.1, 0.2],
    "lr":             [3e-4, 1e-3],
    "weight_decay":   [0.0, 1e-5],
}

DEFAULT_FIXED: dict = {
    "epochs":       25,
    "patience":     5,
    "batch_size":   32,
    "seed":         216,
    "sample_limit": 0,
    "device":       "auto",
    "num_workers":  0,
}

DEFAULT_DATASETS = ["ETTh1"]
DEFAULT_HORIZONS = [96]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_int_csv(value: str) -> list[int]:
    return [int(v) for v in parse_csv(value)]


def maybe_subset(dataset, limit: int):
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def load_data(dataset_name: str, horizon: int, batch_size: int,
              sample_limit: int, num_workers: int, data_dir: Path):
    train_ds = TimeSeriesDataset(data_dir, dataset_name, horizon, "train")
    val_ds = TimeSeriesDataset(data_dir, dataset_name, horizon, "val")
    test_ds = TimeSeriesDataset(data_dir, dataset_name, horizon, "test")

    train_ds = maybe_subset(train_ds, sample_limit)
    val_ds = maybe_subset(val_ds, sample_limit)
    test_ds = maybe_subset(test_ds, sample_limit)

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


def to_jsonable(v):
    if isinstance(v, dict):
        return {k: to_jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [to_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Core: run one trial (with skip-existing support)
# ---------------------------------------------------------------------------
def run_trial(
    dataset_name: str,
    horizon: int,
    trial_params: dict,
    fixed: dict,
    data_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    skip_existing: bool = True,
) -> dict | None:
    """Train one LSTM config and return its summary dict."""
    seed = fixed["seed"]
    set_seed(seed)

    run_name = (
        f"{dataset_name}_h{horizon}_lstm_"
        f"h{trial_params['hidden_size']}_l{trial_params['num_layers']}_"
        f"dp{str(trial_params['dropout']).replace('.', '')}_"
        f"lr{trial_params['lr']}_wd{trial_params['weight_decay']}"
    )

    # Skip existing
    if skip_existing and not dry_run:
        results_path = output_dir / f"{run_name}_results.npy"
        summary_path = output_dir / f"{run_name}_summary.json"
        if results_path.exists() and summary_path.exists():
            print(f"  [SKIP] {run_name} (已有结果)")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            return summary

    # Create loaders to determine input_size
    train_loader, val_loader, test_loader = load_data(
        dataset_name, horizon,
        fixed["batch_size"], fixed["sample_limit"],
        fixed["num_workers"], data_dir,
    )
    base_ds = (train_loader.dataset.dataset
               if isinstance(train_loader.dataset, Subset)
               else train_loader.dataset)
    input_size = base_ds.input_size
    target_idx = base_ds.target_idx

    # Build model
    model = LSTMModel(
        input_size=input_size,
        hidden_size=trial_params["hidden_size"],
        num_layers=trial_params["num_layers"],
        dropout=trial_params["dropout"],
        horizon=horizon,
    )

    if dry_run:
        print(f"  [DRY-RUN] {run_name}  params={count_params(model):,}")
        return {}

    trainer = Trainer(
        model,
        device=fixed["device"],
        lr=trial_params["lr"],
        weight_decay=trial_params["weight_decay"],
        seed=seed,
    )

    started = time.time()
    history = trainer.train(
        train_loader, val_loader,
        epochs=fixed["epochs"],
        patience=fixed["patience"],
        save_dir=str(output_dir / "checkpoints"),
        model_name=run_name,
        log_dir=None,
    )
    train_sec = time.time() - started

    predictions, targets = trainer.predict(test_loader)
    metrics = trainer.compute_metrics(predictions, targets, target_idx=target_idx)

    summary = {
        "dataset": dataset_name,
        "horizon": horizon,
        "model": "lstm",
        "run_name": run_name,
        # trial hyperparams
        "hidden_size": trial_params["hidden_size"],
        "num_layers": trial_params["num_layers"],
        "dropout": trial_params["dropout"],
        "lr": trial_params["lr"],
        "weight_decay": trial_params["weight_decay"],
        # fixed
        "epochs": fixed["epochs"],
        "patience": fixed["patience"],
        "batch_size": fixed["batch_size"],
        "seed": seed,
        "device": trainer.device,
        # data
        "sample_limit": fixed["sample_limit"],
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "input_size": input_size,
        "target_idx": target_idx,
        # model
        "model_params": count_params(model),
        # training
        "trained_epochs": len(history["train_losses"]),
        "best_epoch": int(np.argmin(history["val_losses"]) + 1),
        "train_time_seconds": train_sec,
        "best_val_loss": history["best_val_loss"],
        "best_val_r2": history["best_val_r2"],
        # test metrics
        "metrics": metrics,
    }

    # Save per-trial files
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{run_name}_results.npy",
            {**summary, "history": history}, allow_pickle=True)
    (output_dir / f"{run_name}_summary.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------
SUMMARY_COLS = [
    "run_name", "hidden_size", "num_layers", "dropout",
    "lr", "weight_decay", "model_params",
    "trained_epochs", "best_epoch", "train_time_seconds",
    "best_val_loss", "best_val_r2",
    "MSE", "MAE", "R2", "MSE_target", "MAE_target", "R2_target",
]


def write_summary(summaries: list[dict], output_dir: Path) -> None:
    """Write CSV and Markdown summary tables sorted by validation loss."""
    rows = []
    for s in summaries:
        m = s.get("metrics", {})
        row = {}
        for k in SUMMARY_COLS:
            if k in s:
                row[k] = s[k]
            elif k in m:
                row[k] = m[k]
            else:
                row[k] = ""
        rows.append(row)

    rows.sort(key=lambda r: r.get("best_val_loss", float("inf")))

    # CSV
    csv_path = output_dir / "lstm_search_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV summary: {csv_path}")

    # Markdown
    md_path = output_dir / "lstm_search_summary.md"
    lines = [
        "# LSTM Hyperparameter Search Summary",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Total trials**: {len(rows)}",
        "",
        "| " + " | ".join(SUMMARY_COLS) + " |",
        "| " + " | ".join(["---"] * len(SUMMARY_COLS)) + " |",
    ]
    for r in rows:
        cells = []
        for k in SUMMARY_COLS:
            v = r[k]
            if isinstance(v, float):
                cells.append(f"{v:.6f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"MD   summary: {md_path}")

    # Print top-5 to console
    print("\n--- Top-5 by val_loss ---")
    for i, r in enumerate(rows[:5], 1):
        print(f"  {i}. {r['run_name']}")
        print(f"     MSE={r['MSE']:.6f}  MAE={r['MAE']:.6f}  R2={r['R2']:.4f}  "
              f"val_loss={r['best_val_loss']:.6f}  epochs={r['trained_epochs']}  "
              f"time={r['train_time_seconds']:.0f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_search_grid(space: dict[str, list]) -> list[dict]:
    keys = list(space.keys())
    for k in keys:
        if not isinstance(space[k], list):
            space[k] = [space[k]]
    combos = list(itertools.product(*[space[k] for k in keys]))
    return [dict(zip(keys, c)) for c in combos]


def main():
    parser = argparse.ArgumentParser(
        description="LSTM hyperparameter grid search")
    parser.add_argument("--config", default="",
                        help="JSON config with search_space / fixed / datasets / horizons")
    parser.add_argument("--datasets", default=None,
                        help="Comma-separated dataset names")
    parser.add_argument("--horizons", default=None,
                        help="Comma-separated horizon values")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print trial configs without training")
    parser.add_argument("--max-trials", type=int, default=0,
                        help="Limit number of trials (0 = all)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Do not skip existing results")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    search_space = cfg.get("search_space", DEFAULT_SEARCH_SPACE)
    fixed = {**DEFAULT_FIXED, **cfg.get("fixed", {})}
    datasets = (parse_csv(args.datasets) if args.datasets
                else cfg.get("datasets", DEFAULT_DATASETS))
    horizons = (parse_int_csv(args.horizons) if args.horizons
                else cfg.get("horizons", DEFAULT_HORIZONS))
    skip_existing = not args.no_skip and fixed.get("skip_existing", True)
    data_dir = Path(fixed.get("data_dir", ROOT / "data" / "processed"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    grid = build_search_grid(search_space)
    total = len(grid) * len(datasets) * len(horizons)
    print(f"Search space: {len(grid)} configs x {len(datasets)} datasets x "
          f"{len(horizons)} horizons = {total} trials")
    if args.max_trials and args.max_trials < len(grid):
        grid = grid[:args.max_trials]
        print(f"  (limited to {args.max_trials} configs per dataset/horizon)")
    print(f"Skip existing: {skip_existing}")

    device_str = resolve_device(fixed["device"])
    print(f"Device: {device_str}")
    print(f"Seed: {fixed['seed']}")
    print()

    for dataset_name in datasets:
        for horizon in horizons:
            output_dir = (ROOT / "test_results" / f"h{horizon}"
                          / dataset_name / "lstm")
            print(f"{'='*60}")
            print(f"Dataset: {dataset_name}  Horizon: {horizon}")
            print(f"Output:  {output_dir}")
            print(f"{'='*60}")

            all_summaries = []
            skipped = 0
            for i, trial in enumerate(grid, 1):
                print(f"\n[{i}/{len(grid)}] "
                      + "  ".join(f"{k}={v}" for k, v in trial.items()))
                summary = run_trial(
                    dataset_name, horizon, trial, fixed,
                    data_dir, output_dir, dry_run=args.dry_run,
                    skip_existing=skip_existing,
                )
                if summary:
                    if "metrics" not in summary and not args.dry_run:
                        # loaded from existing file, has metrics at top level
                        pass
                    all_summaries.append(summary)

            if all_summaries and not args.dry_run:
                write_summary(all_summaries, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
