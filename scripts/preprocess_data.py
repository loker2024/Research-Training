"""Preprocess selected long-term forecasting datasets.

This script mirrors the data-preparation notebook and creates compressed NPZ
files under data/processed/{dataset}/h{horizon}/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = DATA_DIR / "processed"

DATASET_CONFIG = {
    "ETTh1": {
        "file": "ETTh1.csv",
        "date_col": "date",
        "target_col": "OT",
        "freq": "h",
        "split": [12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24],
    },
    "ETTm1": {
        "file": "ETTm1.csv",
        "date_col": "date",
        "target_col": "OT",
        "freq": "15min",
        "split": [12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4],
    },
    "ECL": {
        "file": "ECL.csv",
        "date_col": "date",
        "target_col": None,
        "freq": "h",
        "split": [0.7, 0.1, 0.2],
    },
}


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def detect_columns(path: Path) -> tuple[str, list[str]]:
    df = pd.read_csv(path, nrows=1)
    columns = list(df.columns)
    date_col = next(
        (col for col in columns if col.lower() in {"date", "timestamp", "time", "datetime"}),
        columns[0],
    )
    num_cols = [col for col in columns if col != date_col]
    return date_col, num_cols


def load_dataset(name: str) -> tuple[pd.DataFrame, dict, list[str]]:
    cfg = dict(DATASET_CONFIG[name])
    path = DATA_DIR / cfg["file"]
    if not path.exists():
        raise FileNotFoundError(f"Missing raw dataset: {path}")

    if cfg["target_col"] is None:
        _, num_cols = detect_columns(path)
        cfg["target_col"] = num_cols[-1]

    df = pd.read_csv(path)
    df[cfg["date_col"]] = pd.to_datetime(df[cfg["date_col"]])
    df = df.sort_values(cfg["date_col"]).reset_index(drop=True)
    num_cols = [col for col in df.columns if col != cfg["date_col"]]
    return df, cfg, num_cols


def split_dataset(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    split = cfg["split"]
    if isinstance(split[0], float):
        train_end = int(n_rows * split[0])
        val_end = int(n_rows * (split[0] + split[1]))
    else:
        train_end = min(split[0], n_rows)
        val_end = min(split[0] + split[1], n_rows)

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def normalize_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    train_data = train_df[num_cols].values.astype(np.float32)
    scaler.fit(train_data)
    train_norm = scaler.transform(train_data).astype(np.float32)
    val_norm = scaler.transform(val_df[num_cols].values.astype(np.float32)).astype(np.float32)
    test_norm = scaler.transform(test_df[num_cols].values.astype(np.float32)).astype(np.float32)
    return train_norm, val_norm, test_norm, scaler


def create_sliding_windows(
    data: np.ndarray,
    lookback: int,
    horizon: int,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = len(data) - lookback - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough rows for lookback={lookback}, horizon={horizon}: rows={len(data)}"
        )
    if max_samples is not None and max_samples > 0:
        n_samples = min(n_samples, max_samples)

    windows = np.lib.stride_tricks.sliding_window_view(data, lookback + horizon, axis=0)
    windows = np.moveaxis(windows, 2, 1)
    windows = windows[:n_samples].astype(np.float32, copy=False)
    return windows[:, :lookback, :], windows[:, lookback:, :]


def save_split(output_dir: Path, split: str, x_data: np.ndarray, y_data: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"{split}.npz", X=x_data, Y=y_data)


def preprocess_dataset(
    name: str,
    horizons: list[int],
    lookback: int,
    force: bool,
    output_dir: Path,
    max_samples_per_split: int | None,
) -> dict:
    df, cfg, num_cols = load_dataset(name)
    train_df, val_df, test_df = split_dataset(df, cfg)
    train_norm, val_norm, test_norm, scaler = normalize_data(train_df, val_df, test_df, num_cols)

    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dataset_dir / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

    target_idx = num_cols.index(cfg["target_col"])
    meta = {
        "dataset": name,
        "file": cfg["file"],
        "freq": cfg["freq"],
        "lookback": lookback,
        "horizons": horizons,
        "n_features": len(num_cols),
        "feature_cols": num_cols,
        "target_col": cfg["target_col"],
        "target_idx": target_idx,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "max_samples_per_split": max_samples_per_split,
    }
    (dataset_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for horizon in horizons:
        horizon_dir = dataset_dir / f"h{horizon}"
        expected = [horizon_dir / f"{split}.npz" for split in ("train", "val", "test")]
        if not force and all(path.exists() for path in expected):
            print(f"[{name}] h{horizon}: already exists")
            continue

        train_x, train_y = create_sliding_windows(
            train_norm, lookback, horizon, max_samples=max_samples_per_split
        )
        val_x, val_y = create_sliding_windows(
            val_norm, lookback, horizon, max_samples=max_samples_per_split
        )
        test_x, test_y = create_sliding_windows(
            test_norm, lookback, horizon, max_samples=max_samples_per_split
        )
        save_split(horizon_dir, "train", train_x, train_y)
        save_split(horizon_dir, "val", val_x, val_y)
        save_split(horizon_dir, "test", test_x, test_y)
        print(
            f"[{name}] h{horizon}: train={train_x.shape}, "
            f"val={val_x.shape}, test={test_x.shape}"
        )

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess forecasting datasets")
    parser.add_argument("--datasets", default="ETTh1,ETTm1,ECL")
    parser.add_argument("--horizons", default="24,48,96,168,336")
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    datasets = parse_csv_list(args.datasets)
    horizons = parse_int_list(args.horizons)
    for dataset in datasets:
        if dataset not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    max_samples = args.max_samples_per_split if args.max_samples_per_split > 0 else None
    for dataset in datasets:
        preprocess_dataset(dataset, horizons, args.lookback, args.force, output_dir, max_samples)


if __name__ == "__main__":
    main()
