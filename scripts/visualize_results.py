"""Generate report-ready figures for step 6 analysis.

The script only reads existing result tables and checkpoints. It does not train
or overwrite experiment outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import TimeSeriesDataset  # noqa: E402
from scripts.run_experiments import MODEL_BUILDERS  # noqa: E402


RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"

MODEL_ORDER = ["lstm", "transformer", "informer", "autoformer", "patchtst"]
MODEL_LABELS = {
    "lstm": "LSTM",
    "transformer": "Transformer",
    "informer": "Informer",
    "autoformer": "Autoformer",
    "patchtst": "PatchTST",
}
DATASET_ORDER = ["ETTh1", "ETTm1"]
METRICS = ["MSE", "MAE", "R2"]

PREDICTION_CASES = [
    ("ETTh1", 96, "patchtst"),
    ("ETTh1", 336, "patchtst"),
    ("ETTm1", 96, "autoformer"),
    ("ETTm1", 336, "autoformer"),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def read_inputs(formal_path: Path, ablation_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not formal_path.exists():
        raise FileNotFoundError(f"Missing formal result table: {formal_path}")
    if not ablation_path.exists():
        raise FileNotFoundError(f"Missing ablation comparison table: {ablation_path}")

    formal = pd.read_csv(formal_path)
    ablation = pd.read_csv(ablation_path)
    formal = formal[formal["run_tag"].eq("formal_seed42")].copy()
    formal["model"] = pd.Categorical(formal["model"], MODEL_ORDER, ordered=True)
    formal["dataset"] = pd.Categorical(formal["dataset"], DATASET_ORDER, ordered=True)
    formal = formal.sort_values(["dataset", "horizon", "model"])
    return formal, ablation


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_metric_trends(formal: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "formal_metric_trends.png"
    fig, axes = plt.subplots(len(METRICS), len(DATASET_ORDER), figsize=(13, 10), sharex=True)
    for row_idx, metric in enumerate(METRICS):
        for col_idx, dataset in enumerate(DATASET_ORDER):
            ax = axes[row_idx, col_idx]
            subset = formal[formal["dataset"].astype(str).eq(dataset)]
            for model in MODEL_ORDER:
                model_df = subset[subset["model"].astype(str).eq(model)]
                if model_df.empty:
                    continue
                ax.plot(
                    model_df["horizon"],
                    model_df[metric],
                    marker="o",
                    linewidth=1.8,
                    label=MODEL_LABELS[model],
                )
            ax.set_title(f"{dataset} {metric}")
            ax.set_xlabel("Horizon")
            ax.set_ylabel(metric)
            ax.set_xticks(sorted(subset["horizon"].unique()))
            if row_idx == 0 and col_idx == len(DATASET_ORDER) - 1:
                ax.legend(loc="best", fontsize=8)
    save_figure(path)
    return path


def plot_best_models(formal: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "formal_best_model_by_horizon.png"
    best = formal.loc[formal.groupby(["dataset", "horizon"], observed=True)["MSE"].idxmin()].copy()
    best["label"] = best["model"].astype(str).map(MODEL_LABELS)

    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(12, 4), sharey=True)
    colors = {
        "lstm": "#7f7f7f",
        "transformer": "#d62728",
        "informer": "#ff7f0e",
        "autoformer": "#2ca02c",
        "patchtst": "#1f77b4",
    }
    for ax, dataset in zip(axes, DATASET_ORDER):
        subset = best[best["dataset"].astype(str).eq(dataset)]
        bar_colors = [colors[str(model)] for model in subset["model"]]
        ax.bar(subset["horizon"].astype(str), subset["MSE"], color=bar_colors)
        ax.set_title(f"{dataset} best MSE")
        ax.set_xlabel("Horizon")
        ax.set_ylabel("MSE")
        for _, row in subset.iterrows():
            ax.text(
                str(row["horizon"]),
                row["MSE"],
                row["label"],
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=20,
            )
    save_figure(path)
    return path


def plot_complexity(formal: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "formal_complexity_tradeoff.png"
    complexity = (
        formal.groupby(["model"], observed=True)
        .agg(
            mean_mse=("MSE", "mean"),
            mean_params=("model_params", "mean"),
            mean_train_time=("train_time_seconds", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = np.clip(complexity["mean_train_time"] / complexity["mean_train_time"].max() * 1100, 180, 1100)
    ax.scatter(complexity["mean_params"], complexity["mean_mse"], s=sizes, alpha=0.72)
    for _, row in complexity.iterrows():
        model = str(row["model"])
        ax.annotate(
            MODEL_LABELS.get(model, model),
            (row["mean_params"], row["mean_mse"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title("Average Accuracy vs Complexity")
    ax.set_xlabel("Average trainable parameters")
    ax.set_ylabel("Average MSE")
    save_figure(path)
    return path


def plot_ablation_delta(ablation: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "ablation_delta_mse_pct.png"
    df = ablation.copy()
    df["case"] = (
        df["dataset"].astype(str)
        + " h"
        + df["horizon"].astype(str)
        + " "
        + df["ablation_model"].astype(str)
    )
    df = df.sort_values(["dataset", "horizon", "base_model", "ablation_model"])

    fig, ax = plt.subplots(figsize=(12, 6))
    values = df["delta_MSE_pct"].astype(float)
    colors = ["#2ca02c" if value < 0 else "#d62728" for value in values]
    ax.barh(df["case"], values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Ablation Impact on MSE")
    ax.set_xlabel("Delta MSE (%) vs original model")
    ax.set_ylabel("")
    save_figure(path)
    return path


def load_target_scaler(dataset: str, target_idx: int) -> tuple[float, float]:
    scaler = np.load(DATA_DIR / dataset / "scaler.npz", allow_pickle=True)
    mean = float(scaler["mean"][target_idx])
    scale = float(scaler["scale"][target_idx])
    return mean, scale


def inverse_target(values: np.ndarray, dataset: str, target_idx: int) -> np.ndarray:
    mean, scale = load_target_scaler(dataset, target_idx)
    return values * scale + mean


def load_prediction_batch(dataset: str, horizon: int, model_name: str, batch_size: int, device: str) -> dict:
    test_dataset = TimeSeriesDataset(DATA_DIR, dataset, horizon, "test")
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    input_size = test_dataset.input_size
    target_idx = test_dataset.target_idx

    model = MODEL_BUILDERS[model_name](input_size, horizon)
    checkpoint = CHECKPOINT_DIR / f"{dataset}_h{horizon}_{model_name}_formal_seed42.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    batch_x, batch_y = next(iter(loader))
    batch_x = batch_x.to(device)
    with torch.no_grad():
        pred_y = model(batch_x).cpu().numpy()

    targets = batch_y.numpy()
    pred_target = inverse_target(pred_y[:, :, target_idx], dataset, target_idx)
    true_target = inverse_target(targets[:, :, target_idx], dataset, target_idx)
    return {
        "dataset": dataset,
        "horizon": horizon,
        "model": model_name,
        "target_idx": target_idx,
        "prediction_shape": list(pred_y.shape),
        "target_shape": list(targets.shape),
        "pred_target": pred_target,
        "true_target": true_target,
    }


def plot_prediction_case(case: dict, sample_idx: int) -> tuple[Path, Path]:
    dataset = case["dataset"]
    horizon = int(case["horizon"])
    model_name = case["model"]
    pred = case["pred_target"][sample_idx]
    true = case["true_target"][sample_idx]
    residual = pred - true
    steps = np.arange(1, horizon + 1)
    stem = f"prediction_{dataset}_h{horizon}_{model_name}"

    curve_path = FIGURES_DIR / f"{stem}.png"
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(steps, true, label="True", linewidth=2)
    ax.plot(steps, pred, label="Prediction", linewidth=1.8, linestyle="--")
    ax.set_title(f"{dataset} h{horizon} {MODEL_LABELS.get(model_name, model_name)} Target Forecast")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Target value")
    ax.legend(loc="best")
    save_figure(curve_path)

    residual_path = FIGURES_DIR / f"residual_{dataset}_h{horizon}_{model_name}.png"
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot(steps, residual, color="#d62728", linewidth=1.5)
    ax.set_title(f"{dataset} h{horizon} {MODEL_LABELS.get(model_name, model_name)} Residual")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Prediction - True")
    save_figure(residual_path)
    return curve_path, residual_path


def plot_predictions(batch_size: int, sample_idx: int, device: str) -> tuple[list[Path], list[dict]]:
    paths: list[Path] = []
    summaries: list[dict] = []
    for dataset, horizon, model_name in PREDICTION_CASES:
        case = load_prediction_batch(dataset, horizon, model_name, batch_size, device)
        curve_path, residual_path = plot_prediction_case(case, sample_idx)
        paths.extend([curve_path, residual_path])
        summaries.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "model": model_name,
                "target_idx": case["target_idx"],
                "prediction_shape": "x".join(map(str, case["prediction_shape"])),
                "target_shape": "x".join(map(str, case["target_shape"])),
                "prediction_figure": str(curve_path.relative_to(ROOT)).replace("\\", "/"),
                "residual_figure": str(residual_path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    return paths, summaries


def write_manifest(paths: list[Path], prediction_summaries: list[dict]) -> Path:
    manifest_path = FIGURES_DIR / "manifest.json"
    payload = {
        "figures": [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths],
        "prediction_cases": prediction_summaries,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(prediction_summaries).to_csv(FIGURES_DIR / "prediction_samples_summary.csv", index=False)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate step 6 result visualizations")
    parser.add_argument("--formal", default=str(RESULTS_DIR / "formal_seed42_all.csv"))
    parser.add_argument("--ablation", default=str(RESULTS_DIR / "ablation_seed42_vs_formal_comparison.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="Inference device for checkpoint plots; default keeps this read-only pass portable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    formal, ablation = read_inputs(Path(args.formal), Path(args.ablation))
    figure_paths = [
        plot_metric_trends(formal),
        plot_best_models(formal),
        plot_complexity(formal),
        plot_ablation_delta(ablation),
    ]
    prediction_paths, prediction_summaries = plot_predictions(args.batch_size, args.sample_idx, args.device)
    figure_paths.extend(prediction_paths)
    manifest_path = write_manifest(figure_paths, prediction_summaries)

    print(f"Saved {len(figure_paths)} figures under {FIGURES_DIR}")
    print(f"Saved manifest: {manifest_path}")
    for summary in prediction_summaries:
        print(
            f"{summary['dataset']} h{summary['horizon']} {summary['model']}: "
            f"prediction_shape={summary['prediction_shape']}, target_shape={summary['target_shape']}"
        )


if __name__ == "__main__":
    main()
