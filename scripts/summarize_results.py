"""Summarize saved experiment result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def load_rows(
    results_dir: Path,
    datasets: set[str],
    horizons: set[int],
    models: set[str],
    run_tags: set[str],
) -> list[dict]:
    rows = []
    for path in sorted(results_dir.rglob("*_summary.json")):
        if path.is_symlink():
            continue
        if "/by_horizon/" in path.as_posix() or "/by_model/" in path.as_posix():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset = data["dataset"]
        horizon = int(data["horizon"])
        model = data["model"]
        run_tag = data.get("run_tag", "")
        if datasets and dataset not in datasets:
            continue
        if horizons and horizon not in horizons:
            continue
        if models and model not in models:
            continue
        if run_tags and run_tag not in run_tags:
            continue

        metrics = data["metrics"]
        rows.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "model": model,
                "run_tag": run_tag,
                "MSE": metrics["MSE"],
                "MAE": metrics["MAE"],
                "R2": metrics["R2"],
                "MSE_target": metrics.get("MSE_target"),
                "MAE_target": metrics.get("MAE_target"),
                "R2_target": metrics.get("R2_target"),
                "best_val_loss": data["best_val_loss"],
                "best_val_r2": data["best_val_r2"],
                "trained_epochs": data["trained_epochs"],
                "seed": data.get("seed"),
                "sample_limit": data.get("sample_limit"),
                "train_samples": data["train_samples"],
                "val_samples": data["val_samples"],
                "test_samples": data["test_samples"],
                "model_params": data["model_params"],
                "train_time_seconds": data["train_time_seconds"],
                "device": data["device"],
                "data_dir": data.get("data_dir"),
            }
        )
    return rows


def format_markdown_table(df: pd.DataFrame) -> str:
    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
        else:
            display_df[column] = display_df[column].map(lambda value: "" if pd.isna(value) else str(value))

    headers = list(display_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display_df.iterrows():
        lines.append("| " + " | ".join(row[headers].tolist()) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--horizons", default="")
    parser.add_argument("--models", default="")
    parser.add_argument("--run-tags", default="")
    parser.add_argument("--output-prefix", default="summary")
    args = parser.parse_args()

    rows = load_rows(
        RESULTS_DIR,
        set(parse_csv_list(args.datasets)),
        set(parse_int_list(args.horizons)) if args.horizons else set(),
        set(parse_csv_list(args.models)),
        set(parse_csv_list(args.run_tags)),
    )
    if not rows:
        raise FileNotFoundError("No matching *_summary.json files found")

    df = pd.DataFrame(rows).sort_values(["dataset", "horizon", "model"])
    csv_path = RESULTS_DIR / f"{args.output_prefix}.csv"
    md_path = RESULTS_DIR / f"{args.output_prefix}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(format_markdown_table(df), encoding="utf-8")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")
    print(df[["dataset", "horizon", "model", "MSE", "MAE", "R2", "trained_epochs"]].to_string(index=False))


if __name__ == "__main__":
    main()
