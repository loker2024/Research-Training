"""Build supplemental MAPE tables from formal seed-42 result summaries."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
CSV_DIR = RESULTS_DIR / "v1_csv" / "formal"
MD_DIR = RESULTS_DIR / "v1_md" / "formal"

DATASETS = ("ETTh1", "ETTm1")
HORIZONS = (24, 48, 96, 168, 336)
MODELS = ("autoformer", "informer", "lstm", "patchtst", "transformer")
RUN_TAG = "formal_seed42"


DETAIL_COLUMNS = [
    "dataset",
    "horizon",
    "model",
    "run_tag",
    "MAPE",
    "MAPE_target",
    "MSE",
    "MAE",
    "R2",
    "MSE_target",
    "MAE_target",
    "R2_target",
]

AGG_COLUMNS = [
    "dataset",
    "model",
    "avg_MAPE",
    "avg_MAPE_target",
    "min_MAPE_target",
    "min_MAPE_target_horizon",
    "max_MAPE_target",
    "max_MAPE_target_horizon",
    "num_runs",
]


def load_detail_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    missing: list[str] = []

    for dataset in DATASETS:
        for horizon in HORIZONS:
            for model in MODELS:
                pattern = (
                    RESULTS_DIR
                    / f"h{horizon}"
                    / dataset
                    / model
                    / RUN_TAG
                    / "*_summary.json"
                )
                paths = sorted(pattern.parent.glob(pattern.name))
                if len(paths) != 1:
                    missing.append(str(pattern))
                    continue

                data = json.loads(paths[0].read_text(encoding="utf-8"))
                metrics = data["metrics"]
                rows.append(
                    {
                        "dataset": data["dataset"],
                        "horizon": int(data["horizon"]),
                        "model": data["model"],
                        "run_tag": data.get("run_tag", ""),
                        "MAPE": metrics.get("MAPE"),
                        "MAPE_target": metrics.get("MAPE_target"),
                        "MSE": metrics.get("MSE"),
                        "MAE": metrics.get("MAE"),
                        "R2": metrics.get("R2"),
                        "MSE_target": metrics.get("MSE_target"),
                        "MAE_target": metrics.get("MAE_target"),
                        "R2_target": metrics.get("R2_target"),
                    }
                )

    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing or ambiguous formal summaries:\n{joined}")

    return sorted(rows, key=lambda row: (row["dataset"], row["horizon"], row["model"]))


def build_aggregate_rows(detail_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        groups[(str(row["dataset"]), str(row["model"]))].append(row)

    aggregate_rows: list[dict[str, object]] = []
    for (dataset, model), rows in sorted(groups.items()):
        mape_targets = [float(row["MAPE_target"]) for row in rows]
        min_row = min(rows, key=lambda row: float(row["MAPE_target"]))
        max_row = max(rows, key=lambda row: float(row["MAPE_target"]))
        aggregate_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "avg_MAPE": mean(float(row["MAPE"]) for row in rows),
                "avg_MAPE_target": mean(mape_targets),
                "min_MAPE_target": float(min_row["MAPE_target"]),
                "min_MAPE_target_horizon": int(min_row["horizon"]),
                "max_MAPE_target": float(max_row["MAPE_target"]),
                "max_MAPE_target_horizon": int(max_row["horizon"]),
                "num_runs": len(rows),
            }
        )

    all_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        all_groups[str(row["model"])].append(row)
    for model, rows in sorted(all_groups.items()):
        mape_targets = [float(row["MAPE_target"]) for row in rows]
        min_row = min(rows, key=lambda row: float(row["MAPE_target"]))
        max_row = max(rows, key=lambda row: float(row["MAPE_target"]))
        aggregate_rows.append(
            {
                "dataset": "ALL",
                "model": model,
                "avg_MAPE": mean(float(row["MAPE"]) for row in rows),
                "avg_MAPE_target": mean(mape_targets),
                "min_MAPE_target": float(min_row["MAPE_target"]),
                "min_MAPE_target_horizon": int(min_row["horizon"]),
                "max_MAPE_target": float(max_row["MAPE_target"]),
                "max_MAPE_target_horizon": int(max_row["horizon"]),
                "num_runs": len(rows),
            }
        )

    return aggregate_rows


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row[column]) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detail_rows = load_detail_rows()
    aggregate_rows = build_aggregate_rows(detail_rows)

    outputs = [
        (CSV_DIR / "formal_seed42_mape.csv", detail_rows, DETAIL_COLUMNS),
        (MD_DIR / "formal_seed42_mape.md", detail_rows, DETAIL_COLUMNS),
        (CSV_DIR / "formal_seed42_mape_by_model.csv", aggregate_rows, AGG_COLUMNS),
        (MD_DIR / "formal_seed42_mape_by_model.md", aggregate_rows, AGG_COLUMNS),
    ]
    for path, rows, columns in outputs:
        if path.suffix == ".csv":
            write_csv(path, rows, columns)
        else:
            write_markdown(path, rows, columns)
        print(f"Saved {path.relative_to(ROOT)}")

    print(f"Rows: detail={len(detail_rows)}, aggregate={len(aggregate_rows)}")


if __name__ == "__main__":
    main()
