"""Benchmark pure model forward-pass inference latency.

This script complements the existing experiment summaries, which record
``train_time_seconds`` but do not time inference.  The benchmark deliberately
measures only ``model(x)`` under ``torch.inference_mode()``:

- no DataLoader iteration;
- no metric computation;
- no CPU/GPU transfer inside the timed region;
- no checkpoint loading by default, because learned weights do not change the
  number of operations in a forward pass.

The resulting CSV is intended for the report's "model complexity / inference
time" section.  It uses the tuned v2 architecture parameters documented in
``docs/best_model_params.md`` so that latency is measured on the same model
structures that produced the validation-best full-matrix results.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import AutoformerModel, InformerModel, LSTMModel, PatchTSTModel, TransformerModel  # noqa: E402
from models.trainer import resolve_device  # noqa: E402


# The official experiment design uses 96 historical steps and five forecast
# horizons.  Keeping these defaults aligned with AGENTS.md avoids accidentally
# benchmarking a shape that is unrelated to the report.
DEFAULT_DATASETS = "ETTh1,ETTm1"
DEFAULT_HORIZONS = "24,48,96,168,336"
DEFAULT_MODELS = "lstm,transformer,informer,autoformer,patchtst"
DEFAULT_BATCH_SIZES = "1,128"
DEFAULT_LOOKBACK = 96

DEFAULT_OUTPUT_CSV = (
    ROOT
    / "archive"
    / "v2_results"
    / "experiments"
    / "validation_best_full_matrix"
    / "summaries"
    / "csv"
    / "pure_forward_inference_benchmark.csv"
)


# Tuned structures selected from ETTh1 h96 validation experiments.  These values
# intentionally differ from the historical default MODEL_BUILDERS in
# scripts/run_experiments.py for several models, so the benchmark keeps a local,
# explicit config table instead of importing the older defaults.
TUNED_MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "lstm": {
        "hidden_size": 256,
        "num_layers": 1,
        "dropout": 0.1,
    },
    "transformer": {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
    "informer": {
        "d_model": 64,
        "n_heads": 4,
        "n_encoder_layers": 2,
        "n_decoder_layers": 2,
        "d_ff": 256,
        "factor": 3,
        "dropout": 0.1,
    },
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


def parse_csv_list(value: str) -> list[str]:
    """Parse a comma-separated CLI value while ignoring extra whitespace."""

    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated integers such as ``24,48,96``."""

    return [int(item) for item in parse_csv_list(value)]


def set_seed(seed: int) -> None:
    """Make model initialization and synthetic inputs reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters, matching the existing experiment summaries."""

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_tuned_model(model_name: str, input_size: int, horizon: int) -> torch.nn.Module:
    """Instantiate one of the five tuned v2 model structures."""

    cfg = TUNED_MODEL_CONFIGS[model_name]
    if model_name == "lstm":
        return LSTMModel(input_size=input_size, horizon=horizon, **cfg)
    if model_name == "transformer":
        return TransformerModel(input_size=input_size, horizon=horizon, **cfg)
    if model_name == "informer":
        return InformerModel(input_size=input_size, horizon=horizon, lookback=DEFAULT_LOOKBACK, **cfg)
    if model_name == "autoformer":
        return AutoformerModel(input_size=input_size, horizon=horizon, lookback=DEFAULT_LOOKBACK, **cfg)
    if model_name == "patchtst":
        return PatchTSTModel(input_size=input_size, horizon=horizon, **cfg)
    raise KeyError(f"Unknown model: {model_name}")


def infer_input_size(data_dir: Path, dataset: str, horizon: int, lookback: int) -> int:
    """Read processed data only to recover the feature dimension.

    The timed benchmark uses synthetic tensors, but taking ``input_size`` from
    the processed test split keeps the benchmark tied to the actual experiment
    data layout.  For ETTh1/ETTm1 this should be 7.
    """

    split_path = data_dir / dataset / f"h{horizon}" / "test.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Cannot infer input_size; missing {split_path}")
    with np.load(split_path) as data:
        x = data["X"]
    if x.ndim != 3:
        raise ValueError(f"Expected X to have shape (N, lookback, features), got {x.shape}")
    if x.shape[1] != lookback:
        raise ValueError(f"Expected lookback={lookback}, got {x.shape[1]} in {split_path}")
    return int(x.shape[2])


def synchronize(device: torch.device) -> None:
    """Wait for queued accelerator work before reading the wall clock."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        # MPS, like CUDA, executes many operations asynchronously.  Synchronizing
        # around the measured region prevents under-counting forward latency.
        try:
            torch.mps.synchronize()
        except RuntimeError:
            # Older PyTorch/MPS builds may expose the method but fail at runtime;
            # in that case the benchmark still runs, and the device field makes
            # the timing context explicit in the CSV.
            pass


def percentile(values: np.ndarray, q: float) -> float:
    """Return a percentile as a plain float for JSON/CSV serialization."""

    return float(np.percentile(values, q))


@torch.inference_mode()
def benchmark_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    horizon: int,
    input_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    """Measure repeated pure forward passes for one model/input shape.

    ``warmup`` absorbs lazy kernel setup and one-time graph/cache effects.  Each
    timed repeat synchronizes before and after ``model(x)`` so the recorded time
    corresponds to completed model work, not only operation enqueue time.
    """

    model.eval()

    # Shape validation is outside the timed loop.  It catches accidental config
    # mismatches early, which is more useful than silently writing invalid rows.
    output = model(x)
    expected_shape = (x.shape[0], horizon, input_size)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"Expected output shape {expected_shape}, got {tuple(output.shape)}")

    for _ in range(warmup):
        model(x)
    synchronize(x.device)

    times_ms: list[float] = []
    for _ in range(repeats):
        synchronize(x.device)
        start = time.perf_counter()
        model(x)
        synchronize(x.device)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(times_ms, dtype=np.float64)
    batch_size = int(x.shape[0])
    median_ms = float(np.median(arr))
    return {
        "latency_mean_ms": float(np.mean(arr)),
        "latency_median_ms": median_ms,
        "latency_std_ms": float(np.std(arr, ddof=0)),
        "latency_min_ms": float(np.min(arr)),
        "latency_p95_ms": percentile(arr, 95),
        "latency_per_sample_ms": median_ms / batch_size,
        "throughput_samples_per_sec": batch_size / (median_ms / 1000.0) if median_ms > 0 else float("inf"),
    }


def device_label(device: torch.device) -> str:
    """Return a compact human-readable device name for metadata."""

    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return platform.processor() or platform.machine() or "CPU"


def make_row(
    *,
    dataset: str,
    horizon: int,
    model_name: str,
    batch_size: int,
    lookback: int,
    input_size: int,
    device: torch.device,
    warmup: int,
    repeats: int,
    model_params: int,
    stats: dict[str, float],
) -> dict[str, object]:
    """Combine identifying metadata and measured latency statistics."""

    row: dict[str, object] = {
        "dataset": dataset,
        "horizon": horizon,
        "model": model_name,
        "batch_size": batch_size,
        "lookback": lookback,
        "input_size": input_size,
        "device": device.type,
        "device_name": device_label(device),
        "dtype": "float32",
        "weights_source": "random_initialized",
        "checkpoint_loaded": False,
        "warmup": warmup,
        "repeats": repeats,
        "model_params": model_params,
    }
    row.update(stats)
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows with a stable column order."""

    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: Iterable[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    """Average task-level medians into compact report-friendly summaries."""

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)

    summary_rows: list[dict[str, object]] = []
    for key_values, group in sorted(grouped.items()):
        medians = [float(row["latency_median_ms"]) for row in group]
        per_sample = [float(row["latency_per_sample_ms"]) for row in group]
        throughput = [float(row["throughput_samples_per_sec"]) for row in group]
        item = {key: value for key, value in zip(keys, key_values)}
        item.update(
            {
                "n_tasks": len(group),
                "mean_latency_median_ms": statistics.fmean(medians),
                "median_latency_median_ms": statistics.median(medians),
                "mean_latency_per_sample_ms": statistics.fmean(per_sample),
                "mean_throughput_samples_per_sec": statistics.fmean(throughput),
                "model_params": int(group[0]["model_params"]),
                "device": group[0]["device"],
                "device_name": group[0]["device_name"],
            }
        )
        summary_rows.append(item)
    return summary_rows


def markdown_table(rows: list[dict[str, object]], columns: list[str], max_rows: int | None = None) -> str:
    """Render a tiny Markdown table without adding a pandas dependency."""

    selected = rows[:max_rows] if max_rows else rows
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in selected:
        rendered = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def display_path(path: Path) -> str:
    """Prefer project-relative paths, but allow ad-hoc outputs outside ROOT."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def default_markdown_path(output_csv: Path) -> Path:
    """Place Markdown next to report summaries when using the project layout."""

    if output_csv.parent.name == "csv" and output_csv.parent.parent.name == "summaries":
        return output_csv.parent.parent / "md" / output_csv.with_suffix(".md").name
    return output_csv.with_suffix(".md")


def write_markdown_report(
    path: Path,
    rows: list[dict[str, object]],
    by_model: list[dict[str, object]],
    output_csv: Path,
    by_model_csv: Path,
    by_model_horizon_csv: Path,
    metadata_json: Path,
) -> None:
    """Create a short method note that can be copied into the final report."""

    batch1 = [row for row in by_model if int(row["batch_size"]) == 1]
    batch128 = [row for row in by_model if int(row["batch_size"]) == 128]
    lines = [
        "# Pure forward inference benchmark",
        "",
        "## 计时口径",
        "",
        "- 只测 `torch.inference_mode()` 下的 `model(x)` 前向传播。",
        "- 不包含 DataLoader、数据搬运、反归一化、指标计算或结果保存时间。",
        "- 使用 `docs/best_model_params.md` 记录的 v2 调优后结构参数。",
        "- 权重使用随机初始化；这不影响模型结构的前向计算量，但不代表某个具体 checkpoint 的预测精度。",
        "- 每个配置先 warmup，再重复计时，并在 CUDA/MPS 上做同步。",
        "",
        "## 输出文件",
        "",
        f"- 明细 CSV：`{display_path(output_csv)}`",
        f"- 按模型汇总 CSV：`{display_path(by_model_csv)}`",
        f"- 按模型/步长汇总 CSV：`{display_path(by_model_horizon_csv)}`",
        f"- 元数据 JSON：`{display_path(metadata_json)}`",
        "",
        "## Batch size = 1：五模型平均单次 forward 延迟",
        "",
        markdown_table(
            batch1,
            ["model", "n_tasks", "mean_latency_median_ms", "mean_latency_per_sample_ms", "model_params", "device"],
        ),
        "",
        "## Batch size = 128：五模型平均吞吐",
        "",
        markdown_table(
            batch128,
            [
                "model",
                "n_tasks",
                "mean_latency_median_ms",
                "mean_latency_per_sample_ms",
                "mean_throughput_samples_per_sec",
                "model_params",
                "device",
            ],
        ),
        "",
        "## 明细预览",
        "",
        markdown_table(
            rows,
            [
                "dataset",
                "horizon",
                "model",
                "batch_size",
                "latency_median_ms",
                "latency_p95_ms",
                "throughput_samples_per_sec",
            ],
            max_rows=12,
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark pure model forward inference latency")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS, help="Comma-separated datasets")
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS, help="Comma-separated forecast horizons")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated model names")
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES, help="Comma-separated batch sizes")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--warmup", type=int, default=30, help="Untimed forward passes before measurement")
    parser.add_argument("--repeats", type=int, default=200, help="Timed forward repeats")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    datasets = parse_csv_list(args.datasets)
    horizons = parse_int_list(args.horizons)
    models = parse_csv_list(args.models)
    batch_sizes = parse_int_list(args.batch_sizes)

    unknown = sorted(set(models) - set(TUNED_MODEL_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {sorted(TUNED_MODEL_CONFIGS)}")

    set_seed(args.seed)
    device = torch.device(resolve_device(args.device))
    rows: list[dict[str, object]] = []

    for dataset in datasets:
        for horizon in horizons:
            input_size = infer_input_size(args.data_dir, dataset, horizon, args.lookback)
            for model_name in models:
                # Recreate the model per horizon because the prediction head
                # changes shape with horizon.  It is reused across batch sizes.
                set_seed(args.seed)
                model = build_tuned_model(model_name, input_size, horizon).to(device)
                model_params = count_parameters(model)
                for batch_size in batch_sizes:
                    # Synthetic input isolates model computation.  Loading a
                    # real batch here would add disk/DataLoader variation without
                    # changing the tensor shape being benchmarked.
                    set_seed(args.seed)
                    x = torch.randn(
                        batch_size,
                        args.lookback,
                        input_size,
                        dtype=torch.float32,
                        device=device,
                    )
                    stats = benchmark_forward(
                        model=model,
                        x=x,
                        horizon=horizon,
                        input_size=input_size,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    row = make_row(
                        dataset=dataset,
                        horizon=horizon,
                        model_name=model_name,
                        batch_size=batch_size,
                        lookback=args.lookback,
                        input_size=input_size,
                        device=device,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        model_params=model_params,
                        stats=stats,
                    )
                    rows.append(row)
                    print(
                        f"{dataset} h{horizon:>3} {model_name:<11} "
                        f"bs={batch_size:<3} median={stats['latency_median_ms']:.3f} ms "
                        f"throughput={stats['throughput_samples_per_sec']:.1f}/s"
                    )

    output_csv = args.output_csv
    by_model_csv = output_csv.with_name(output_csv.stem + "_by_model.csv")
    by_model_horizon_csv = output_csv.with_name(output_csv.stem + "_by_model_horizon.csv")
    metadata_json = output_csv.with_name(output_csv.stem + "_metadata.json")
    markdown_path = default_markdown_path(output_csv)

    by_model = aggregate_rows(rows, ("batch_size", "model"))
    by_model_horizon = aggregate_rows(rows, ("batch_size", "model", "horizon"))

    write_csv(output_csv, rows)
    write_csv(by_model_csv, by_model)
    write_csv(by_model_horizon_csv, by_model_horizon)
    write_markdown_report(
        markdown_path,
        rows,
        by_model,
        output_csv,
        by_model_csv,
        by_model_horizon_csv,
        metadata_json,
    )

    metadata = {
        "benchmark_type": "pure_model_forward",
        "description": "Only model(x) is timed; DataLoader, transfers, metrics, and checkpoint loading are excluded.",
        "datasets": datasets,
        "horizons": horizons,
        "models": models,
        "batch_sizes": batch_sizes,
        "lookback": args.lookback,
        "device": device.type,
        "device_name": device_label(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "weights_source": "random_initialized",
        "config_source": "docs/best_model_params.md",
        "output_csv": display_path(output_csv),
        "by_model_csv": display_path(by_model_csv),
        "by_model_horizon_csv": display_path(by_model_horizon_csv),
        "markdown_report": display_path(markdown_path),
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved detail CSV: {output_csv}")
    print(f"Saved by-model CSV: {by_model_csv}")
    print(f"Saved by-model/horizon CSV: {by_model_horizon_csv}")
    print(f"Saved markdown report: {markdown_path}")
    print(f"Saved metadata: {metadata_json}")


if __name__ == "__main__":
    main()
