"""Organize legacy top-level result files into horizon folders.

New experiment results are written directly to:

- results/h{horizon}/{dataset}/{model}/{run_tag}/

This script keeps older top-level result files discoverable by linking them into
the same horizon-first layout and linking aggregate tables under results/summaries/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def link_or_copy(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return
        destination.unlink()

    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            shutil.copy2(source, destination)
            return

    relative_source = os.path.relpath(source, start=destination.parent)
    try:
        destination.symlink_to(relative_source)
    except OSError:
        shutil.copy2(source, destination)


def classify_result_pair(summary_path: Path, mode: str, overwrite: bool) -> int:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    dataset = data["dataset"]
    horizon = int(data["horizon"])
    model = data["model"]
    run_tag = data.get("run_tag") or "default"

    result_path = summary_path.with_name(summary_path.name.replace("_summary.json", "_results.npy"))
    sources = [summary_path]
    if result_path.exists():
        sources.append(result_path)

    destinations = [
        RESULTS_DIR / f"h{horizon}" / dataset / model / run_tag,
    ]

    created = 0
    for destination_dir in destinations:
        for source in sources:
            link_or_copy(source, destination_dir / source.name, mode, overwrite)
            created += 1
    return created


def classify_summaries(mode: str, overwrite: bool) -> int:
    created = 0
    destination_dir = RESULTS_DIR / "summaries"
    for source in sorted(list(RESULTS_DIR.glob("*.csv")) + list(RESULTS_DIR.glob("*.md"))):
        link_or_copy(source, destination_dir / source.name, mode, overwrite)
        created += 1
    return created


def write_index(total_runs: int, total_summary_files: int) -> None:
    lines = [
        "# Results Index",
        "",
        "New result files are written directly into horizon-first folders.",
        "Legacy top-level result files are linked into the same layout.",
        "",
        "Browse classified views:",
        "",
        "- `h{horizon}/{dataset}/{model}/{run_tag}/`",
        "- `summaries/`",
        "",
        f"Indexed experiment summaries: {total_runs}",
        f"Linked aggregate summary files: {total_summary_files}",
        "",
    ]
    (RESULTS_DIR / "RESULTS_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize legacy result files by horizon")
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary_paths = sorted(RESULTS_DIR.glob("*_summary.json"))
    linked_results = 0
    for summary_path in summary_paths:
        linked_results += classify_result_pair(summary_path, args.mode, args.overwrite)

    linked_summaries = classify_summaries(args.mode, args.overwrite)
    write_index(len(summary_paths), linked_summaries)

    print(f"Indexed experiment summaries: {len(summary_paths)}")
    print(f"Created/updated result view entries: {linked_results}")
    print(f"Created/updated aggregate summary entries: {linked_summaries}")
    print(f"Index: {RESULTS_DIR / 'RESULTS_INDEX.md'}")


if __name__ == "__main__":
    main()
