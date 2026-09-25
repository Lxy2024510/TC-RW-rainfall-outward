#!/usr/bin/env python3
"""RUN3: Keep population-expanded rainfall rows with RAIN_DIST from 200 to 500 km."""

from __future__ import annotations

import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
DATA_ROOT = PROJECT_ROOT / "Data/Intermediate/Exposure/MSWEP_DIST30"
QC_ROOT = PROJECT_ROOT / "Results/Quality_control/Exposure/RUN3"
GROUPS = ("RW", "RI")
DEFAULT_WORKERS = 16
MIN_RAIN_DISTANCE_KM = 200.0
MAX_RAIN_DISTANCE_KM = 500.0
EXPECTED_INPUT_COUNTS = {"RW": 7335, "RI": 4800}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter RUN2 population-point CSV files and retain rows whose rainfall "
            "grid-center distance is within the inclusive 200-500 km interval."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=GROUPS,
        default=list(GROUPS),
        help="Event groups to process (default: RW RI).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of worker processes (default: 16).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute files that already have a successful RUN3 marker.",
    )
    return parser.parse_args()


def write_csv_atomic(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a CSV through a temporary file to prevent partial final outputs."""
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    dataframe.to_csv(temporary_path, index=False, encoding="utf-8")
    os.replace(temporary_path, path)


def write_text_atomic(text: str, path: Path) -> None:
    """Write a completion marker atomically."""
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(text, encoding="utf-8")
    os.replace(temporary_path, path)


def process_one_csv(task: dict[str, Any]) -> dict[str, Any]:
    """Filter one RUN2 CSV using the inclusive RAIN_DIST interval."""
    input_path = Path(task["input_path"])
    output_path = Path(task["output_path"])
    done_marker = Path(task["done_marker"])
    group = task["group"]

    record: dict[str, Any] = {
        "GROUP": group,
        "INPUT_FILE": input_path.name,
        "STATUS": "failed",
        "INPUT_ROW_COUNT": 0,
        "OUTPUT_ROW_COUNT": 0,
        "REMOVED_ROW_COUNT": 0,
        "MESSAGE": "",
    }

    try:
        data = pd.read_csv(input_path)
        record["INPUT_ROW_COUNT"] = len(data)

        if "RAIN_DIST" not in data.columns:
            raise KeyError("Missing required column: RAIN_DIST")

        # Invalid distance values do not satisfy the confirmed interval.
        rain_distance = pd.to_numeric(data["RAIN_DIST"], errors="coerce")
        keep_mask = rain_distance.between(
            MIN_RAIN_DISTANCE_KM,
            MAX_RAIN_DISTANCE_KM,
            inclusive="both",
        )
        filtered = data.loc[keep_mask].copy()

        # An empty result is valid and is saved with the original column headers.
        write_csv_atomic(filtered, output_path)
        write_text_atomic(
            f"Completed by RUN3_FILTER_POPULATION_POINTS_200_500KM.py at "
            f"{datetime.now().isoformat()}\n",
            done_marker,
        )

        record.update(
            {
                "STATUS": "completed_empty" if filtered.empty else "completed",
                "OUTPUT_ROW_COUNT": len(filtered),
                "REMOVED_ROW_COUNT": len(data) - len(filtered),
            }
        )
        return record

    except Exception as exc:
        record["MESSAGE"] = (
            f"{type(exc).__name__}: {exc} | {traceback.format_exc(limit=3)}"
        )
        return record


def discover_input_files(input_dir: Path) -> list[Path]:
    """Return only the three RUN2 CSV product types."""
    files: list[Path] = []
    for path in input_dir.glob("row_*.csv"):
        if path.name.endswith(("_S.csv", "_E.csv", "_S_ON_E.csv")):
            files.append(path)
    return sorted(files, key=lambda path: path.name)


def build_tasks(
    group: str,
    input_files: list[Path],
    output_dir: Path,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Create pending tasks and skip files with RUN3 completion markers."""
    tasks: list[dict[str, Any]] = []
    skipped = 0

    for input_path in input_files:
        output_path = output_dir / input_path.name
        done_marker = output_dir / f"{input_path.name}.done"
        if done_marker.is_file() and not overwrite:
            skipped += 1
            continue

        tasks.append(
            {
                "group": group,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "done_marker": str(done_marker),
            }
        )
    return tasks, skipped


def run_group(
    group: str,
    data_root: Path,
    workers: int,
    overwrite: bool,
) -> None:
    """Process every RUN2 CSV file for one event group."""
    group_dir = data_root / group
    input_dir = group_dir / "POP_POINT_MATCH"
    output_dir = group_dir / "POP_POINT_200_500"
    log_dir = QC_ROOT / group

    if not input_dir.is_dir():
        raise FileNotFoundError(f"RUN2 input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    input_files = discover_input_files(input_dir)
    actual_count = len(input_files)
    expected_count = EXPECTED_INPUT_COUNTS[group]
    print(f"[{group}] RUN2 input CSV files: {actual_count} (expected: {expected_count})")
    if actual_count != expected_count:
        print(f"[{group}] WARNING: Input file count differs from the confirmed count.")

    tasks, skipped = build_tasks(group, input_files, output_dir, overwrite)
    print(f"[{group}] Completed files skipped: {skipped}")
    print(f"[{group}] Files submitted: {len(tasks)}")

    if not tasks:
        print(f"[{group}] No pending work.")
        return

    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_file = {
            executor.submit(process_one_csv, task): Path(task["input_path"]).name
            for task in tasks
        }
        progress = tqdm(
            as_completed(future_to_file),
            total=len(future_to_file),
            desc=f"{group} 200-500 km filtering",
        )
        for future in progress:
            input_name = future_to_file[future]
            try:
                records.append(future.result())
            except Exception as exc:
                records.append(
                    {
                        "GROUP": group,
                        "INPUT_FILE": input_name,
                        "STATUS": "failed",
                        "INPUT_ROW_COUNT": 0,
                        "OUTPUT_ROW_COUNT": 0,
                        "REMOVED_ROW_COUNT": 0,
                        "MESSAGE": f"{type(exc).__name__}: {exc}",
                    }
                )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"RUN3_{group}_task_log_{run_stamp}.csv"
    log_data = pd.DataFrame(records).sort_values("INPUT_FILE")
    write_csv_atomic(log_data, log_path)

    completed_mask = log_data["STATUS"].isin(["completed", "completed_empty"])
    completed = int(completed_mask.sum())
    empty = int((log_data["STATUS"] == "completed_empty").sum())
    failed = int((log_data["STATUS"] == "failed").sum())
    input_rows = int(log_data.loc[completed_mask, "INPUT_ROW_COUNT"].sum())
    output_rows = int(log_data.loc[completed_mask, "OUTPUT_ROW_COUNT"].sum())

    print(f"[{group}] Completed in this run: {completed}")
    print(f"[{group}] Empty filtered files: {empty}")
    print(f"[{group}] Failed in this run: {failed}")
    print(f"[{group}] Input rows read: {input_rows}")
    print(f"[{group}] Rows retained: {output_rows}")
    print(f"[{group}] Rows removed: {input_rows - output_rows}")
    print(f"[{group}] Log: {log_path}")


def main() -> None:
    args = parse_arguments()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    print(f"Data root: {args.data_root}")
    print(f"Workers: {args.workers}")
    print(
        f"RAIN_DIST interval: [{MIN_RAIN_DISTANCE_KM}, "
        f"{MAX_RAIN_DISTANCE_KM}] km"
    )

    for group in args.groups:
        run_group(
            group=group,
            data_root=args.data_root,
            workers=args.workers,
            overwrite=args.overwrite,
        )

    print("RUN3 finished.")


if __name__ == "__main__":
    main()
