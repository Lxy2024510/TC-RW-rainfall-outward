#!/usr/bin/env python3
"""RUN6: Create final event-level new-population-exposure tables."""
from __future__ import annotations
import argparse, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
DATA_ROOT = PROJECT_ROOT / "Data/Intermediate/Exposure/MSWEP_DIST30"
OUTPUT_ROOT = PROJECT_ROOT / "Data/Processed/Exposure/MSWEP_DIST30"
GROUPS = ("RW", "RI")
DEFAULT_WORKERS = 16

def parse_arguments():
    p = argparse.ArgumentParser(
        description=(
            "Create the final event-level population exposure table using "
            "E minus the intersection of S and S_ON_E."
        )
    )
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return p.parse_args()

def write_csv_atomic(data, path):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)

def summarize(task):
    row_id, directory = int(task["row_id"]), Path(task["directory"])
    path = directory / f"row_{row_id}_E_NEW_FINAL.csv"
    record = {"SOURCE_INDEX": row_id, "STATUS": "error", "MESSAGE": ""}
    try:
        if not path.is_file(): raise FileNotFoundError(path)
        data = pd.read_csv(path)
        required = {"RAIN_LAT", "RAIN_LON", "POP_LAT", "POP_LON", "POP_VALUE"}
        missing = sorted(required.difference(data.columns))
        if missing: raise KeyError(f"Missing columns: {', '.join(missing)}")
        values = pd.to_numeric(data.POP_VALUE, errors="raise")
        if data[["POP_LAT", "POP_LON"]].duplicated().any():
            raise ValueError("Final RUN5 file contains duplicate population coordinates")
        record.update(
            STATUS="empty_valid" if data.empty else "valid",
            NEW_EXPOSURE_RAIN_POINT=len(data[["RAIN_LAT", "RAIN_LON"]].drop_duplicates()),
            NEW_EXPOSURE_POP_POINT=len(data),
            NEW_EXPOSURE_POP_VALUE=float(values.sum()),
            NEW_EXPOSURE_FILE=str(path),
        )
    except pd.errors.EmptyDataError:
        record["MESSAGE"] = "Empty CSV without headers"
    except Exception as exc:
        record["MESSAGE"] = f"{type(exc).__name__}: {exc}"
    return record

def run_group(group, root, output_root, workers):
    group_dir = root / group
    main_path = group_dir / f"FILTERED_{group}.csv"
    directory = group_dir / "NEW_EXPOSURE_POINTS"
    if not main_path.is_file(): raise FileNotFoundError(main_path)
    main = pd.read_csv(main_path)
    if "SOURCE_INDEX" not in main.columns: raise KeyError("SOURCE_INDEX is missing")
    ids = pd.to_numeric(main.SOURCE_INDEX, errors="raise").astype(int)
    if ids.duplicated().any(): raise ValueError("Duplicate SOURCE_INDEX values")
    tasks = [{"row_id": int(row_id), "directory": str(directory)} for row_id in ids]
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(summarize, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{group} RUN6"):
            records.append(future.result())
    summary = pd.DataFrame(records)
    result = main.merge(summary, on="SOURCE_INDEX", how="left", validate="one_to_one")
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"{group}_FINAL_NEW_POPULATION_EXPOSURE.csv"
    write_csv_atomic(result, output)
    valid = summary.STATUS.isin(["valid", "empty_valid"])
    print(f"[{group}] valid={valid.sum()}, errors={(~valid).sum()}")
    print(f"[{group}] final population total={summary.loc[valid, 'NEW_EXPOSURE_POP_VALUE'].sum():.6f}")
    print(f"[{group}] output={output}")

def main():
    args = parse_arguments()
    if args.workers < 1: raise ValueError("--workers must be at least 1")
    for group in args.groups:
        run_group(group, args.data_root, args.output_root, args.workers)

if __name__ == "__main__":
    main()
