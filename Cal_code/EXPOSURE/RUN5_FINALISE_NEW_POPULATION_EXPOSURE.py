#!/usr/bin/env python3
"""RUN5: Deduplicate and validate final new-exposure cells."""
from __future__ import annotations
import argparse, os, re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(
    os.environ.get("TC_RW_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).resolve()
DATA_ROOT = PROJECT_ROOT / "Data/Intermediate/Exposure/MSWEP_DIST30"
QC_ROOT = PROJECT_ROOT / "Results/Quality_control/Exposure/RUN5"
GROUPS = ("RW", "RI")
DEFAULT_WORKERS = 16
COORDINATE_DECIMALS = 4
RAW_PATTERN = re.compile(r"^row_(\d+)_E_NEW_RAW\.csv$")

def parse_arguments():
    p = argparse.ArgumentParser(description="Finalize new-exposure files.")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--coordinate-decimals", type=int, default=COORDINATE_DECIMALS)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def write_csv_atomic(data, path):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)

def keyed(data, decimals):
    missing = sorted({"POP_LAT", "POP_LON"}.difference(data.columns))
    if missing: raise KeyError(f"Missing columns: {', '.join(missing)}")
    out = data.copy()
    out["__LAT_KEY"] = pd.to_numeric(out.POP_LAT, errors="raise").round(decimals)
    out["__LON_KEY"] = pd.to_numeric(out.POP_LON, errors="raise").round(decimals)
    return out

def idx(data):
    return pd.MultiIndex.from_frame(data[["__LAT_KEY", "__LON_KEY"]])

def process_one(task):
    row_id, directory, decimals = int(task["row_id"]), Path(task["directory"]), int(task["decimals"])
    record = {"SOURCE_INDEX": row_id, "STATUS": "failed", "MESSAGE": ""}
    try:
        raw = keyed(pd.read_csv(directory / f"row_{row_id}_E_NEW_RAW.csv"), decimals)
        overlap = keyed(pd.read_csv(directory / f"row_{row_id}_S_INTERSECT_S_ON_E.csv"), decimals)
        residual = int(idx(raw).isin(idx(overlap.drop_duplicates(["__LAT_KEY", "__LON_KEY"]))).sum())
        if residual: raise RuntimeError(f"Final candidate still contains {residual} overlap rows")
        if "POP_VALUE" not in raw.columns: raise KeyError("POP_VALUE is missing")
        raw["POP_VALUE"] = pd.to_numeric(raw.POP_VALUE, errors="raise")
        key_cols = ["__LAT_KEY", "__LON_KEY"]
        conflicts = int((raw.groupby(key_cols).POP_VALUE.nunique() > 1).sum())
        final = raw.drop_duplicates(key_cols, keep="first").copy()
        population = float(final.POP_VALUE.sum())
        final = final.drop(columns=key_cols)
        write_csv_atomic(final, directory / f"row_{row_id}_E_NEW_FINAL.csv")
        marker = directory / f"row_{row_id}.run5.done"
        marker.write_text(f"Completed at {datetime.now().isoformat()}\n", encoding="utf-8")
        record.update(STATUS="completed", RAW_ROWS=len(raw), FINAL_UNIQUE_COORDS=len(final), DUPLICATE_ROWS_REMOVED=len(raw)-len(final), CONFLICTING_POP_VALUES=conflicts, FINAL_POPULATION=population)
    except Exception as exc:
        record["MESSAGE"] = f"{type(exc).__name__}: {exc}"
    return record

def run_group(group, root, workers, decimals, overwrite):
    group_dir = root / group
    directory = group_dir / "NEW_EXPOSURE_POINTS"
    log_dir = QC_ROOT / group
    log_dir.mkdir(parents=True, exist_ok=True)
    row_ids = sorted(int(m.group(1)) for p in directory.glob("row_*_E_NEW_RAW.csv") if (m := RAW_PATTERN.fullmatch(p.name)))
    tasks, skipped = [], 0
    for row_id in row_ids:
        marker = directory / f"row_{row_id}.run5.done"
        if marker.is_file() and not overwrite: skipped += 1
        else: tasks.append({"row_id": row_id, "directory": str(directory), "decimals": decimals})
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{group} RUN5"):
            records.append(future.result())
    if records:
        log = pd.DataFrame(records).sort_values("SOURCE_INDEX")
        write_csv_atomic(log, log_dir / f"RUN5_{group}_{datetime.now():%Y%m%d_%H%M%S}.csv")
        print(f"[{group}] completed={(log.STATUS == 'completed').sum()}, failed={(log.STATUS == 'failed').sum()}")
    print(f"[{group}] discovered={len(row_ids)}, submitted={len(tasks)}, skipped={skipped}")

def main():
    args = parse_arguments()
    if args.workers < 1: raise ValueError("--workers must be at least 1")
    for group in args.groups: run_group(group, args.data_root, args.workers, args.coordinate_decimals, args.overwrite)

if __name__ == "__main__":
    main()
