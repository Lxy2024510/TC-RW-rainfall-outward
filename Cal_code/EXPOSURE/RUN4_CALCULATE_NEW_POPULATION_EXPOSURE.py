#!/usr/bin/env python3
"""RUN4: Calculate E_new = E - (S intersection S_ON_E)."""
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
QC_ROOT = PROJECT_ROOT / "Results/Quality_control/Exposure/RUN4"
GROUPS = ("RW", "RI")
DEFAULT_WORKERS = 16
COORDINATE_DECIMALS = 4
S_PATTERN = re.compile(r"^row_(\d+)_S\.csv$")

def parse_arguments():
    p = argparse.ArgumentParser(description="Calculate raw new exposure: E minus (S intersect S_ON_E).")
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

def write_text_atomic(value, path):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)

def keyed(data, decimals):
    missing = sorted({"POP_LAT", "POP_LON"}.difference(data.columns))
    if missing:
        raise KeyError(f"Missing coordinate columns: {', '.join(missing)}")
    out = data.copy()
    out["__LAT_KEY"] = pd.to_numeric(out.POP_LAT, errors="raise").round(decimals)
    out["__LON_KEY"] = pd.to_numeric(out.POP_LON, errors="raise").round(decimals)
    return out

def index_of(data):
    return pd.MultiIndex.from_frame(data[["__LAT_KEY", "__LON_KEY"]])

def process_one(task):
    row_id, input_dir, output_dir = int(task["row_id"]), Path(task["input_dir"]), Path(task["output_dir"])
    decimals = int(task["decimals"])
    record = {"SOURCE_INDEX": row_id, "STATUS": "failed", "MESSAGE": ""}
    try:
        frames = {}
        for name in ("S", "S_ON_E", "E"):
            path = input_dir / f"row_{row_id}_{name}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing input: {path}")
            frames[name] = keyed(pd.read_csv(path), decimals)
        keys = ["__LAT_KEY", "__LON_KEY"]
        s = frames["S"].drop_duplicates(keys)
        soe = frames["S_ON_E"].drop_duplicates(keys)
        e = frames["E"]
        overlap = s.loc[index_of(s).isin(index_of(soe))].copy()
        remove = index_of(e).isin(index_of(overlap))
        e_new = e.loc[~remove].copy()
        write_csv_atomic(e_new.drop(columns=keys), output_dir / f"row_{row_id}_E_NEW_RAW.csv")
        write_csv_atomic(overlap.drop(columns=keys), output_dir / f"row_{row_id}_S_INTERSECT_S_ON_E.csv")
        write_text_atomic(f"Completed by RUN4 at {datetime.now().isoformat()}\n", output_dir / f"row_{row_id}.run4.done")
        record.update(STATUS="completed", S_UNIQUE_COORDS=len(s), S_ON_E_UNIQUE_COORDS=len(soe), E_ROWS=len(e), E_UNIQUE_COORDS=len(e.drop_duplicates(keys)), OVERLAP_COORDS=len(overlap), E_ROWS_REMOVED=int(remove.sum()), E_NEW_RAW_ROWS=len(e_new))
    except Exception as exc:
        record["MESSAGE"] = f"{type(exc).__name__}: {exc}"
    return record

def run_group(group, root, workers, decimals, overwrite):
    group_dir = root / group
    input_dir = group_dir / "POP_POINT_200_500"
    output_dir = group_dir / "NEW_EXPOSURE_POINTS"
    log_dir = QC_ROOT / group
    output_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    row_ids = sorted(int(m.group(1)) for p in input_dir.glob("row_*_S.csv") if (m := S_PATTERN.fullmatch(p.name)))
    tasks, skipped = [], 0
    for row_id in row_ids:
        marker = output_dir / f"row_{row_id}.run4.done"
        if marker.is_file() and not overwrite:
            skipped += 1
        else:
            tasks.append({"row_id": row_id, "input_dir": str(input_dir), "output_dir": str(output_dir), "decimals": decimals})
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{group} RUN4"):
            records.append(future.result())
    if records:
        log = pd.DataFrame(records).sort_values("SOURCE_INDEX")
        write_csv_atomic(log, log_dir / f"RUN4_{group}_{datetime.now():%Y%m%d_%H%M%S}.csv")
        print(f"[{group}] completed={(log.STATUS == 'completed').sum()}, failed={(log.STATUS == 'failed').sum()}")
    print(f"[{group}] discovered={len(row_ids)}, submitted={len(tasks)}, skipped={skipped}")

def main():
    args = parse_arguments()
    if args.workers < 1: raise ValueError("--workers must be at least 1")
    if args.coordinate_decimals < 0: raise ValueError("--coordinate-decimals must be non-negative")
    for group in args.groups:
        run_group(group, args.data_root, args.workers, args.coordinate_decimals, args.overwrite)

if __name__ == "__main__":
    main()
