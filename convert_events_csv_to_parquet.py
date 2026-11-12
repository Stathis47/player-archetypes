"""One-off conversion of the raw Wyscout `events.csv` export to `events.parquet`.

Every pipeline in this repository reads `events.parquet` rather than the raw
CSV: the parquet file is ~10x smaller on disk and supports column-pruned
reads, which matters for a ~2.2M-row x 160-column export. Run this script once
after obtaining `events.csv`, before running any of the pipelines.

The conversion is deliberately lossless and unopinionated: all columns are
kept, dtypes are whatever pandas infers from the CSV, and no rows are
filtered or reordered. All cleaning/feature engineering lives downstream
(`enrich_events_pipeline.py`, `player_profile_pipeline.py`, ...).

Usage:
    python convert_events_csv_to_parquet.py
    python convert_events_csv_to_parquet.py --csv path/to/events.csv --output path/to/events.parquet
    python convert_events_csv_to_parquet.py --overwrite

Paths default to the same `.env` / environment configuration used by the
pipelines (`GOALUNIT_DATA_DIR`, `GOALUNIT_EVENTS_PARQUET`), plus an optional
`GOALUNIT_EVENTS_CSV` override for the input file.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Final

import pandas as pd


def _load_local_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file into os.environ."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
_load_local_env_file(PROJECT_ROOT / ".env")

DATA_DIR: Final[Path] = Path(os.getenv("GOALUNIT_DATA_DIR", str(PROJECT_ROOT / "data")))
EVENTS_CSV: Final[Path] = Path(
    os.getenv("GOALUNIT_EVENTS_CSV", str(DATA_DIR / "events.csv"))
)
EVENTS_PARQUET: Final[Path] = Path(
    os.getenv("GOALUNIT_EVENTS_PARQUET", str(DATA_DIR / "events.parquet"))
)

# Rows per parquet row group. Smaller than pyarrow's ~1M default so that
# column-pruned reads keep peak memory low.
ROW_GROUP_SIZE: Final[int] = 200_000


def convert_events_csv_to_parquet(
    csv_path: Path, parquet_path: Path, *, overwrite: bool = False
) -> None:
    """Read the raw events CSV and write it unchanged as a snappy parquet file."""
    if not csv_path.exists():
        raise FileNotFoundError(
            "events.csv was not found. Configure GOALUNIT_DATA_DIR or "
            "GOALUNIT_EVENTS_CSV in .env or your shell environment, or pass --csv. "
            f"Resolved path: {csv_path}"
        )
    if parquet_path.exists() and not overwrite:
        raise FileExistsError(
            f"{parquet_path} already exists. Pass --overwrite to replace it, "
            "or --output to write somewhere else."
        )

    print(f"Reading {csv_path} ...")
    # low_memory=False makes pandas infer each column's dtype from the whole
    # file instead of per internal chunk, so the result is deterministic.
    events_df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(events_df):,} rows x {events_df.shape[1]} columns.")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {parquet_path} ...")
    events_df.to_parquet(
        parquet_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
        row_group_size=ROW_GROUP_SIZE,
    )
    size_mb = parquet_path.stat().st_size / 1024**2
    print(f"Done. Wrote {size_mb:,.1f} MB.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--csv", type=Path, default=EVENTS_CSV,
        help=f"Input events CSV (default: {EVENTS_CSV})",
    )
    parser.add_argument(
        "--output", type=Path, default=EVENTS_PARQUET,
        help=f"Output parquet path (default: {EVENTS_PARQUET})",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace the output file if it already exists.",
    )
    args = parser.parse_args()
    convert_events_csv_to_parquet(args.csv, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
