#!/usr/bin/env python3
"""Audit the downloaded LaDe-D source without exporting package-level IDs."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "pyarrow>=15",
# ]
# ///

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "order_id",
    "region_id",
    "city",
    "courier_id",
    "lng",
    "lat",
    "aoi_id",
    "aoi_type",
    "accept_time",
    "accept_gps_time",
    "accept_gps_lng",
    "accept_gps_lat",
    "delivery_time",
    "delivery_gps_time",
    "delivery_gps_lng",
    "delivery_gps_lat",
    "ds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 LaDe-D 数据源和上海子集。")
    parser.add_argument("--root", type=Path, default=Path("external_data/LaDe-D"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/lade_audit")
    )
    return parser.parse_args()


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def audit_source(root: Path) -> dict[str, object]:
    data_dir = root / "data"
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"没有找到 LaDe-D Parquet 文件: {data_dir}")
    file_records = []
    for path in files:
        frame = pd.read_parquet(path)
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        file_records.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "rows": len(frame),
                "columns": [str(column) for column in frame.columns],
                "missing_required_columns": missing,
            }
        )
    return {
        "source": "https://huggingface.co/datasets/Cainiao-AI/LaDe-D",
        "repository_commit": git_value(root, "rev-parse", "HEAD"),
        "repository_commit_time": git_value(root, "show", "-s", "--format=%cI", "HEAD"),
        "license_in_readme": "Apache-2.0",
        "research_use_note": "README states that LaDe can be used for research purposes.",
        "time_range_in_readme": "2022-05-01 to 2022-10-31",
        "files": file_records,
        "time_window_fields_in_source": False,
        "time_window_note": "LaDe-D provides accept and delivery event times but no explicit start/end time-window fields.",
    }


def parse_event_time(values: pd.Series) -> pd.Series:
    return pd.to_datetime("2022-" + values.astype(str), errors="coerce")


def audit_shanghai(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path)
    frame["accept_dt"] = parse_event_time(frame["accept_time"])
    frame["delivery_dt"] = parse_event_time(frame["delivery_time"])
    frame["event_duration_minutes"] = (
        frame["delivery_dt"] - frame["accept_dt"]
    ).dt.total_seconds() / 60
    coordinate_columns = ["lng", "lat", "accept_gps_lng", "accept_gps_lat", "delivery_gps_lng", "delivery_gps_lat"]
    coordinate_missing = frame[coordinate_columns].isna().sum().to_dict()
    coordinate_invalid = (
        (frame["lng"].notna() & ~frame["lng"].between(120, 122))
        | (frame["lat"].notna() & ~frame["lat"].between(30, 32))
    )
    duration = frame["event_duration_minutes"]
    return {
        "file": path.name,
        "city_values": sorted(frame["city"].dropna().unique().tolist()),
        "rows": len(frame),
        "unique_orders": int(frame["order_id"].nunique()),
        "unique_couriers": int(frame["courier_id"].nunique()),
        "working_days": int(frame["ds"].nunique()),
        "date_code_min": int(frame["ds"].min()),
        "date_code_max": int(frame["ds"].max()),
        "missing_values": {str(key): int(value) for key, value in frame.isna().sum().items()},
        "coordinate_missing": {str(key): int(value) for key, value in coordinate_missing.items()},
        "invalid_delivery_coordinates": int(coordinate_invalid.sum()),
        "accept_time_parse_failures": int(frame["accept_dt"].isna().sum()),
        "delivery_time_parse_failures": int(frame["delivery_dt"].isna().sum()),
        "delivery_before_accept": int((duration < 0).sum()),
        "event_duration_minutes": {
            "p50": float(duration.quantile(0.5)),
            "p90": float(duration.quantile(0.9)),
            "p99": float(duration.quantile(0.99)),
            "max": float(duration.max()),
        },
        "aoi_count": int(frame["aoi_id"].nunique()),
        "aoi_type_count": int(frame["aoi_type"].nunique()),
        "explicit_time_window_available": False,
        "time_window_action": "Use observed delivery events for validation; define proxy service deadlines only as a separate sensitivity input.",
    }


def main() -> None:
    args = parse_args()
    root = args.root
    shanghai_path = next(root.glob("data/delivery_sh-*.parquet"), None)
    if shanghai_path is None:
        raise FileNotFoundError("没有找到上海 LaDe-D Parquet 文件")
    source = audit_source(root)
    shanghai = audit_shanghai(shanghai_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "shanghai_audit.json").write_text(
        json.dumps(shanghai, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"source": source, "shanghai": shanghai}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
