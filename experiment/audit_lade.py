#!/usr/bin/env python3
"""Audit the downloaded LaDe-D source without exporting package-level IDs."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "pyarrow>=15",
# ]
# ///

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
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


def haversine_km(
    lng1: pd.Series,
    lat1: pd.Series,
    lng2: pd.Series,
    lat2: pd.Series,
) -> pd.Series:
    """Calculate great-circle distance for aligned coordinate series."""
    lng1_rad = np.radians(lng1.astype(float))
    lat1_rad = np.radians(lat1.astype(float))
    lng2_rad = np.radians(lng2.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    delta_lng = lng2_rad - lng1_rad
    delta_lat = lat2_rad - lat1_rad
    value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lng / 2) ** 2
    )
    return pd.Series(6371.0088 * 2 * np.arcsin(np.sqrt(value)), index=lng1.index)


def quantiles(values: pd.Series) -> dict[str, float | None]:
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return {"p50": None, "p90": None, "p99": None, "max": None}
    return {
        "p50": float(valid.quantile(0.50)),
        "p90": float(valid.quantile(0.90)),
        "p99": float(valid.quantile(0.99)),
        "max": float(valid.max()),
    }


def within_shanghai_bounds(lng: pd.Series, lat: pd.Series) -> pd.Series:
    return lng.between(120, 122) & lat.between(30, 32)


def audit_route_sequence(frame: pd.DataFrame) -> dict[str, object]:
    duplicate_mask = frame.duplicated(
        subset=["courier_id", "ds", "order_id"], keep=False
    )
    ordered = frame.sort_values(
        ["courier_id", "ds", "delivery_dt", "order_id"], kind="stable"
    ).copy()
    groups = ordered.groupby(["courier_id", "ds"], sort=False)
    ordered["previous_delivery_dt"] = groups["delivery_dt"].shift()
    ordered["previous_lng"] = groups["delivery_gps_lng"].shift()
    ordered["previous_lat"] = groups["delivery_gps_lat"].shift()
    ordered["delivery_coordinate_valid"] = within_shanghai_bounds(
        ordered["delivery_gps_lng"], ordered["delivery_gps_lat"]
    )
    ordered["previous_coordinate_valid"] = groups[
        "delivery_coordinate_valid"
    ].shift(fill_value=False)
    ordered["gap_minutes"] = (
        ordered["delivery_dt"] - ordered["previous_delivery_dt"]
    ).dt.total_seconds() / 60
    ordered["adjacent_distance_km"] = haversine_km(
        ordered["previous_lng"],
        ordered["previous_lat"],
        ordered["delivery_gps_lng"],
        ordered["delivery_gps_lat"],
    )
    valid_adjacent_coordinates = (
        ordered["delivery_coordinate_valid"] & ordered["previous_coordinate_valid"]
    )
    ordered.loc[
        ~valid_adjacent_coordinates, "adjacent_distance_km"
    ] = np.nan
    positive_gap = ordered["gap_minutes"] > 0
    ordered["implied_speed_kmh"] = np.where(
        positive_gap,
        ordered["adjacent_distance_km"] / (ordered["gap_minutes"] / 60),
        np.nan,
    )
    location_drift = haversine_km(
        frame["lng"],
        frame["lat"],
        frame["delivery_gps_lng"],
        frame["delivery_gps_lat"],
    )
    valid_drift_coordinates = within_shanghai_bounds(
        frame["lng"], frame["lat"]
    ) & within_shanghai_bounds(
        frame["delivery_gps_lng"], frame["delivery_gps_lat"]
    )
    location_drift = location_drift.where(valid_drift_coordinates)
    comparable_pairs = ordered["previous_delivery_dt"].notna()
    zero_gap_movement = (
        comparable_pairs
        & (ordered["gap_minutes"] == 0)
        & (ordered["adjacent_distance_km"] > 0.1)
    )
    return {
        "duplicate_orders_within_courier_day": int(duplicate_mask.sum()),
        "duplicate_order_groups_within_courier_day": int(
            frame.loc[duplicate_mask, ["courier_id", "ds", "order_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "recovered_sequence_negative_time_gaps": int(
            (ordered["gap_minutes"] < 0).sum()
        ),
        "equal_completion_time_pairs": int(
            (comparable_pairs & (ordered["gap_minutes"] == 0)).sum()
        ),
        "equal_time_pairs_over_100m": int(zero_gap_movement.sum()),
        "coordinate_drift_km": quantiles(location_drift),
        "coordinate_drift_over_1km": int((location_drift > 1).sum()),
        "coordinate_drift_pairs_excluded": int((~valid_drift_coordinates).sum()),
        "adjacent_pairs": int(comparable_pairs.sum()),
        "adjacent_pairs_with_valid_coordinates": int(
            (comparable_pairs & valid_adjacent_coordinates).sum()
        ),
        "adjacent_pairs_excluded_for_coordinates": int(
            (comparable_pairs & ~valid_adjacent_coordinates).sum()
        ),
        "adjacent_distance_km": quantiles(ordered["adjacent_distance_km"]),
        "adjacent_distance_over_10km": int(
            (ordered["adjacent_distance_km"] > 10).sum()
        ),
        "positive_time_gap_pairs": int(positive_gap.sum()),
        "implied_speed_kmh": quantiles(ordered["implied_speed_kmh"]),
        "implied_speed_over_80kmh": int(
            (ordered["implied_speed_kmh"] > 80).sum()
        ),
        "threshold_note": "Flags are audit diagnostics only: drift >1 km, adjacent jump >10 km, or implied speed >80 km/h; they are not automatically removed.",
        "ordering_note": "Parquet row order has no route meaning. Sequences are recovered by delivery time, with order_id used only as a deterministic tie-breaker.",
    }


def audit_shanghai(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path)
    frame["accept_dt"] = parse_event_time(frame["accept_time"])
    frame["delivery_dt"] = parse_event_time(frame["delivery_time"])
    frame["event_duration_minutes"] = (
        frame["delivery_dt"] - frame["accept_dt"]
    ).dt.total_seconds() / 60
    coordinate_columns = ["lng", "lat", "accept_gps_lng", "accept_gps_lat", "delivery_gps_lng", "delivery_gps_lat"]
    coordinate_missing = frame[coordinate_columns].isna().sum().to_dict()
    coordinate_outside_bounds = {
        "order": int((~within_shanghai_bounds(frame["lng"], frame["lat"])).sum()),
        "accept_gps": int(
            (
                ~within_shanghai_bounds(
                    frame["accept_gps_lng"], frame["accept_gps_lat"]
                )
            ).sum()
        ),
        "delivery_gps": int(
            (
                ~within_shanghai_bounds(
                    frame["delivery_gps_lng"], frame["delivery_gps_lat"]
                )
            ).sum()
        ),
    }
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
        "coordinate_outside_shanghai_bounds": coordinate_outside_bounds,
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
        "route_sequence_audit": audit_route_sequence(frame),
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
