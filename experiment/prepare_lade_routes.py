#!/usr/bin/env python3
"""Recover anonymized courier-day delivery waves from LaDe-D."""

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
from pathlib import Path

import numpy as np
import pandas as pd

from audit_lade import haversine_km, parse_event_time, quantiles, within_shanghai_bounds

LONG_GAP_MINUTES = 120.0
LONG_JUMP_KM = 10.0
MAX_PLAUSIBLE_SPEED_KMH = 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="恢复 LaDe-D 上海快递员日配送波次。")
    parser.add_argument("--root", type=Path, default=Path("external_data/LaDe-D"))
    parser.add_argument("--processed-dir", type=Path, default=Path("processed/lade"))
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/lade_audit")
    )
    return parser.parse_args()


def recover_routes(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = frame.copy()
    data["accept_dt"] = parse_event_time(data["accept_time"])
    data["delivery_dt"] = parse_event_time(data["delivery_time"])
    data = data.sort_values(
        ["courier_id", "ds", "delivery_dt", "order_id"], kind="stable"
    ).reset_index(drop=True)

    day_groups = data.groupby(["courier_id", "ds"], sort=False)
    data["previous_delivery_dt"] = day_groups["delivery_dt"].shift()
    data["previous_lng"] = day_groups["delivery_gps_lng"].shift()
    data["previous_lat"] = day_groups["delivery_gps_lat"].shift()
    data["previous_region_id"] = day_groups["region_id"].shift()
    data["gap_minutes"] = (
        data["delivery_dt"] - data["previous_delivery_dt"]
    ).dt.total_seconds() / 60
    data["adjacent_distance_km"] = haversine_km(
        data["previous_lng"],
        data["previous_lat"],
        data["delivery_gps_lng"],
        data["delivery_gps_lat"],
    )
    data["coordinate_valid"] = within_shanghai_bounds(
        data["delivery_gps_lng"], data["delivery_gps_lat"]
    )
    previous_valid = day_groups["coordinate_valid"].shift(fill_value=False)
    valid_pair = data["coordinate_valid"] & previous_valid
    data.loc[~valid_pair, "adjacent_distance_km"] = np.nan
    data["implied_speed_kmh"] = np.where(
        data["gap_minutes"] > 0,
        data["adjacent_distance_km"] / (data["gap_minutes"] / 60),
        np.nan,
    )

    day_start = data["previous_delivery_dt"].isna()
    data["break_long_gap"] = data["gap_minutes"] > LONG_GAP_MINUTES
    data["break_long_jump"] = data["adjacent_distance_km"] > LONG_JUMP_KM
    data["break_high_speed"] = data["implied_speed_kmh"] > MAX_PLAUSIBLE_SPEED_KMH
    data["break_region_change"] = (
        ~day_start & data["region_id"].ne(data["previous_region_id"])
    )
    data["break_invalid_coordinate"] = ~day_start & ~valid_pair
    data["wave_start"] = (
        day_start
        | data["break_long_gap"]
        | data["break_long_jump"]
        | data["break_high_speed"]
        | data["break_region_change"]
        | data["break_invalid_coordinate"]
    )
    data["wave_index"] = (
        data.groupby(["courier_id", "ds"], sort=False)["wave_start"].cumsum().astype(int)
    )
    route_keys = pd.MultiIndex.from_frame(
        data[["courier_id", "ds", "wave_index"]]
    )
    route_codes, _ = pd.factorize(route_keys, sort=False)
    data["route_id"] = pd.Series(route_codes + 1, index=data.index).map(
        lambda value: f"R{value:08d}"
    )
    route_groups = data.groupby("route_id", sort=False)
    data["stop_index"] = route_groups.cumcount() + 1

    route_start = data["wave_start"]
    data.loc[route_start, ["gap_minutes", "adjacent_distance_km", "implied_speed_kmh"]] = np.nan
    data["centroid_lng"] = route_groups["delivery_gps_lng"].transform("mean")
    data["centroid_lat"] = route_groups["delivery_gps_lat"].transform("mean")
    data["distance_to_centroid_km"] = haversine_km(
        data["centroid_lng"],
        data["centroid_lat"],
        data["delivery_gps_lng"],
        data["delivery_gps_lat"],
    ).where(data["coordinate_valid"])

    summary = route_groups.agg(
        service_date=("delivery_dt", lambda values: values.min().date().isoformat()),
        wave_index=("wave_index", "first"),
        stop_count=("order_id", "size"),
        valid_coordinate_stops=("coordinate_valid", "sum"),
        first_accept_time=("accept_dt", "first"),
        first_delivery_time=("delivery_dt", "first"),
        last_delivery_time=("delivery_dt", "last"),
        straight_line_distance_km=("adjacent_distance_km", "sum"),
        max_adjacent_distance_km=("adjacent_distance_km", "max"),
        max_service_radius_km=("distance_to_centroid_km", "max"),
        start_accept_lng=("accept_gps_lng", "first"),
        start_accept_lat=("accept_gps_lat", "first"),
        first_delivery_lng=("delivery_gps_lng", "first"),
        first_delivery_lat=("delivery_gps_lat", "first"),
        end_delivery_lng=("delivery_gps_lng", "last"),
        end_delivery_lat=("delivery_gps_lat", "last"),
    ).reset_index()
    summary["work_span_minutes"] = (
        summary["last_delivery_time"] - summary["first_delivery_time"]
    ).dt.total_seconds() / 60
    valid_start = within_shanghai_bounds(
        summary["start_accept_lng"], summary["start_accept_lat"]
    )
    summary["start_proxy_lng"] = summary["start_accept_lng"].where(
        valid_start, summary["first_delivery_lng"]
    )
    summary["start_proxy_lat"] = summary["start_accept_lat"].where(
        valid_start, summary["first_delivery_lat"]
    )
    summary["start_proxy_source"] = np.where(
        valid_start, "first_task_accept_gps", "first_delivery_gps_fallback"
    )
    summary["complete_coordinate_share"] = (
        summary["valid_coordinate_stops"] / summary["stop_count"]
    )

    node_columns = [
        "route_id",
        "stop_index",
        "region_id",
        "lng",
        "lat",
        "aoi_id",
        "aoi_type",
        "accept_dt",
        "delivery_dt",
        "accept_gps_lng",
        "accept_gps_lat",
        "delivery_gps_lng",
        "delivery_gps_lat",
        "coordinate_valid",
        "gap_minutes",
        "adjacent_distance_km",
        "implied_speed_kmh",
        "break_long_gap",
        "break_long_jump",
        "break_high_speed",
        "break_region_change",
        "break_invalid_coordinate",
    ]
    public_summary_columns = [
        "route_id",
        "service_date",
        "wave_index",
        "stop_count",
        "valid_coordinate_stops",
        "complete_coordinate_share",
        "first_accept_time",
        "first_delivery_time",
        "last_delivery_time",
        "work_span_minutes",
        "straight_line_distance_km",
        "max_adjacent_distance_km",
        "max_service_radius_km",
        "start_proxy_lng",
        "start_proxy_lat",
        "start_proxy_source",
        "end_delivery_lng",
        "end_delivery_lat",
    ]
    route_summary = summary[public_summary_columns].copy()
    report = {
        "source_rows": int(len(data)),
        "courier_days": int(day_groups.ngroups),
        "recovered_waves": int(data["route_id"].nunique()),
        "break_rules": {
            "long_gap_minutes": LONG_GAP_MINUTES,
            "long_jump_km": LONG_JUMP_KM,
            "max_plausible_speed_kmh": MAX_PLAUSIBLE_SPEED_KMH,
            "region_change": True,
            "invalid_coordinate_pair": True,
        },
        "break_counts": {
            "day_start": int(day_start.sum()),
            "long_gap": int(data["break_long_gap"].sum()),
            "long_jump": int(data["break_long_jump"].sum()),
            "high_speed": int(data["break_high_speed"].sum()),
            "region_change": int(data["break_region_change"].sum()),
            "invalid_coordinate": int(data["break_invalid_coordinate"].sum()),
        },
        "route_stop_count": quantiles(route_summary["stop_count"]),
        "route_work_span_minutes": quantiles(route_summary["work_span_minutes"]),
        "route_straight_line_distance_km": quantiles(
            route_summary["straight_line_distance_km"]
        ),
        "route_max_service_radius_km": quantiles(
            route_summary["max_service_radius_km"]
        ),
        "routes_with_at_least_10_stops": int(
            (route_summary["stop_count"] >= 10).sum()
        ),
        "routes_with_complete_coordinates": int(
            (route_summary["complete_coordinate_share"] == 1).sum()
        ),
        "complete_coordinate_route_share": float(
            (route_summary["complete_coordinate_share"] == 1).mean()
        ),
        "start_proxy_fallback_routes": int((~valid_start).sum()),
        "endpoint_note": "The first task accept GPS is a wave-start proxy, not a verified depot. The final delivery GPS is only the observed route endpoint.",
        "privacy_note": "Raw courier_id and order_id are omitted from processed outputs; route_id is sequential and dataset-local.",
    }
    return data[node_columns], route_summary, report


def main() -> None:
    args = parse_args()
    source_path = next(args.root.glob("data/delivery_sh-*.parquet"), None)
    if source_path is None:
        raise FileNotFoundError("没有找到上海 LaDe-D Parquet 文件")
    frame = pd.read_parquet(source_path)
    nodes, routes, report = recover_routes(frame)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(
        args.processed_dir / "shanghai_historical_routes.parquet", index=False
    )
    routes.to_csv(args.processed_dir / "shanghai_route_summary.csv", index=False)
    (args.result_dir / "route_recovery_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
