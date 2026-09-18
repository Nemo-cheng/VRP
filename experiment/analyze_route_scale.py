#!/usr/bin/env python3
"""Summarize recovered LaDe route scale without claiming vehicle capacity fit."""

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

import pandas as pd

from audit_lade import haversine_km, within_shanghai_bounds

METRICS = [
    "stop_count",
    "work_span_minutes",
    "straight_line_distance_km",
    "max_adjacent_distance_km",
    "max_service_radius_km",
    "start_leg_km",
    "return_to_start_proxy_km",
    "closed_proxy_distance_km",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 LaDe-D 历史路线尺度。")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed/lade"))
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/lade_audit")
    )
    return parser.parse_args()


def add_endpoint_sensitivity(
    routes: pd.DataFrame, nodes: pd.DataFrame
) -> pd.DataFrame:
    ordered = nodes.sort_values(["route_id", "stop_index"], kind="stable")
    endpoints = ordered.groupby("route_id", sort=False).agg(
        first_delivery_lng=("delivery_gps_lng", "first"),
        first_delivery_lat=("delivery_gps_lat", "first"),
        last_delivery_lng=("delivery_gps_lng", "last"),
        last_delivery_lat=("delivery_gps_lat", "last"),
    )
    output = routes.merge(endpoints, on="route_id", how="left", validate="one_to_one")
    output["start_leg_km"] = haversine_km(
        output["start_proxy_lng"],
        output["start_proxy_lat"],
        output["first_delivery_lng"],
        output["first_delivery_lat"],
    )
    output["return_to_start_proxy_km"] = haversine_km(
        output["last_delivery_lng"],
        output["last_delivery_lat"],
        output["start_proxy_lng"],
        output["start_proxy_lat"],
    )
    valid_endpoints = (
        within_shanghai_bounds(
            output["start_proxy_lng"], output["start_proxy_lat"]
        )
        & within_shanghai_bounds(
            output["first_delivery_lng"], output["first_delivery_lat"]
        )
        & within_shanghai_bounds(
            output["last_delivery_lng"], output["last_delivery_lat"]
        )
    )
    output.loc[
        ~valid_endpoints, ["start_leg_km", "return_to_start_proxy_km"]
    ] = pd.NA
    output["closed_proxy_distance_km"] = (
        output["straight_line_distance_km"]
        + output["start_leg_km"]
        + output["return_to_start_proxy_km"]
    )
    return output


def metric_rows(frame: pd.DataFrame, population: str) -> list[dict[str, object]]:
    rows = []
    for metric in METRICS:
        values = frame[metric].dropna()
        rows.append(
            {
                "population": population,
                "metric": metric,
                "route_count": int(len(values)),
                "mean": float(values.mean()),
                "p50": float(values.quantile(0.50)),
                "p90": float(values.quantile(0.90)),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
            }
        )
    return rows


def analyze_scale(
    routes: pd.DataFrame, nodes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    enriched = add_endpoint_sensitivity(routes, nodes)
    eligible_mask = (
        (enriched["stop_count"] >= 10)
        & (enriched["complete_coordinate_share"] == 1)
    )
    eligible = enriched.loc[eligible_mask].copy()
    scale_summary = pd.DataFrame(
        metric_rows(enriched, "all_recovered_waves")
        + metric_rows(eligible, "eligible_at_least_10_stops")
    )

    assessment = pd.DataFrame(
        [
            {
                "vehicle_class": "cargo_two_wheeler",
                "geometry_status": "candidate_for_compact_routes",
                "required_before_final_selection": "payload, road access, range, cost and emission parameters",
            },
            {
                "vehicle_class": "cargo_tricycle",
                "geometry_status": "candidate_for_compact_and_medium_routes",
                "required_before_final_selection": "payload, road access, range, cost and emission parameters",
            },
            {
                "vehicle_class": "microvan",
                "geometry_status": "candidate_for_medium_and_extended_routes",
                "required_before_final_selection": "payload, road access, range, cost and emission parameters",
            },
            {
                "vehicle_class": "4.2m_box_truck",
                "geometry_status": "capacity_case_only",
                "required_before_final_selection": "payload proof, urban access, range, cost and emission parameters",
            },
        ]
    )
    report = {
        "all_recovered_waves": int(len(enriched)),
        "eligible_routes": int(len(eligible)),
        "eligibility_rule": "At least 10 stops and 100% valid delivery coordinates.",
        "eligible_share": float(eligible_mask.mean()),
        "eligible_orders": int(eligible["stop_count"].sum()),
        "start_proxy_sensitivity": {
            "interpretation": "Closed proxy distance adds the start-to-first and last-to-start straight-line legs. The start proxy is not a verified depot.",
            "median_open_distance_km": float(
                eligible["straight_line_distance_km"].median()
            ),
            "median_closed_proxy_distance_km": float(
                eligible["closed_proxy_distance_km"].median()
            ),
            "p90_closed_proxy_distance_km": float(
                eligible["closed_proxy_distance_km"].quantile(0.90)
            ),
        },
        "distance_limit": "All distances are great-circle lower bounds. Road-network distance is deferred to stage C.",
        "vehicle_decision": "Geometry supports retaining four candidate classes for later checks. Final vehicle selection requires stage C road access, stage D route payloads, and stage E vehicle parameters.",
    }
    return scale_summary, assessment, report


def main() -> None:
    args = parse_args()
    routes = pd.read_csv(
        args.processed_dir / "shanghai_route_summary.csv",
        parse_dates=["first_accept_time", "first_delivery_time", "last_delivery_time"],
    )
    nodes = pd.read_parquet(args.processed_dir / "shanghai_historical_routes.parquet")
    summary, assessment, report = analyze_scale(routes, nodes)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.result_dir / "route_scale_summary.csv", index=False)
    assessment.to_csv(args.result_dir / "vehicle_scope_assessment.csv", index=False)
    (args.result_dir / "route_scale_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
