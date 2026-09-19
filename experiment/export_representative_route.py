#!/usr/bin/env python3
"""Export one anonymous, fully recomputable vehicle route."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出匿名代表性车辆路线。")
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path("processed/company/vrp/time_dependent_vrp_schedules.csv"),
    )
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/company_transport")
    )
    parser.add_argument("--output-prefix", default="representative_vehicle_route")
    parser.add_argument("--solution-label", default="time_dependent_vrp")
    return parser.parse_args()


def select_representative_route(schedule: pd.DataFrame) -> pd.DataFrame:
    route_sizes = (
        schedule.groupby(["instance_id", "vehicle_id"])
        .size()
        .rename("task_count")
        .reset_index()
        .sort_values(
            ["task_count", "instance_id", "vehicle_id"],
            ascending=[False, True, True],
        )
    )
    selected = route_sizes.iloc[0]
    route = schedule[
        schedule["instance_id"].eq(selected["instance_id"])
        & schedule["vehicle_id"].eq(selected["vehicle_id"])
    ].copy()
    route = route.sort_values("sequence").reset_index(drop=True)
    route["departed_at"] = pd.to_datetime(route["departed_at"])
    route["arrived_at"] = pd.to_datetime(route["arrived_at"])
    route_start = route["departed_at"].min()
    route["task_label"] = [f"task_{index:02d}" for index in range(1, len(route) + 1)]
    task_labels = dict(zip(route["task_id"], route["task_label"], strict=True))
    route["next_task_label"] = route["next_task_id"].map(task_labels)
    route["departure_offset_hours"] = (
        route["departed_at"] - route_start
    ).dt.total_seconds() / 3600
    route["arrival_offset_hours"] = (
        route["arrived_at"] - route_start
    ).dt.total_seconds() / 3600
    next_departure = route["departed_at"].shift(-1)
    route["waiting_after_deadhead_hours"] = (
        next_departure
        - route["arrived_at"]
        - pd.to_timedelta(route["deadhead_to_next_duration_hours"], unit="h")
    ).dt.total_seconds() / 3600
    route.loc[route.index[-1], "waiting_after_deadhead_hours"] = pd.NA
    return route[
        [
            "task_label",
            "sequence",
            "vehicle_type_name",
            "origin_site_id",
            "destination_site_id",
            "loaded_path",
            "distance_km",
            "departure_offset_hours",
            "arrival_offset_hours",
            "next_task_label",
            "deadhead_to_next_path",
            "deadhead_to_next_distance_km",
            "deadhead_to_next_duration_hours",
            "waiting_after_deadhead_hours",
        ]
    ]


def summarize_route(route: pd.DataFrame) -> dict[str, object]:
    return {
        "source_only": "订单数据.xlsx",
        "privacy": "Dates, vehicle identifiers and original site names are omitted.",
        "task_count": len(route),
        "vehicle_type_name": str(route["vehicle_type_name"].iloc[0]),
        "loaded_distance_km": float(route["distance_km"].sum()),
        "internal_deadhead_distance_km": float(
            route["deadhead_to_next_distance_km"].sum()
        ),
        "route_elapsed_hours": float(route["arrival_offset_hours"].max()),
        "minimum_waiting_after_deadhead_hours": float(
            route["waiting_after_deadhead_hours"].dropna().min()
        ),
        "vehicle_type_consistent": bool(route["vehicle_type_name"].nunique() == 1),
        "all_deadhead_links_time_feasible": bool(
            route["waiting_after_deadhead_hours"].dropna().ge(0).all()
        ),
    }


def main() -> None:
    args = parse_args()
    schedule = pd.read_csv(args.schedule, low_memory=False)
    route = select_representative_route(schedule)
    summary = summarize_route(route)
    summary["solution_label"] = args.solution_label
    route.to_csv(
        args.result_dir / f"{args.output_prefix}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.result_dir / f"{args.output_prefix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
