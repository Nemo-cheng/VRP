#!/usr/bin/env python3
"""Reconcile the final-test VRP subset with the carbon baseline scope."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openpyxl>=3.1",
#   "pandas>=2.2",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiment.build_carbon_baseline import build_baseline
from experiment.prepare_excel_transport_data import build_transport_events, load_data
from experiment.validate_parameter_registry import load_registry

JOIN_KEY = [
    "vehicle_id",
    "departed_at",
    "departure_site",
    "arrival_site",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对账VRP子集与碳基线范围。")
    parser.add_argument("--orders", type=Path, default=Path("订单数据.xlsx"))
    parser.add_argument("--vehicles", type=Path, default=Path("车辆数据.xlsx"))
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path(
            "processed/company/vrp_final_test/"
            "independently_selected_p90_solution_schedules.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/company_transport/final_test/vrp_carbon_scope.json"
        ),
    )
    return parser.parse_args()


def attach_vehicle_ids(events: pd.DataFrame) -> pd.DataFrame:
    result = events.copy()
    codes, _ = pd.factorize(result["vehicle"], sort=True)
    result["vehicle_id"] = pd.Series(codes, index=result.index).map(
        lambda value: f"vehicle_{value + 1:05d}"
    )
    return result


def reconcile_scope(
    baseline: pd.DataFrame,
    events: pd.DataFrame,
    schedule: pd.DataFrame,
) -> dict[str, object]:
    scheduled_task_ids = set(schedule["task_id"].astype(str))
    selected_events = events[events["event_id"].isin(scheduled_task_ids)].copy()
    if len(selected_events) != len(schedule):
        raise ValueError("VRP日程中的任务未全部匹配到运输事件")
    selected = selected_events.merge(
        baseline[
            [
                *JOIN_KEY,
                "emissions_kgco2_min",
                "emissions_kgco2_max",
            ]
        ],
        on=JOIN_KEY,
        how="left",
        validate="one_to_one",
    )
    if selected["emissions_kgco2_min"].isna().any():
        raise ValueError("VRP任务未全部匹配到碳基线")

    baseline_distance = float(baseline["distance_km"].sum())
    subset_distance = float(selected["distance_km"].sum())
    baseline_min = float(baseline["emissions_kgco2_min"].sum())
    baseline_max = float(baseline["emissions_kgco2_max"].sum())
    subset_min = float(selected["emissions_kgco2_min"].sum())
    subset_max = float(selected["emissions_kgco2_max"].sum())
    return {
        "carbon_baseline_eligible_legs": len(baseline),
        "vrp_final_test_tasks": len(selected),
        "task_scope_share": len(selected) / len(baseline),
        "carbon_baseline_distance_km": baseline_distance,
        "vrp_final_test_loaded_distance_km": subset_distance,
        "distance_scope_share": subset_distance / baseline_distance,
        "carbon_baseline_emissions_kgco2": {
            "minimum": baseline_min,
            "maximum": baseline_max,
        },
        "vrp_final_test_loaded_emissions_kgco2": {
            "minimum": subset_min,
            "maximum": subset_max,
        },
        "emission_scope_share": {
            "minimum_case": subset_min / baseline_min,
            "maximum_case": subset_max / baseline_max,
        },
        "scope_period": "2023-12 independent final test versus all observed months",
        "extrapolation_allowed": False,
        "reason": (
            "最终测试只包含12月网络连通且实例规模达标的任务，"
            "不是从全部有效运输段随机抽样。"
        ),
    }


def main() -> None:
    args = parse_args()
    registry = load_registry()
    baseline, _, _ = build_baseline(args.orders, args.vehicles, registry)
    orders, waybills = load_data(args.orders)
    events, _ = build_transport_events(orders, waybills)
    events = attach_vehicle_ids(events)
    schedule = pd.read_csv(args.schedule)
    summary = reconcile_scope(baseline, events, schedule)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
