#!/usr/bin/env python3
"""Build an evidence-bounded report for the two retained P90 solutions."""

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
    parser = argparse.ArgumentParser(description="汇总两套保留的 P90 VRP 方案。")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/company_transport/final_test"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_dual_solution_report(
    strict: dict[str, object], compromise: dict[str, object]
) -> tuple[pd.DataFrame, dict[str, object]]:
    historical_vehicles = int(compromise["historical_p90_chain_vehicle_count"])
    historical_deadhead = float(compromise["historical_p90_chain_deadhead_distance_km"])
    rows = [
        {
            "solution": "minimum_vehicle_p90",
            "decision_role": "vehicle compression bound",
            "vehicle_count": int(strict["p90_robust_vehicle_count"]),
            "internal_deadhead_distance_km": float(
                strict["internal_deadhead_distance_km"]
            ),
            "vehicle_reduction_vs_history": historical_vehicles
            - int(strict["p90_robust_vehicle_count"]),
            "deadhead_reduction_vs_history_km": historical_deadhead
            - float(strict["internal_deadhead_distance_km"]),
            "task_service_rate": float(strict["task_service_rate"]),
            "constraint_violations": int(strict["route_type_violations"])
            + int(strict["time_overlap_violations"])
            + int(strict["deadhead_endpoint_violations"])
            + int(strict["missing_selected_link_paths"]),
        },
        {
            "solution": "historical_deadhead_controlled_p90",
            "decision_role": "historical deadhead controlled compromise",
            "vehicle_count": int(compromise["recommended_vehicle_count"]),
            "internal_deadhead_distance_km": float(
                compromise["recommended_deadhead_distance_km"]
            ),
            "vehicle_reduction_vs_history": int(compromise["vehicle_reduction"]),
            "deadhead_reduction_vs_history_km": float(
                compromise["deadhead_reduction_km"]
            ),
            "task_service_rate": float(compromise["task_service_rate"]),
            "constraint_violations": int(compromise["route_type_violations"])
            + int(compromise["time_overlap_violations"])
            + int(compromise["deadhead_endpoint_violations"])
            + int(compromise["missing_selected_link_paths"]),
        },
    ]
    table = pd.DataFrame(rows)
    report = {
        "source_only": "订单数据.xlsx",
        "travel_time_stat": "p90",
        "instances": int(compromise["instances"]),
        "tasks": int(compromise["tasks"]),
        "historical_p90_chain_vehicle_count": historical_vehicles,
        "historical_p90_chain_deadhead_distance_km": historical_deadhead,
        "retained_solutions": table.to_dict(orient="records"),
        "selection_status": "both_retained_no_economic_ranking",
        "reason": (
            "The order data contain no observed vehicle fixed cost or per-kilometre "
            "operating cost, so they cannot identify a unique economic optimum."
        ),
        "vehicle_cost_equivalent_km_role": (
            "The frozen 75 km value identifies the independently selected compromise "
            "and is not an observed monetary cost."
        ),
        "all_retained_solutions_feasible": bool(
            table["task_service_rate"].eq(1.0).all()
            and table["constraint_violations"].eq(0).all()
        ),
    }
    return table, report


def main() -> None:
    args = parse_args()
    strict = read_json(args.result_dir / "robust_p90_vrp_summary.json")
    compromise = read_json(
        args.result_dir / "independently_selected_p90_solution_summary.json"
    )
    table, report = build_dual_solution_report(strict, compromise)
    table.to_csv(
        args.result_dir / "retained_p90_solutions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.result_dir / "retained_p90_solutions.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
