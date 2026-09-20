#!/usr/bin/env python3
"""Separate fleet-chain reduction from route-distance reduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总路线方案的实际里程变化。")
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path(
            "processed/company/vrp_final_test/"
            "independently_selected_p90_solution_schedules.csv"
        ),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path(
            "results/company_transport/final_test/"
            "independently_selected_p90_validation_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/company_transport/final_test/route_distance_effect.json"
        ),
    )
    return parser.parse_args()


def summarize_distance_effect(
    schedule: pd.DataFrame,
    validation: dict[str, object],
) -> dict[str, object]:
    loaded_distance = float(schedule["distance_km"].sum())
    historical_deadhead = float(validation["historical_deadhead_distance_km"])
    recommended_deadhead = float(
        validation["recommended_deadhead_distance_km"]
    )
    historical_total = loaded_distance + historical_deadhead
    recommended_total = loaded_distance + recommended_deadhead
    distance_reduction = historical_total - recommended_total
    distance_reduction_rate = distance_reduction / historical_total
    return {
        "tasks": int(validation["tasks"]),
        "historical_vehicle_chains": int(
            validation["historical_vehicle_chain_count"]
        ),
        "recommended_vehicle_chains": int(
            validation["recommended_vehicle_count"]
        ),
        "vehicle_chain_reduction_rate": float(
            validation["aggregate_vehicle_reduction_rate"]
        ),
        "loaded_distance_km_both_solutions": loaded_distance,
        "historical_deadhead_distance_km": historical_deadhead,
        "recommended_deadhead_distance_km": recommended_deadhead,
        "historical_total_distance_km": historical_total,
        "recommended_total_distance_km": recommended_total,
        "total_distance_reduction_km": distance_reduction,
        "total_distance_reduction_rate": distance_reduction_rate,
        "loaded_distance_changed": False,
        "deadhead_reduction_95_ci_km": validation[
            "aggregate_deadhead_reduction_km_95_ci"
        ],
        "interpretation": (
            "车辆链减少反映所需车辆资产数量变化。载货任务及其里程保持不变，"
            "路线可直接归因的运输距离变化只来自内部空驶。"
        ),
        "carbon_inference": (
            "车型排放因子固定时，碳排放变化与总里程变化同方向；"
            "车型同时变化时必须另做逐段加权，不能用车辆链减少率代替减排率。"
        ),
    }


def main() -> None:
    args = parse_args()
    schedule = pd.read_csv(args.schedule)
    validation = json.loads(args.validation.read_text(encoding="utf-8-sig"))
    summary = summarize_distance_effect(schedule, validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
