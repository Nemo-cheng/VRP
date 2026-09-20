#!/usr/bin/env python3
"""Evaluate candidate vehicles for verified multiday long-haul tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from experiment.evaluate_candidate_vehicle_feasibility import candidate_limits
from experiment.prepare_excel_transport_data import build_transport_events, load_data
from experiment.validate_parameter_registry import load_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估跨日长途任务的候选车型适配性。")
    parser.add_argument("--orders", type=Path, default=Path("订单数据.xlsx"))
    parser.add_argument(
        "--classification",
        type=Path,
        default=Path(
            "results/company_transport/final_test/overlong_task_classification.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/company_transport/final_test"),
    )
    parser.add_argument("--maximum-hours", type=float, default=14.0)
    return parser.parse_args()


def build_multiday_requirements(
    classification: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    selected = classification[
        classification["included_in_multiday_analysis"].astype(bool)
    ].copy()
    event_payload = events[["event_id", "total_weight_kg"]].rename(
        columns={"event_id": "task_id"}
    )
    requirements = selected.merge(
        event_payload, on="task_id", how="left", validate="one_to_one"
    )
    required = [
        "planning_duration_hours",
        "required_operating_days",
        "maximum_daily_distance_km",
        "total_weight_kg",
    ]
    if requirements[required].isna().any().any():
        raise ValueError("跨日任务存在无法关联的规划时长、日里程或载重")
    return requirements


def evaluate_multiday_candidates(
    requirements: pd.DataFrame,
    registry: dict[str, Any],
    maximum_hours: float = 14.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if maximum_hours <= 0:
        raise ValueError("maximum_hours must be positive")

    rows: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(
        registry["candidate_vehicles"], start=1
    ):
        candidate_id = f"candidate_{candidate_index:02d}"
        for scenario in ("lower_bound", "upper_bound"):
            payload_t, effective_range = candidate_limits(
                candidate, scenario, registry
            )
            for task in requirements.itertuples(index=False):
                time_minimum_days = int(task.required_operating_days)
                balanced_distance_at_time_minimum = (
                    float(task.distance_km) / time_minimum_days
                )
                range_minimum_days = math.ceil(
                    float(task.distance_km) / effective_range
                )
                conditional_days = max(time_minimum_days, range_minimum_days)
                payload_feasible = float(task.total_weight_kg) <= payload_t * 1000
                end_to_end_range_feasible = (
                    float(task.distance_km) <= effective_range
                )
                range_feasible_at_time_minimum = (
                    balanced_distance_at_time_minimum <= effective_range
                )
                balanced_daily_distance = float(task.distance_km) / conditional_days
                balanced_daily_hours = (
                    float(task.planning_duration_hours) / conditional_days
                )
                conditional_plan_within_limits = (
                    balanced_daily_distance <= effective_range
                    and balanced_daily_hours <= maximum_hours
                )
                rows.append(
                    {
                        "task_id": task.task_id,
                        "origin_site_id": task.origin_site_id,
                        "destination_site_id": task.destination_site_id,
                        "candidate_id": candidate_id,
                        "candidate_model": candidate["model"],
                        "energy": candidate["energy"],
                        "scenario": scenario,
                        "payload_limit_t": payload_t,
                        "effective_winter_2026_range_km": effective_range,
                        "task_payload_kg": task.total_weight_kg,
                        "task_distance_km": task.distance_km,
                        "planning_duration_hours": task.planning_duration_hours,
                        "time_minimum_operating_days": time_minimum_days,
                        "time_minimum_daily_distance_km": (
                            balanced_distance_at_time_minimum
                        ),
                        "range_minimum_operating_days": range_minimum_days,
                        "conditional_minimum_operating_days": conditional_days,
                        "additional_days_due_to_range": (
                            conditional_days - time_minimum_days
                        ),
                        "conditional_balanced_daily_distance_km": (
                            balanced_daily_distance
                        ),
                        "conditional_balanced_daily_hours": balanced_daily_hours,
                        "payload_feasible": payload_feasible,
                        "end_to_end_range_feasible_without_replenishment": (
                            end_to_end_range_feasible
                        ),
                        "range_feasible_at_time_minimum_days": (
                            range_feasible_at_time_minimum
                        ),
                        "conditional_plan_within_time_and_range": (
                            conditional_plan_within_limits
                        ),
                        "overnight_replenishment_verified": False,
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(
            ["candidate_id", "candidate_model", "energy", "scenario"],
            as_index=False,
        )
        .agg(
            evaluated_tasks=("task_id", "size"),
            payload_feasible_tasks=("payload_feasible", "sum"),
            end_to_end_range_feasible_tasks=(
                "end_to_end_range_feasible_without_replenishment",
                "sum",
            ),
            range_feasible_at_time_minimum_tasks=(
                "range_feasible_at_time_minimum_days",
                "sum",
            ),
            tasks_requiring_additional_days=(
                "additional_days_due_to_range",
                lambda values: int(values.gt(0).sum()),
            ),
            total_additional_days_due_to_range=(
                "additional_days_due_to_range",
                "sum",
            ),
            maximum_conditional_operating_days=(
                "conditional_minimum_operating_days",
                "max",
            ),
        )
    )
    summary["range_feasible_at_time_minimum_rate"] = (
        summary["range_feasible_at_time_minimum_tasks"]
        / summary["evaluated_tasks"]
    )
    return detail, summary


def main() -> None:
    args = parse_args()
    registry = load_registry()
    orders, waybills = load_data(args.orders)
    events, _ = build_transport_events(orders, waybills)
    requirements = build_multiday_requirements(
        pd.read_csv(args.classification), events
    )
    detail, summary = evaluate_multiday_candidates(
        requirements, registry, args.maximum_hours
    )
    result = {
        "source_only": ["订单数据.xlsx", "车辆数据.xlsx", "competition_brief"],
        "classified_multiday_tasks": len(requirements),
        "maximum_daily_operating_hours": args.maximum_hours,
        "maximum_hours_source": "competition_brief",
        "winter_and_degradation_applied": True,
        "spatial_relay_nodes_created": False,
        "overnight_replenishment_verified": False,
        "interpretation": {
            "time_minimum": "只按14小时运营窗口计算的最少天数。",
            "conditional_minimum": (
                "假设每日结束后可完全补能时，同时满足14小时和续航的最少天数。"
            ),
            "evidence_limit": (
                "订单数据没有夜间停靠点和补能设施，条件最少天数不能直接视为现实可行方案。"
            ),
        },
        "decision_scope": (
            "比较候选车型的载重、冬季衰减后续航和跨日运输天数，"
            "不包含途中补能选址、碳排放和成本排序。"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        args.output_dir / "multiday_vehicle_feasibility_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_dir / "multiday_vehicle_feasibility_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "multiday_vehicle_feasibility.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
