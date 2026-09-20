#!/usr/bin/env python3
"""Classify overlong test tasks using training-period lane evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

LANE_KEY = ["origin_site_id", "destination_site_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用训练期线路P90分类最终测试中的超时任务。"
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path(
            "processed/company/vrp_final_test/"
            "independently_selected_p90_solution_schedules.csv"
        ),
    )
    parser.add_argument(
        "--training-edges",
        type=Path,
        default=Path("processed/company/vrp_final_test/network_edges.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/company_transport/final_test"),
    )
    parser.add_argument("--maximum-hours", type=float, default=14.0)
    return parser.parse_args()


def classify_overlong_tasks(
    schedule: pd.DataFrame,
    training_edges: pd.DataFrame,
    maximum_hours: float = 14.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Separate verified multiday lanes from deferred time anomalies."""
    if maximum_hours <= 0:
        raise ValueError("maximum_hours must be positive")

    frame = schedule.copy()
    frame["departed_at"] = pd.to_datetime(frame["departed_at"], errors="raise")
    frame["arrived_at"] = pd.to_datetime(frame["arrived_at"], errors="raise")
    frame["observed_duration_hours"] = (
        frame["arrived_at"] - frame["departed_at"]
    ).dt.total_seconds() / 3600
    overlong = frame[frame["observed_duration_hours"].gt(maximum_hours)].copy()

    evidence_columns = LANE_KEY + [
        "observations",
        "distance_km_p50",
        "duration_hours_p50",
        "duration_hours_p90",
        "high_confidence",
    ]
    overlong = overlong.merge(
        training_edges[evidence_columns], on=LANE_KEY, how="left", validate="many_to_one"
    )
    verified = (
        overlong["high_confidence"].fillna(False).astype(bool)
        & overlong["duration_hours_p90"].gt(maximum_hours)
    )
    overlong["classification"] = "deferred_time_anomaly"
    overlong.loc[verified, "classification"] = "verified_multiday_long_haul"
    overlong["included_in_multiday_analysis"] = verified
    overlong["planning_duration_hours"] = overlong["duration_hours_p90"].where(
        verified
    )
    overlong["required_operating_days"] = pd.Series(pd.NA, index=overlong.index, dtype="Int64")
    overlong.loc[verified, "required_operating_days"] = overlong.loc[
        verified, "planning_duration_hours"
    ].map(lambda value: math.ceil(float(value) / maximum_hours))
    overlong["maximum_daily_distance_km"] = (
        overlong["distance_km"]
        * maximum_hours
        / overlong["planning_duration_hours"]
    ).where(verified)
    overlong["classification_basis"] = (
        "训练期高置信线路P90超过14小时"
    )
    overlong.loc[~verified, "classification_basis"] = (
        "测试任务超过14小时，但训练期线路P90未超过14小时，原始时长暂不采用"
    )

    output_columns = [
        "task_id",
        "instance_id",
        "vehicle_id",
        "origin_site_id",
        "destination_site_id",
        "distance_km",
        "observed_duration_hours",
        "observations",
        "distance_km_p50",
        "duration_hours_p50",
        "duration_hours_p90",
        "classification",
        "classification_basis",
        "included_in_multiday_analysis",
        "planning_duration_hours",
        "required_operating_days",
        "maximum_daily_distance_km",
    ]
    detail = overlong[output_columns].sort_values(
        ["classification", "origin_site_id", "destination_site_id", "task_id"]
    )
    included = detail[detail["included_in_multiday_analysis"]]
    deferred = detail[~detail["included_in_multiday_analysis"]]
    summary = {
        "schedule": "independently_selected_p90_solution",
        "classification_evidence": "training_period_lane_statistics_only",
        "test_period_used_for_threshold_estimation": False,
        "maximum_daily_operating_hours": maximum_hours,
        "maximum_hours_source": "competition_brief",
        "overlong_tasks": len(detail),
        "verified_multiday_long_haul_tasks": len(included),
        "verified_multiday_long_haul_distance_km": float(included["distance_km"].sum()),
        "deferred_time_anomaly_tasks": len(deferred),
        "deferred_time_anomaly_distance_km": float(deferred["distance_km"].sum()),
        "multiday_operating_days_distribution": {
            str(int(days)): int(count)
            for days, count in included["required_operating_days"]
            .value_counts()
            .sort_index()
            .items()
        },
        "spatial_relay_nodes_created": False,
        "overnight_energy_replenishment_verified": False,
        "decision": (
            "训练期高置信线路P90超过14小时的任务进入跨日运营分析。"
            "其余超时任务作为时间异常暂缓，不采用原始时长，也不进入本轮结果。"
        ),
    }
    return detail.reset_index(drop=True), summary


def main() -> None:
    args = parse_args()
    detail, summary = classify_overlong_tasks(
        pd.read_csv(args.schedule),
        pd.read_csv(args.training_edges),
        args.maximum_hours,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        args.output_dir / "overlong_task_classification.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "overlong_task_classification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
