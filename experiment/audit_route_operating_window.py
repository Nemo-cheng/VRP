#!/usr/bin/env python3
"""Audit retained VRP schedules against the competition operating window."""

from __future__ import annotations

import argparse
import json
from datetime import time
from pathlib import Path

import pandas as pd

CHAIN_KEY = ["instance_id", "vehicle_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计最终路线的每日运营窗口约束。")
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path(
            "processed/company/vrp_final_test/"
            "independently_selected_p90_solution_schedules.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/company_transport/final_test"),
    )
    return parser.parse_args()


def within_clock_window(
    departed_at: pd.Series,
    arrived_at: pd.Series,
    start: time = time(6, 0),
    end: time = time(20, 0),
) -> pd.Series:
    same_day = departed_at.dt.normalize().eq(arrived_at.dt.normalize())
    return (
        same_day
        & departed_at.dt.time.map(lambda value: value >= start)
        & arrived_at.dt.time.map(lambda value: value <= end)
    )


def audit_schedule(
    schedule: pd.DataFrame,
    maximum_hours: float = 14.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = schedule.copy()
    frame["departed_at"] = pd.to_datetime(frame["departed_at"], errors="raise")
    frame["arrived_at"] = pd.to_datetime(frame["arrived_at"], errors="raise")
    frame["task_duration_hours"] = (
        frame["arrived_at"] - frame["departed_at"]
    ).dt.total_seconds() / 3600
    frame["task_within_0600_2000"] = within_clock_window(
        frame["departed_at"], frame["arrived_at"]
    )
    frame["task_duration_within_14h"] = frame["task_duration_hours"].le(
        maximum_hours
    )

    chains = (
        frame.groupby(CHAIN_KEY, as_index=False)
        .agg(
            tasks=("task_id", "size"),
            first_departure=("departed_at", "min"),
            last_arrival=("arrived_at", "max"),
            loaded_distance_km=("distance_km", "sum"),
            deadhead_distance_km=("deadhead_to_next_distance_km", "sum"),
            tasks_within_clock_window=("task_within_0600_2000", "sum"),
            tasks_within_duration_limit=("task_duration_within_14h", "sum"),
        )
    )
    chains["chain_elapsed_hours"] = (
        chains["last_arrival"] - chains["first_departure"]
    ).dt.total_seconds() / 3600
    chains["all_tasks_within_clock_window"] = chains[
        "tasks_within_clock_window"
    ].eq(chains["tasks"])
    chains["all_tasks_within_duration_limit"] = chains[
        "tasks_within_duration_limit"
    ].eq(chains["tasks"])
    chains["chain_within_duration_limit"] = chains["chain_elapsed_hours"].le(
        maximum_hours
    )
    chains["operating_window_compliant"] = (
        chains["all_tasks_within_clock_window"]
        & chains["all_tasks_within_duration_limit"]
        & chains["chain_within_duration_limit"]
    )
    tasks_outside = int((~frame["task_within_0600_2000"]).sum())
    tasks_over_duration = int((~frame["task_duration_within_14h"]).sum())
    compliant_chains = int(chains["operating_window_compliant"].sum())
    summary = {
        "schedule": "independently_selected_p90_solution",
        "constraint_source": "competition_brief",
        "clock_window": "06:00-20:00",
        "maximum_daily_operating_hours": maximum_hours,
        "tasks": len(frame),
        "tasks_outside_clock_window": tasks_outside,
        "tasks_outside_clock_window_rate": tasks_outside / len(frame),
        "tasks_over_14h": tasks_over_duration,
        "vehicle_chains": len(chains),
        "chains_over_14h": int(
            (~chains["chain_within_duration_limit"]).sum()
        ),
        "chains_with_out_of_window_task": int(
            (~chains["all_tasks_within_clock_window"]).sum()
        ),
        "operating_window_compliant_chains": compliant_chains,
        "operating_window_compliant_chain_rate": compliant_chains / len(chains),
        "decision": (
            "Only compliant chains may enter vehicle replacement feasibility. "
            "Noncompliant tasks require a separate rescheduling model because "
            "the dataset has no customer service time windows."
        ),
    }
    return chains, summary


def main() -> None:
    args = parse_args()
    schedule = pd.read_csv(args.schedule)
    chains, summary = audit_schedule(schedule)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chains.to_csv(
        args.output_dir / "operating_window_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "operating_window_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
