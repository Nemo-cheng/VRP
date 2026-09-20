#!/usr/bin/env python3
"""Audit whether VRP chains can be retimed into one 14-hour operating shift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiment.audit_route_operating_window import CHAIN_KEY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计路线链能否重排进14小时班次。")
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


def audit_retimed_schedule(
    schedule: pd.DataFrame,
    maximum_hours: float = 14.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = schedule.copy()
    frame["departed_at"] = pd.to_datetime(frame["departed_at"], errors="raise")
    frame["arrived_at"] = pd.to_datetime(frame["arrived_at"], errors="raise")
    frame["task_duration_hours"] = (
        frame["arrived_at"] - frame["departed_at"]
    ).dt.total_seconds() / 3600
    frame["task_duration_within_limit"] = frame["task_duration_hours"].le(
        maximum_hours
    )
    chains = (
        frame.groupby(CHAIN_KEY, as_index=False)
        .agg(
            tasks=("task_id", "size"),
            loaded_duration_hours=("task_duration_hours", "sum"),
            deadhead_duration_hours=(
                "deadhead_to_next_duration_hours",
                "sum",
            ),
            loaded_distance_km=("distance_km", "sum"),
            deadhead_distance_km=("deadhead_to_next_distance_km", "sum"),
            tasks_within_duration_limit=(
                "task_duration_within_limit",
                "sum",
            ),
        )
    )
    chains["retimed_shift_hours"] = (
        chains["loaded_duration_hours"] + chains["deadhead_duration_hours"]
    )
    chains["all_tasks_within_duration_limit"] = chains[
        "tasks_within_duration_limit"
    ].eq(chains["tasks"])
    chains["retimed_shift_within_limit"] = chains["retimed_shift_hours"].le(
        maximum_hours
    )
    chains["retimed_operating_window_compliant"] = (
        chains["all_tasks_within_duration_limit"]
        & chains["retimed_shift_within_limit"]
    )
    compliant = chains["retimed_operating_window_compliant"]
    summary = {
        "schedule": "independently_selected_p90_solution",
        "constraint_source": "competition_brief",
        "maximum_daily_operating_hours": maximum_hours,
        "retiming_rule": (
            "保留任务顺序、实际载货行驶时长和P90空驶时长，"
            "不把历史等待时间视为客户时间窗。"
        ),
        "customer_time_windows_available": False,
        "vehicle_chains": len(chains),
        "retimed_compliant_chains": int(compliant.sum()),
        "retimed_compliant_chain_rate": float(compliant.mean()),
        "tasks": len(frame),
        "tasks_on_retimed_compliant_chains": int(
            chains.loc[compliant, "tasks"].sum()
        ),
        "tasks_on_retimed_noncompliant_chains": int(
            chains.loc[~compliant, "tasks"].sum()
        ),
        "individual_tasks_over_14h": int(
            (~frame["task_duration_within_limit"]).sum()
        ),
        "chains_over_14h_after_retiming": int(
            (~chains["retimed_shift_within_limit"]).sum()
        ),
        "decision": (
            "重排后合规链可进入车辆替换可行性实验。超长任务需要中继拆分、"
            "多日运输或保留其他运营方案，订单数据不足以自动选择处理方式。"
        ),
    }
    return chains, summary


def main() -> None:
    args = parse_args()
    schedule = pd.read_csv(args.schedule)
    chains, summary = audit_retimed_schedule(schedule)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chains.to_csv(
        args.output_dir / "retimed_operating_window_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "retimed_operating_window_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
