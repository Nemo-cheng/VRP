#!/usr/bin/env python3
"""Compare holdout VRP solutions with verifiable historical vehicle chains."""

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
    parser = argparse.ArgumentParser(description="比较历史车辆分配与时间外 VRP 结果。")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("processed/company/vrp_holdout"),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/company_transport/holdout"),
    )
    return parser.parse_args()


def attach_qualified_instances(
    tasks: pd.DataFrame, instances: pd.DataFrame
) -> pd.DataFrame:
    qualified = instances.loc[
        instances["qualifies_for_vrp"].astype(str).str.lower().eq("true"),
        ["instance_id", "service_date", "component_id"],
    ].copy()
    qualified["service_date"] = qualified["service_date"].astype(str)
    prepared = tasks.copy()
    prepared["service_date"] = prepared["service_date"].astype(str)
    prepared = prepared.merge(
        qualified,
        on=["service_date", "component_id"],
        how="inner",
        validate="many_to_one",
    )
    prepared["departed_at"] = pd.to_datetime(prepared["departed_at"])
    prepared["arrived_at"] = pd.to_datetime(prepared["arrived_at"])
    return prepared


def build_historical_chains(
    tasks: pd.DataFrame,
    feasible_links: pd.DataFrame,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    link_set = set(
        feasible_links[["instance_id", "from_task_id", "to_task_id"]].itertuples(
            index=False, name=None
        )
    )
    ordered = tasks.sort_values(
        ["instance_id", "historical_vehicle", "departed_at", "task_id"],
        na_position="last",
    )
    rows: list[dict[str, object]] = []
    diagnostics = {
        "observed_consecutive_transitions": 0,
        "verified_consecutive_transitions": 0,
        "transitions_split_by_infeasibility": 0,
        "transitions_split_by_vehicle_type_change": 0,
        "tasks_with_missing_historical_vehicle": 0,
    }

    for instance_id, instance_tasks in ordered.groupby("instance_id", sort=True):
        chain_number = 0
        vehicle_values = instance_tasks["historical_vehicle"].astype("string")
        missing = vehicle_values.isna() | vehicle_values.str.strip().eq("")
        diagnostics["tasks_with_missing_historical_vehicle"] += int(missing.sum())
        grouping_key = vehicle_values.where(
            ~missing,
            "__missing__" + instance_tasks["task_id"].astype(str),
        )
        keyed = instance_tasks.assign(_vehicle_group=grouping_key)

        for _, vehicle_tasks in keyed.groupby("_vehicle_group", sort=True):
            vehicle_tasks = vehicle_tasks.sort_values(["departed_at", "task_id"])
            previous = None
            sequence = 0
            for task in vehicle_tasks.itertuples(index=False):
                start_new_chain = previous is None
                if previous is not None:
                    diagnostics["observed_consecutive_transitions"] += 1
                    same_type = str(previous.vehicle_type_name) == str(
                        task.vehicle_type_name
                    )
                    link_is_feasible = (
                        instance_id,
                        previous.task_id,
                        task.task_id,
                    ) in link_set
                    if not same_type:
                        diagnostics["transitions_split_by_vehicle_type_change"] += 1
                    elif link_is_feasible:
                        diagnostics["verified_consecutive_transitions"] += 1
                    else:
                        diagnostics["transitions_split_by_infeasibility"] += 1
                    start_new_chain = not (same_type and link_is_feasible)

                if start_new_chain:
                    chain_number += 1
                    sequence = 1
                else:
                    sequence += 1
                rows.append(
                    {
                        "instance_id": instance_id,
                        "baseline_variant": variant,
                        "historical_chain_id": (
                            f"{instance_id}_{variant}_chain_{chain_number:04d}"
                        ),
                        "sequence": sequence,
                        "task_id": task.task_id,
                    }
                )
                previous = task

    chains = pd.DataFrame(rows)
    instance_metrics = (
        chains.groupby("instance_id")
        .agg(
            task_count=("task_id", "size"),
            verifiable_historical_chain_count=("historical_chain_id", "nunique"),
        )
        .reset_index()
    )
    diagnostics["verifiable_historical_chains"] = int(
        instance_metrics["verifiable_historical_chain_count"].sum()
    )
    return chains, instance_metrics, diagnostics


def raw_historical_counts(tasks: pd.DataFrame) -> pd.DataFrame:
    prepared = tasks.copy()
    values = prepared["historical_vehicle"].astype("string")
    missing = values.isna() | values.str.strip().eq("")
    prepared["_vehicle_group"] = values.where(
        ~missing, "__missing__" + prepared["task_id"].astype(str)
    )
    return (
        prepared.groupby("instance_id")
        .agg(
            task_count=("task_id", "size"),
            raw_historical_vehicle_count=("_vehicle_group", "nunique"),
        )
        .reset_index()
    )


def build_comparison(
    tasks: pd.DataFrame,
    p50_links: pd.DataFrame,
    p90_links: pd.DataFrame,
    basic: pd.DataFrame,
    vehicle_type: pd.DataFrame,
    p50: pd.DataFrame,
    p90: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw = raw_historical_counts(tasks)
    p50_chains, p50_metrics, p50_diagnostics = build_historical_chains(
        tasks, p50_links, "p50"
    )
    p90_chains, p90_metrics, p90_diagnostics = build_historical_chains(
        tasks, p90_links, "p90"
    )
    p50_metrics = p50_metrics.rename(
        columns={"verifiable_historical_chain_count": "p50_historical_chain_count"}
    ).drop(columns="task_count")
    p90_metrics = p90_metrics.rename(
        columns={"verifiable_historical_chain_count": "p90_historical_chain_count"}
    ).drop(columns="task_count")

    comparison = raw.merge(p50_metrics, on="instance_id", validate="one_to_one")
    comparison = comparison.merge(p90_metrics, on="instance_id", validate="one_to_one")
    comparison = comparison.merge(
        basic[["instance_id", "vrp_vehicle_count"]],
        on="instance_id",
        validate="one_to_one",
    )
    comparison = comparison.merge(
        vehicle_type[["instance_id", "type_compatible_vehicle_count"]],
        on="instance_id",
        validate="one_to_one",
    )
    comparison = comparison.merge(
        p50[["instance_id", "time_dependent_vehicle_count"]],
        on="instance_id",
        validate="one_to_one",
    )
    comparison = comparison.merge(
        p90[["instance_id", "p90_robust_vehicle_count"]],
        on="instance_id",
        validate="one_to_one",
    )
    comparison["p50_vehicle_reduction_vs_historical_chains"] = (
        comparison["p50_historical_chain_count"]
        - comparison["time_dependent_vehicle_count"]
    ) / comparison["p50_historical_chain_count"]
    comparison["p90_vehicle_reduction_vs_historical_chains"] = (
        comparison["p90_historical_chain_count"]
        - comparison["p90_robust_vehicle_count"]
    ) / comparison["p90_historical_chain_count"]

    totals = comparison.sum(numeric_only=True)
    p50_history = int(totals["p50_historical_chain_count"])
    p90_history = int(totals["p90_historical_chain_count"])
    p50_vehicles = int(totals["time_dependent_vehicle_count"])
    p90_vehicles = int(totals["p90_robust_vehicle_count"])
    summary = {
        "source_only": "订单数据.xlsx",
        "scope": "Chronological holdout tasks in qualified VRP instances.",
        "instances": len(comparison),
        "tasks": int(totals["task_count"]),
        "raw_historical_vehicle_count": int(totals["raw_historical_vehicle_count"]),
        "p50_historical_chain_count": p50_history,
        "p90_historical_chain_count": p90_history,
        "basic_vrp_vehicle_count": int(totals["vrp_vehicle_count"]),
        "type_compatible_vehicle_count": int(totals["type_compatible_vehicle_count"]),
        "p50_vrp_vehicle_count": p50_vehicles,
        "p90_vrp_vehicle_count": p90_vehicles,
        "p50_vehicle_reduction_vs_historical_chains": (p50_history - p50_vehicles)
        / p50_history,
        "p90_vehicle_reduction_vs_historical_chains": (p90_history - p90_vehicles)
        / p90_history,
        "p50_chain_validation": p50_diagnostics,
        "p90_chain_validation": p90_diagnostics,
        "interpretation_limit": (
            "Reductions apply to verifiable vehicle chains in the observed "
            "holdout sample, not to the company's complete fleet."
        ),
    }
    chains = pd.concat([p50_chains, p90_chains], ignore_index=True)
    return comparison, chains, summary


def main() -> None:
    args = parse_args()
    tasks = pd.read_csv(args.data_dir / "tasks.csv", low_memory=False)
    instances = pd.read_csv(args.data_dir / "instances.csv")
    qualified_tasks = attach_qualified_instances(tasks, instances)
    p50_links = pd.read_csv(args.data_dir / "time_dependent_candidate_task_links.csv")
    p90_links = pd.read_csv(args.data_dir / "robust_p90_candidate_task_links.csv")
    basic = pd.read_csv(args.result_dir / "basic_vrp_instance_comparison.csv")
    vehicle_type = pd.read_csv(args.result_dir / "type_compatible_vrp_comparison.csv")
    p50 = pd.read_csv(args.result_dir / "time_dependent_vrp_comparison.csv")
    p90 = pd.read_csv(args.result_dir / "robust_p90_vrp_comparison.csv")
    comparison, chains, summary = build_comparison(
        qualified_tasks, p50_links, p90_links, basic, vehicle_type, p50, p90
    )

    args.result_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        args.result_dir / "historical_vehicle_instance_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    chains.to_csv(
        args.data_dir / "historical_vehicle_chains.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.result_dir / "historical_vehicle_baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
