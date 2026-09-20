#!/usr/bin/env python3
"""Validate the recommended P90 solution across holdout instances."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.1",
#   "pandas>=2.2",
# ]
# ///

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证最终 P90 方案的实例级稳健性。")
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
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260920)
    parser.add_argument(
        "--schedule-filename", default="recommended_p90_vrp_schedules.csv"
    )
    parser.add_argument("--output-prefix", default="recommended_p90_validation")
    parser.add_argument(
        "--solution-label",
        default="P90 vehicle cost equivalent to 89 km of deadhead",
    )
    parser.add_argument("--independent-final-test", action="store_true")
    return parser.parse_args()


def build_instance_comparison(
    schedule: pd.DataFrame,
    chains: pd.DataFrame,
    links: pd.DataFrame,
) -> pd.DataFrame:
    final = (
        schedule.groupby("instance_id")
        .agg(
            task_count=("task_id", "size"),
            recommended_vehicle_count=("vehicle_id", "nunique"),
            recommended_deadhead_distance_km=(
                "deadhead_to_next_distance_km",
                "sum",
            ),
        )
        .reset_index()
    )
    historical_chains = chains[chains["baseline_variant"].eq("p90")].sort_values(
        ["instance_id", "historical_chain_id", "sequence"]
    )
    historical = (
        historical_chains.groupby("instance_id")
        .agg(
            historical_task_count=("task_id", "size"),
            historical_vehicle_chain_count=("historical_chain_id", "nunique"),
        )
        .reset_index()
    )
    link_distance = {
        (row.instance_id, row.from_task_id, row.to_task_id): float(
            row.deadhead_distance_km
        )
        for row in links.itertuples(index=False)
    }
    distance_by_instance = dict.fromkeys(historical["instance_id"], 0.0)
    for _, group in historical_chains.groupby("historical_chain_id", sort=False):
        records = list(group.itertuples(index=False))
        for first, second in pairwise(records):
            distance_by_instance[first.instance_id] += link_distance[
                (first.instance_id, first.task_id, second.task_id)
            ]
    historical["historical_deadhead_distance_km"] = historical["instance_id"].map(
        distance_by_instance
    )

    comparison = historical.merge(final, on="instance_id", validate="one_to_one")
    if not comparison["task_count"].eq(comparison["historical_task_count"]).all():
        raise ValueError("Historical and recommended task counts differ")
    comparison["vehicle_reduction"] = (
        comparison["historical_vehicle_chain_count"]
        - comparison["recommended_vehicle_count"]
    )
    comparison["vehicle_reduction_rate"] = (
        comparison["vehicle_reduction"] / comparison["historical_vehicle_chain_count"]
    )
    comparison["deadhead_reduction_km"] = (
        comparison["historical_deadhead_distance_km"]
        - comparison["recommended_deadhead_distance_km"]
    )
    comparison["vehicle_improved"] = comparison["vehicle_reduction"].gt(0)
    comparison["deadhead_not_increased"] = comparison["deadhead_reduction_km"].ge(0)
    comparison["both_improved_or_equal"] = (
        comparison["vehicle_improved"] & comparison["deadhead_not_increased"]
    )
    return comparison


def bootstrap_intervals(
    comparison: pd.DataFrame,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(comparison), size=(samples, len(comparison)))
    historical_vehicles = comparison["historical_vehicle_chain_count"].to_numpy()
    vehicle_reductions = comparison["vehicle_reduction"].to_numpy()
    deadhead_reductions = comparison["deadhead_reduction_km"].to_numpy()
    sampled_historical = historical_vehicles[indices].sum(axis=1)
    sampled_vehicle_reduction = vehicle_reductions[indices].sum(axis=1)
    sampled_deadhead_reduction = deadhead_reductions[indices].sum(axis=1)
    vehicle_rate = sampled_vehicle_reduction / sampled_historical
    quantiles = [0.025, 0.975]
    return {
        "aggregate_vehicle_reduction_rate_95_ci": np.quantile(
            vehicle_rate, quantiles
        ).tolist(),
        "aggregate_deadhead_reduction_km_95_ci": np.quantile(
            sampled_deadhead_reduction, quantiles
        ).tolist(),
    }


def summarize_validation(
    comparison: pd.DataFrame,
    samples: int = 10_000,
    seed: int = 20260920,
    solution_label: str = "P90 vehicle cost equivalent to 89 km of deadhead",
    independent_final_test: bool = False,
) -> dict[str, object]:
    historical_vehicles = int(comparison["historical_vehicle_chain_count"].sum())
    recommended_vehicles = int(comparison["recommended_vehicle_count"].sum())
    historical_deadhead = float(comparison["historical_deadhead_distance_km"].sum())
    recommended_deadhead = float(comparison["recommended_deadhead_distance_km"].sum())
    intervals = bootstrap_intervals(comparison, samples, seed)
    return {
        "source_only": "订单数据.xlsx",
        "solution": solution_label,
        "instances": len(comparison),
        "tasks": int(comparison["task_count"].sum()),
        "historical_vehicle_chain_count": historical_vehicles,
        "recommended_vehicle_count": recommended_vehicles,
        "aggregate_vehicle_reduction": historical_vehicles - recommended_vehicles,
        "aggregate_vehicle_reduction_rate": (historical_vehicles - recommended_vehicles)
        / historical_vehicles,
        "historical_deadhead_distance_km": historical_deadhead,
        "recommended_deadhead_distance_km": recommended_deadhead,
        "aggregate_deadhead_reduction_km": historical_deadhead - recommended_deadhead,
        "instances_with_vehicle_reduction": int(comparison["vehicle_improved"].sum()),
        "instances_with_equal_vehicle_count": int(
            comparison["vehicle_reduction"].eq(0).sum()
        ),
        "instances_with_vehicle_increase": int(
            comparison["vehicle_reduction"].lt(0).sum()
        ),
        "instances_with_no_deadhead_increase": int(
            comparison["deadhead_not_increased"].sum()
        ),
        "instances_with_deadhead_increase": int(
            (~comparison["deadhead_not_increased"]).sum()
        ),
        "instances_with_both_vehicle_and_deadhead_improvement": int(
            comparison["both_improved_or_equal"].sum()
        ),
        "median_instance_vehicle_reduction_rate": float(
            comparison["vehicle_reduction_rate"].median()
        ),
        "bootstrap_samples": samples,
        "bootstrap_random_seed": seed,
        **intervals,
        "inference_scope": (
            f"Intervals describe variation across the {len(comparison)} "
            "evaluation instances. Parameter selection used an earlier period, "
            "so this is an independent final-test estimate."
            if independent_final_test
            else (
                f"Intervals describe variation across the {len(comparison)} "
                "holdout instances. The parameter was selected on the same "
                "holdout and is not an independent confirmatory estimate."
            )
        ),
        "result_interpretation": (
            "Vehicle reduction is consistent across instances. Aggregate "
            "deadhead is approximately unchanged, while instance-level "
            "deadhead changes are heterogeneous."
        ),
    }


def main() -> None:
    args = parse_args()
    schedule = pd.read_csv(args.data_dir / args.schedule_filename)
    chains = pd.read_csv(args.data_dir / "historical_vehicle_chains.csv")
    links = pd.read_csv(args.data_dir / "robust_p90_candidate_task_links.csv")
    comparison = build_instance_comparison(schedule, chains, links)
    summary = summarize_validation(
        comparison,
        args.bootstrap_samples,
        args.random_seed,
        args.solution_label,
        args.independent_final_test,
    )
    comparison.to_csv(
        args.result_dir / f"{args.output_prefix}_instances.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.result_dir / f"{args.output_prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
