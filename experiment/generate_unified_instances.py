#!/usr/bin/env python3
"""Attach paired company payload samples to eligible LaDe delivery tasks."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "pandas>=2.2",
#   "pyarrow>=15",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [1103, 2207, 3301, 4409, 5501, 6607, 7703, 8807, 9901, 10103]
PAYLOAD_COLUMNS = ["weight_kg", "volume_cm3", "pieces"]
CORRELATION_TOLERANCE = 0.02
MEAN_RELATIVE_TOLERANCE = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 LaDe 与企业载荷联合实例。")
    parser.add_argument(
        "--payload-pool",
        type=Path,
        default=Path("processed/company/payload_pool.csv"),
    )
    parser.add_argument(
        "--routes",
        type=Path,
        default=Path("processed/lade/shanghai_route_summary.csv"),
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path("processed/lade/shanghai_historical_routes.parquet"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("processed/unified_instances")
    )
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/unified_instances")
    )
    return parser.parse_args()


def eligible_nodes(routes: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    eligible_route_ids = routes.loc[
        (routes["stop_count"] >= 10)
        & (routes["complete_coordinate_share"] == 1),
        "route_id",
    ]
    selected = nodes[nodes["route_id"].isin(eligible_route_ids)].copy()
    return selected.sort_values(["route_id", "stop_index"], kind="stable").reset_index(
        drop=True
    )


def valid_shanghai_pool(payload_pool: pd.DataFrame) -> pd.DataFrame:
    pool = payload_pool.loc[
        payload_pool["region"].eq("shanghai"), PAYLOAD_COLUMNS
    ].copy()
    for column in PAYLOAD_COLUMNS:
        pool[column] = pd.to_numeric(pool[column], errors="coerce")
    pool = pool.dropna(subset=PAYLOAD_COLUMNS)
    pool = pool[
        (pool["weight_kg"] > 0)
        & (pool["volume_cm3"] > 0)
        & (pool["pieces"] > 0)
    ].reset_index(drop=True)
    if len(pool) < 1000:
        raise ValueError("上海载荷池有效样本少于 1000，不能直接用于主实验")
    return pool


def sample_payloads(
    nodes: pd.DataFrame, pool: pd.DataFrame, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(pool), size=len(nodes))
    sampled = pool.iloc[sample_indices].reset_index(drop=True)
    output = nodes.copy()
    output[PAYLOAD_COLUMNS] = sampled[PAYLOAD_COLUMNS]
    output["payload_source_pool"] = "shanghai"
    output["payload_seed"] = seed
    return output


def distribution_record(
    values: pd.DataFrame, population: str, seed: int | None
) -> dict[str, object]:
    return {
        "population": population,
        "seed": seed,
        "orders": int(len(values)),
        "weight_mean_kg": float(values["weight_kg"].mean()),
        "weight_p50_kg": float(values["weight_kg"].quantile(0.5)),
        "weight_p90_kg": float(values["weight_kg"].quantile(0.9)),
        "weight_p99_kg": float(values["weight_kg"].quantile(0.99)),
        "volume_mean_cm3": float(values["volume_cm3"].mean()),
        "volume_p50_cm3": float(values["volume_cm3"].quantile(0.5)),
        "volume_p90_cm3": float(values["volume_cm3"].quantile(0.9)),
        "volume_p99_cm3": float(values["volume_cm3"].quantile(0.99)),
        "pieces_mean": float(values["pieces"].mean()),
        "pieces_p90": float(values["pieces"].quantile(0.9)),
        "weight_volume_corr": float(
            values[["weight_kg", "volume_cm3"]].corr().iloc[0, 1]
        ),
    }


def route_load_record(sampled: pd.DataFrame, seed: int) -> dict[str, object]:
    route_loads = sampled.groupby("route_id", sort=False).agg(
        total_weight_kg=("weight_kg", "sum"),
        total_volume_cm3=("volume_cm3", "sum"),
        max_order_weight_kg=("weight_kg", "max"),
        max_order_volume_cm3=("volume_cm3", "max"),
    )
    route_loads["largest_weight_share"] = (
        route_loads["max_order_weight_kg"] / route_loads["total_weight_kg"]
    )
    route_loads["largest_volume_share"] = (
        route_loads["max_order_volume_cm3"] / route_loads["total_volume_cm3"]
    )
    return {
        "seed": seed,
        "routes": int(len(route_loads)),
        "route_weight_p50_kg": float(route_loads["total_weight_kg"].quantile(0.5)),
        "route_weight_p90_kg": float(route_loads["total_weight_kg"].quantile(0.9)),
        "route_weight_p99_kg": float(route_loads["total_weight_kg"].quantile(0.99)),
        "route_weight_max_kg": float(route_loads["total_weight_kg"].max()),
        "route_volume_p50_m3": float(
            route_loads["total_volume_cm3"].quantile(0.5) / 1_000_000
        ),
        "route_volume_p90_m3": float(
            route_loads["total_volume_cm3"].quantile(0.9) / 1_000_000
        ),
        "route_volume_p99_m3": float(
            route_loads["total_volume_cm3"].quantile(0.99) / 1_000_000
        ),
        "route_volume_max_m3": float(
            route_loads["total_volume_cm3"].max() / 1_000_000
        ),
        "routes_one_order_over_half_weight": int(
            (route_loads["largest_weight_share"] > 0.5).sum()
        ),
        "routes_one_order_over_half_volume": int(
            (route_loads["largest_volume_share"] > 0.5).sum()
        ),
    }


def main() -> None:
    args = parse_args()
    payload_pool = valid_shanghai_pool(pd.read_csv(args.payload_pool))
    routes = pd.read_csv(args.routes)
    nodes = eligible_nodes(routes, pd.read_parquet(args.nodes))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    distribution_rows = [distribution_record(payload_pool, "source_pool", None)]
    route_load_rows = []
    for seed in SEEDS:
        sampled = sample_payloads(nodes, payload_pool, seed)
        sampled.to_parquet(args.output_dir / f"shanghai_seed_{seed}.parquet", index=False)
        distribution_rows.append(
            distribution_record(sampled, "generated_instance", seed)
        )
        route_load_rows.append(route_load_record(sampled, seed))

    distributions = pd.DataFrame(distribution_rows)
    route_loads = pd.DataFrame(route_load_rows)
    distributions.to_csv(args.result_dir / "payload_distribution_check.csv", index=False)
    route_loads.to_csv(args.result_dir / "route_load_summary.csv", index=False)
    source_corr = distributions.loc[
        distributions["population"].eq("source_pool"), "weight_volume_corr"
    ].iloc[0]
    generated = distributions[distributions["population"].eq("generated_instance")]
    source = distributions[distributions["population"].eq("source_pool")].iloc[0]
    maximum_correlation_error = float(
        (generated["weight_volume_corr"] - source_corr).abs().max()
    )
    maximum_weight_mean_relative_error = float(
        ((generated["weight_mean_kg"] / source["weight_mean_kg"]) - 1).abs().max()
    )
    maximum_volume_mean_relative_error = float(
        ((generated["volume_mean_cm3"] / source["volume_mean_cm3"]) - 1)
        .abs()
        .max()
    )
    report = {
        "payload_pool": "shanghai",
        "source_pool_orders": int(len(payload_pool)),
        "eligible_routes": int(nodes["route_id"].nunique()),
        "tasks_per_seed": int(len(nodes)),
        "seeds": SEEDS,
        "versions": len(SEEDS),
        "paired_sampling": True,
        "source_weight_volume_correlation": float(source_corr),
        "validation_thresholds": {
            "maximum_absolute_correlation_error": CORRELATION_TOLERANCE,
            "maximum_relative_mean_error": MEAN_RELATIVE_TOLERANCE,
        },
        "maximum_absolute_correlation_error": maximum_correlation_error,
        "maximum_weight_mean_relative_error": maximum_weight_mean_relative_error,
        "maximum_volume_mean_relative_error": maximum_volume_mean_relative_error,
        "distribution_validation_passed": bool(
            maximum_correlation_error <= CORRELATION_TOLERANCE
            and maximum_weight_mean_relative_error <= MEAN_RELATIVE_TOLERANCE
            and maximum_volume_mean_relative_error <= MEAN_RELATIVE_TOLERANCE
        ),
        "selection_note": "Shanghai pool is used because it has more than 1,000 complete paired payload observations. No Yangtze Delta fallback is required.",
        "capacity_note": "Route load distributions are descriptive until candidate vehicle capacities are sourced in stage E.",
    }
    (args.result_dir / "generation_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
