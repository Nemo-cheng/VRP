#!/usr/bin/env python3
"""Select deterministic representative and stress-test routes for optimization."""

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

SCALES = [20, 30, 50, 100]
REFERENCE_SEED = 1103


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="选择正式优化使用的代表路线实例。")
    parser.add_argument(
        "--routes",
        type=Path,
        default=Path("processed/lade/shanghai_route_summary.csv"),
    )
    parser.add_argument(
        "--reference-instance",
        type=Path,
        default=Path(
            f"processed/unified_instances/shanghai_seed_{REFERENCE_SEED}.parquet"
        ),
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("processed/unified_instances")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("processed/benchmark_instances")
    )
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/benchmark_instances")
    )
    return parser.parse_args()


def route_loads(reference: pd.DataFrame) -> pd.DataFrame:
    return (
        reference.groupby("route_id", as_index=False)
        .agg(
            total_weight_kg=("weight_kg", "sum"),
            total_volume_m3=("volume_cm3", lambda values: values.sum() / 1_000_000),
        )
    )


def robust_distance_from_median(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    scores = pd.Series(0.0, index=frame.index)
    for column in columns:
        median = frame[column].median()
        scale = (frame[column] - median).abs().median()
        if scale == 0:
            continue
        scores += ((frame[column] - median) / scale) ** 2
    return np.sqrt(scores)


def select_routes(routes: pd.DataFrame, loads: pd.DataFrame) -> pd.DataFrame:
    candidates = routes.merge(loads, on="route_id", how="inner", validate="one_to_one")
    selected_rows: list[dict[str, object]] = []
    for scale in SCALES:
        group = candidates[
            (candidates["stop_count"] == scale)
            & (candidates["complete_coordinate_share"] == 1)
        ].copy()
        if group.empty:
            raise ValueError(f"没有 {scale} 客户的完整路线")
        group["representative_score"] = robust_distance_from_median(
            group,
            [
                "straight_line_distance_km",
                "work_span_minutes",
                "max_service_radius_km",
                "total_weight_kg",
                "total_volume_m3",
            ],
        )
        representative = group.sort_values(
            ["representative_score", "route_id"]
        ).iloc[0]
        choices = [("representative", representative)]
        used_route_ids = {representative["route_id"]}
        if scale < 100:
            distance_threshold = group["straight_line_distance_km"].quantile(0.95)
            long_group = group[
                (group["straight_line_distance_km"] >= distance_threshold)
                & ~group["route_id"].isin(used_route_ids)
            ]
            distance_stress = long_group.sort_values(
                ["straight_line_distance_km", "route_id"],
                ascending=[False, True],
            ).iloc[0]
            choices.append(("distance_stress", distance_stress))
            used_route_ids.add(distance_stress["route_id"])
            payload_stress = group[
                ~group["route_id"].isin(used_route_ids)
            ].sort_values(
                ["total_weight_kg", "route_id"], ascending=[False, True]
            ).iloc[0]
            choices.append(
                ("payload_stress", payload_stress)
            )
        for case_type, row in choices:
            selected_rows.append(
                {
                    "instance_id": f"n{scale}_{case_type}",
                    "case_type": case_type,
                    "reference_seed": REFERENCE_SEED,
                    **row.drop(labels=["representative_score"]).to_dict(),
                }
            )
    return pd.DataFrame(selected_rows)


def export_instances(
    selection: pd.DataFrame, input_dir: Path, output_dir: Path
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    route_to_instance = selection.set_index("route_id")["instance_id"]
    files = []
    for path in sorted(input_dir.glob("shanghai_seed_*.parquet")):
        seed = int(path.stem.rsplit("_", 1)[1])
        frame = pd.read_parquet(path)
        selected = frame[frame["route_id"].isin(route_to_instance.index)].copy()
        selected["instance_id"] = selected["route_id"].map(route_to_instance)
        target = output_dir / f"benchmark_seed_{seed}.parquet"
        selected.to_parquet(target, index=False)
        files.append(
            {
                "seed": seed,
                "file": target.name,
                "instances": int(selected["instance_id"].nunique()),
                "tasks": int(len(selected)),
            }
        )
    return {
        "reference_seed": REFERENCE_SEED,
        "selection_count": int(len(selection)),
        "selection_types": selection.groupby("stop_count")["case_type"]
        .apply(list)
        .to_dict(),
        "generated_files": files,
        "selection_note": "The 100-customer class has one eligible route and is treated as a boundary stress case, not a representative route.",
    }


def main() -> None:
    args = parse_args()
    routes = pd.read_csv(args.routes)
    reference = pd.read_parquet(args.reference_instance)
    selection = select_routes(routes, route_loads(reference))
    report = export_instances(selection, args.input_dir, args.output_dir)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    public_columns = [
        "instance_id",
        "case_type",
        "reference_seed",
        "route_id",
        "stop_count",
        "work_span_minutes",
        "straight_line_distance_km",
        "max_adjacent_distance_km",
        "max_service_radius_km",
        "total_weight_kg",
        "total_volume_m3",
    ]
    selection[public_columns].to_csv(
        args.result_dir / "benchmark_selection.csv", index=False
    )
    (args.result_dir / "benchmark_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
