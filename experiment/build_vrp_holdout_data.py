#!/usr/bin/env python3
"""Build a chronological train-test split for out-of-time VRP validation."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "networkx>=3.4",
#   "openpyxl>=3.1",
#   "pandas>=2.2",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from build_vrp_data_layer import (
    assign_components,
    build_high_confidence_graph,
    build_instances,
    build_lane_vehicle_types,
    build_network_tables,
    build_readiness_report,
    build_tasks,
)
from prepare_excel_transport_data import build_transport_events, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造 VRP 时间外验证数据。")
    parser.add_argument("--input", type=Path, default=Path("订单数据.xlsx"))
    parser.add_argument("--cutoff", default="2023-10-01")
    parser.add_argument("--test-end-exclusive", default=None)
    parser.add_argument("--min-edge-observations", type=int, default=10)
    parser.add_argument("--min-period-observations", type=int, default=5)
    parser.add_argument("--min-instance-tasks", type=int, default=10)
    parser.add_argument("--required-instances", type=int, default=20)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("processed/company/vrp_holdout")
    )
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/company_transport/holdout")
    )
    return parser.parse_args()


def build_holdout_layer(
    events: pd.DataFrame,
    cutoff: pd.Timestamp,
    min_edge_observations: int,
    min_period_observations: int,
    min_instance_tasks: int,
    required_instances: int,
    test_end_exclusive: pd.Timestamp | None = None,
) -> dict[str, object]:
    all_tasks = build_tasks(events)
    departure_time = pd.to_datetime(all_tasks["departed_at"])
    train_tasks = all_tasks[departure_time < cutoff].copy()
    test_mask = departure_time >= cutoff
    if test_end_exclusive is not None:
        test_mask &= departure_time < test_end_exclusive
    test_tasks = all_tasks[test_mask].copy()
    edges, periods = build_network_tables(
        train_tasks, min_edge_observations, min_period_observations
    )
    lane_vehicle_types = build_lane_vehicle_types(train_tasks)
    graph = build_high_confidence_graph(edges)
    test_tasks = assign_components(test_tasks, graph)
    instances, links = build_instances(test_tasks, graph, min_instance_tasks)
    report = build_readiness_report(
        test_tasks,
        edges,
        periods,
        instances,
        links,
        min_edge_observations,
        min_period_observations,
        min_instance_tasks,
        required_instances,
    )
    report["validation_protocol"] = {
        "split": "chronological",
        "cutoff": cutoff.strftime("%Y-%m-%d"),
        "train_period_end_exclusive": cutoff.strftime("%Y-%m-%d"),
        "test_period_start_inclusive": cutoff.strftime("%Y-%m-%d"),
        "test_period_end_exclusive": (
            test_end_exclusive.strftime("%Y-%m-%d")
            if test_end_exclusive is not None
            else None
        ),
        "train_tasks": len(train_tasks),
        "test_tasks": len(test_tasks),
        "network_and_vehicle_type_evidence_from_train_only": True,
    }
    return {
        "train_tasks": train_tasks,
        "test_tasks": test_tasks,
        "edges": edges,
        "periods": periods,
        "lane_vehicle_types": lane_vehicle_types,
        "instances": instances,
        "links": links,
        "report": report,
    }


def main() -> None:
    args = parse_args()
    orders, waybills = load_data(args.input)
    events, _ = build_transport_events(orders, waybills)
    layer = build_holdout_layer(
        events,
        pd.Timestamp(args.cutoff),
        args.min_edge_observations,
        args.min_period_observations,
        args.min_instance_tasks,
        args.required_instances,
        pd.Timestamp(args.test_end_exclusive) if args.test_end_exclusive else None,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "tasks.csv": layer["test_tasks"],
        "network_edges.csv": layer["edges"],
        "edge_period_stats.csv": layer["periods"],
        "lane_vehicle_types.csv": layer["lane_vehicle_types"],
        "instances.csv": layer["instances"],
        "candidate_task_links.csv": layer["links"],
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")
    public_instances = layer["instances"].drop(
        columns=["service_date"], errors="ignore"
    )
    public_instances.to_csv(
        args.result_dir / "vrp_instance_summary.csv", index=False, encoding="utf-8-sig"
    )
    (args.result_dir / "vrp_data_readiness.json").write_text(
        json.dumps(layer["report"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(layer["report"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
