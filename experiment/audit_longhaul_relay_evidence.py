#!/usr/bin/env python3
"""Audit observed relay-path evidence for unresolved long-haul tasks."""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from experiment.evaluate_candidate_vehicle_feasibility import candidate_limits
from experiment.validate_parameter_registry import load_registry

CHAIN_KEY = ["instance_id", "vehicle_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="核验长途任务的历史中继路径证据。")
    base = Path("results/company_transport/final_test")
    parser.add_argument(
        "--boundary-detail",
        type=Path,
        default=base / "vehicle_replacement_boundary_detail.csv",
    )
    parser.add_argument(
        "--classification",
        type=Path,
        default=base / "overlong_task_classification.csv",
    )
    parser.add_argument(
        "--network-edges",
        type=Path,
        default=Path("processed/company/vrp_final_test/network_edges.csv"),
    )
    parser.add_argument(
        "--carbon-detail",
        type=Path,
        default=base / "carbon_target_attainability_detail.csv",
    )
    parser.add_argument(
        "--attainability-summary",
        type=Path,
        default=base / "carbon_target_attainability_summary.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=base)
    return parser.parse_args()


def maximum_candidate_range(
    registry: dict[str, Any], scenario: str = "upper_bound"
) -> float:
    return max(
        candidate_limits(candidate, scenario, registry)[1]
        for candidate in registry["candidate_vehicles"]
    )


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.origin_site_id,
            row.destination_site_id,
            distance_km=float(row.distance_km_p50),
            duration_hours=float(row.duration_hours_p90),
            observations=int(row.observations),
            high_confidence=bool(row.high_confidence),
        )
    return graph


def shortest_relay_path(
    graph: nx.DiGraph, origin: str, destination: str
) -> dict[str, object] | None:
    candidate = graph.copy()
    if candidate.has_edge(origin, destination):
        candidate.remove_edge(origin, destination)
    try:
        path = nx.shortest_path(
            candidate, origin, destination, weight="distance_km"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    path_edges = list(pairwise(path))
    distances = [candidate.edges[edge]["distance_km"] for edge in path_edges]
    durations = [candidate.edges[edge]["duration_hours"] for edge in path_edges]
    observations = [candidate.edges[edge]["observations"] for edge in path_edges]
    return {
        "path": ">".join(path),
        "relay_count": len(path) - 2,
        "segment_count": len(path_edges),
        "distance_km": float(sum(distances)),
        "duration_hours_without_transfer": float(sum(durations)),
        "maximum_segment_distance_km": float(max(distances)),
        "maximum_segment_duration_hours": float(max(durations)),
        "minimum_segment_observations": int(min(observations)),
    }


def audit_relay_paths(
    targets: pd.DataFrame,
    edges: pd.DataFrame,
    maximum_range_km: float,
    maximum_segment_hours: float = 14.0,
) -> pd.DataFrame:
    high_confidence = edges[edges["high_confidence"]].copy()
    graph_high_topology = build_graph(high_confidence)
    graph_all_topology = build_graph(edges)
    graph_high_feasible = build_graph(
        high_confidence[
            high_confidence["distance_km_p50"].le(maximum_range_km)
            & high_confidence["duration_hours_p90"].le(maximum_segment_hours)
        ]
    )
    graph_all_feasible = build_graph(
        edges[
            edges["distance_km_p50"].le(maximum_range_km)
            & edges["duration_hours_p90"].le(maximum_segment_hours)
        ]
    )
    rows: list[dict[str, object]] = []
    for task in targets.itertuples(index=False):
        high_topology = shortest_relay_path(
            graph_high_topology, task.origin_site_id, task.destination_site_id
        )
        all_topology = shortest_relay_path(
            graph_all_topology, task.origin_site_id, task.destination_site_id
        )
        high_feasible = shortest_relay_path(
            graph_high_feasible, task.origin_site_id, task.destination_site_id
        )
        all_feasible = shortest_relay_path(
            graph_all_feasible, task.origin_site_id, task.destination_site_id
        )
        if high_feasible is not None:
            status = "verified_relay_feasible"
        elif all_feasible is not None:
            status = "low_confidence_relay_feasible"
        elif all_topology is not None:
            status = "topology_only_not_segment_feasible"
        else:
            status = "no_alternative_observed_topology"
        row = {
            "task_id": task.task_id,
            "instance_id": task.instance_id,
            "vehicle_id": task.vehicle_id,
            "origin_site_id": task.origin_site_id,
            "destination_site_id": task.destination_site_id,
            "direct_distance_km": task.distance_km,
            "maximum_candidate_range_km": maximum_range_km,
            "maximum_segment_hours": maximum_segment_hours,
            "high_confidence_topology_exists": high_topology is not None,
            "all_observed_topology_exists": all_topology is not None,
            "verified_relay_feasible": high_feasible is not None,
            "sensitivity_relay_feasible": all_feasible is not None,
            "relay_evidence_status": status,
        }
        for prefix, path in (
            ("high_confidence_topology", high_topology),
            ("all_observed_topology", all_topology),
            ("verified_relay", high_feasible),
            ("sensitivity_relay", all_feasible),
        ):
            row[f"{prefix}_path"] = path["path"] if path else pd.NA
            row[f"{prefix}_distance_km"] = (
                path["distance_km"] if path else pd.NA
            )
            row[f"{prefix}_maximum_segment_distance_km"] = (
                path["maximum_segment_distance_km"] if path else pd.NA
            )
            row[f"{prefix}_minimum_segment_observations"] = (
                path["minimum_segment_observations"] if path else pd.NA
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    registry = load_registry()
    maximum_range_km = maximum_candidate_range(registry)
    boundary = pd.read_csv(args.boundary_detail)
    target_ids = set(
        boundary.loc[
            boundary["analysis_layer"].eq("multiday_task")
            & boundary["scenario"].eq("lower_bound")
            & boundary["replacement_outcome"].eq(
                "additional_days_or_retain_conventional"
            ),
            "task_id",
        ]
    )
    classification = pd.read_csv(args.classification)
    targets = classification[classification["task_id"].isin(target_ids)].copy()
    detail = audit_relay_paths(
        targets,
        pd.read_csv(args.network_edges),
        maximum_range_km,
    )

    carbon = pd.read_csv(args.carbon_detail)
    carbon = carbon[carbon["scenario"].eq("lower_bound")]
    carbon = carbon[[*CHAIN_KEY, "emissions_kgco2_min", "emissions_kgco2_max"]]
    detail = detail.merge(carbon, on=CHAIN_KEY, how="left", validate="one_to_one")
    verified = detail["verified_relay_feasible"]
    sensitivity = detail["sensitivity_relay_feasible"]
    attainability = pd.read_csv(args.attainability_summary)
    prior = attainability[
        attainability["scenario"].eq("lower_bound")
        & attainability["screening_tier"].eq(
            "plus_conditional_multiday_zero_emission_bound"
        )
    ].iloc[0]
    verified_avoided_min = float(
        detail.loc[verified, "emissions_kgco2_min"].sum() / 1000
    )
    verified_avoided_max = float(
        detail.loc[verified, "emissions_kgco2_max"].sum() / 1000
    )
    revised_rate_min = (
        prior["avoided_emissions_tco2_min"] + verified_avoided_min
    ) / prior["baseline_emissions_tco2_min"]
    revised_rate_max = (
        prior["avoided_emissions_tco2_max"] + verified_avoided_max
    ) / prior["baseline_emissions_tco2_max"]
    summary = {
        "source_only": "订单数据.xlsx",
        "training_network_only": True,
        "target_longhaul_tasks": len(detail),
        "maximum_candidate_effective_range_km": maximum_range_km,
        "maximum_segment_hours": 14.0,
        "maximum_segment_hours_source": "competition_brief",
        "high_confidence_alternative_topology_tasks": int(
            detail["high_confidence_topology_exists"].sum()
        ),
        "all_observed_alternative_topology_tasks_sensitivity": int(
            detail["all_observed_topology_exists"].sum()
        ),
        "verified_relay_feasible_tasks": int(verified.sum()),
        "all_observed_relay_feasible_tasks_sensitivity": int(sensitivity.sum()),
        "target_task_emissions_tco2": {
            "minimum": float(detail["emissions_kgco2_min"].sum() / 1000),
            "maximum": float(detail["emissions_kgco2_max"].sum() / 1000),
        },
        "verified_relay_unlocked_emissions_tco2": {
            "minimum": verified_avoided_min,
            "maximum": verified_avoided_max,
        },
        "zero_emission_reduction_bound_after_verified_relay": {
            "minimum": min(revised_rate_min, revised_rate_max),
            "maximum": max(revised_rate_min, revised_rate_max),
        },
        "transfer_time_available": False,
        "relay_handling_capacity_available": False,
        "decision": (
            "只有训练期高置信网络中每段同时满足续航和14小时约束的路径"
            "才能进入主方案。低频历史线路仅用于敏感性诊断。"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        args.output_dir / "longhaul_relay_evidence_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "longhaul_relay_evidence.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
