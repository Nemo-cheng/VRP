#!/usr/bin/env python3
"""Combine single-day and multiday candidate-vehicle replacement boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CHAIN_KEY = ["instance_id", "vehicle_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总单日与跨日车辆替换边界。")
    base = Path("results/company_transport/final_test")
    parser.add_argument(
        "--single-day-detail",
        type=Path,
        default=base / "candidate_vehicle_feasibility_detail.csv",
    )
    parser.add_argument(
        "--multiday-detail",
        type=Path,
        default=base / "multiday_vehicle_feasibility_detail.csv",
    )
    parser.add_argument(
        "--overlong-classification",
        type=Path,
        default=base / "overlong_task_classification.csv",
    )
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path(
            "processed/company/vrp_final_test/"
            "independently_selected_p90_solution_schedules.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=base)
    return parser.parse_args()


def build_single_day_boundary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = detail.groupby(CHAIN_KEY + ["scenario"], sort=True)
    for (*chain_values, scenario), group in grouped:
        if not bool(group["operating_window_feasible"].iloc[0]):
            continue
        electric = group[group["energy"].eq("electric")]
        hydrogen = group[group["energy"].eq("hydrogen")]
        electric_feasible = bool(electric["technical_feasible"].any())
        hydrogen_feasible = bool(hydrogen["technical_feasible"].any())
        if electric_feasible:
            outcome = "direct_electric"
        elif hydrogen_feasible:
            outcome = "hydrogen_only"
        else:
            outcome = "retain_conventional_or_change_plan"
        rows.append(
            {
                "analysis_layer": "single_day_chain",
                "unit_id": ":".join(map(str, chain_values)),
                "instance_id": chain_values[0],
                "vehicle_id": chain_values[1],
                "task_id": pd.NA,
                "scenario": scenario,
                "replacement_outcome": outcome,
                "electric_feasible": electric_feasible,
                "hydrogen_feasible": hydrogen_feasible,
                "requires_verified_overnight_replenishment": False,
                "additional_operating_days_required": False,
                "distance_km": float(group["required_distance_km"].iloc[0]),
                "evidence_status": "resolved",
            }
        )
    return pd.DataFrame(rows)


def build_multiday_boundary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = detail.groupby(["task_id", "scenario"], sort=True)
    for (task_id, scenario), group in grouped:
        electric = group[group["energy"].eq("electric")]
        hydrogen = group[group["energy"].eq("hydrogen")]
        electric_feasible = bool(
            (
                electric["payload_feasible"]
                & electric["range_feasible_at_time_minimum_days"]
            ).any()
        )
        hydrogen_feasible = bool(
            (
                hydrogen["payload_feasible"]
                & hydrogen["range_feasible_at_time_minimum_days"]
            ).any()
        )
        if electric_feasible:
            outcome = "conditional_electric_with_overnight_replenishment"
        elif hydrogen_feasible:
            outcome = "conditional_hydrogen_with_overnight_replenishment"
        else:
            outcome = "additional_days_or_retain_conventional"
        rows.append(
            {
                "analysis_layer": "multiday_task",
                "unit_id": task_id,
                "instance_id": pd.NA,
                "vehicle_id": pd.NA,
                "task_id": task_id,
                "scenario": scenario,
                "replacement_outcome": outcome,
                "electric_feasible": electric_feasible,
                "hydrogen_feasible": hydrogen_feasible,
                "requires_verified_overnight_replenishment": (
                    electric_feasible or hydrogen_feasible
                ),
                "additional_operating_days_required": not (
                    electric_feasible or hydrogen_feasible
                ),
                "distance_km": float(group["task_distance_km"].iloc[0]),
                "evidence_status": "conditional_unverified_replenishment",
            }
        )
    return pd.DataFrame(rows)


def build_unresolved_boundary(
    schedule: pd.DataFrame,
    single_day_detail: pd.DataFrame,
    overlong: pd.DataFrame,
) -> pd.DataFrame:
    deferred = overlong[~overlong["included_in_multiday_analysis"].astype(bool)]
    rows = [
        {
            "analysis_layer": "deferred_time_anomaly",
            "unit_id": row.task_id,
            "instance_id": row.instance_id,
            "vehicle_id": row.vehicle_id,
            "task_id": row.task_id,
            "scenario": "not_applicable",
            "replacement_outcome": "evidence_pending",
            "electric_feasible": pd.NA,
            "hydrogen_feasible": pd.NA,
            "requires_verified_overnight_replenishment": pd.NA,
            "additional_operating_days_required": pd.NA,
            "distance_km": float(row.distance_km),
            "evidence_status": "deferred_time_anomaly",
        }
        for row in deferred.itertuples(index=False)
    ]

    chain_window = (
        single_day_detail.groupby(CHAIN_KEY, as_index=False)
        .agg(operating_window_feasible=("operating_window_feasible", "first"))
    )
    noncompliant = chain_window[~chain_window["operating_window_feasible"]]
    overlong_chains = set(
        map(tuple, overlong[["instance_id", "vehicle_id"]].itertuples(index=False, name=None))
    )
    unresolved_chains = noncompliant[
        ~noncompliant[CHAIN_KEY].apply(tuple, axis=1).isin(overlong_chains)
    ]
    chain_distance = schedule.groupby(CHAIN_KEY, as_index=False).agg(
        distance_km=("distance_km", "sum")
    )
    unresolved_chains = unresolved_chains.merge(
        chain_distance, on=CHAIN_KEY, how="left", validate="one_to_one"
    )
    for row in unresolved_chains.itertuples(index=False):
        rows.append(
            {
                "analysis_layer": "aggregate_shift_overrun",
                "unit_id": f"{row.instance_id}:{row.vehicle_id}",
                "instance_id": row.instance_id,
                "vehicle_id": row.vehicle_id,
                "task_id": pd.NA,
                "scenario": "not_applicable",
                "replacement_outcome": "operating_plan_pending",
                "electric_feasible": pd.NA,
                "hydrogen_feasible": pd.NA,
                "requires_verified_overnight_replenishment": pd.NA,
                "additional_operating_days_required": pd.NA,
                "distance_km": float(row.distance_km),
                "evidence_status": "aggregate_shift_exceeds_14h",
            }
        )
    return pd.DataFrame(rows)


def summarize_boundary(detail: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail.groupby(
            [
                "analysis_layer",
                "scenario",
                "replacement_outcome",
                "evidence_status",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(units=("unit_id", "size"), distance_km=("distance_km", "sum"))
        .sort_values(["analysis_layer", "scenario", "replacement_outcome"])
    )
    summary["unit_share_within_layer_scenario"] = summary["units"] / summary.groupby(
        ["analysis_layer", "scenario"]
    )["units"].transform("sum")
    return summary


def main() -> None:
    args = parse_args()
    single_day_source = pd.read_csv(args.single_day_detail)
    multiday_source = pd.read_csv(args.multiday_detail)
    overlong = pd.read_csv(args.overlong_classification)
    schedule = pd.read_csv(args.schedule)

    resolved_single_day = build_single_day_boundary(single_day_source)
    conditional_multiday = build_multiday_boundary(multiday_source)
    unresolved = build_unresolved_boundary(schedule, single_day_source, overlong)
    detail = pd.concat(
        [resolved_single_day, conditional_multiday, unresolved], ignore_index=True
    )
    summary = summarize_boundary(detail)
    primary = summary[summary["scenario"].isin(["lower_bound", "not_applicable"])]
    result = {
        "primary_scenario": "lower_bound",
        "opportunity_scenario": "upper_bound",
        "single_day_unit": "vehicle_chain",
        "multiday_unit": "transport_task",
        "units_are_not_additive_across_layers": True,
        "direct_replacement_requires_en_route_replenishment": False,
        "multiday_replacement_is_conditional_on_overnight_replenishment": True,
        "primary_boundary": primary.to_dict(orient="records"),
        "decision_scope": (
            "单日合规链可形成直接替换边界。跨日任务只形成有条件替换边界。"
            "时间异常和班次总时长超限单独保留，不归因于车型。"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        args.output_dir / "vehicle_replacement_boundary_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_dir / "vehicle_replacement_boundary_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "vehicle_replacement_boundary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
