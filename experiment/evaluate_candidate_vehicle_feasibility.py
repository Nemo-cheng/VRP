#!/usr/bin/env python3
"""Evaluate candidate vehicles on retained P90 chains without invented energy data."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openpyxl>=3.1",
#   "pandas>=2.2",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiment.audit_route_operating_window import CHAIN_KEY, audit_schedule
from experiment.prepare_excel_transport_data import build_transport_events, load_data
from experiment.validate_parameter_registry import load_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估候选新能源车的路线链可行性。")
    parser.add_argument("--orders", type=Path, default=Path("订单数据.xlsx"))
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


def fixed_value(registry: dict[str, Any], name: str) -> float:
    item = registry["fixed_parameters"][name]
    if item.get("status") != "active":
        raise ValueError(f"固定参数未启用: {name}")
    return float(item["value"])


def candidate_limits(
    candidate: dict[str, Any],
    scenario: str,
    registry: dict[str, Any],
) -> tuple[float, float]:
    if scenario not in {"lower_bound", "upper_bound"}:
        raise ValueError(f"未知情景: {scenario}")
    index = 0 if scenario == "lower_bound" else 1
    payload_t = float(candidate["payload_t"][index])
    nominal_range = float(candidate["nominal_range_km"][index])
    if candidate["energy"] == "electric":
        winter_factor = fixed_value(registry, "electric_winter_range_factor")
        degradation = fixed_value(
            registry, "annual_battery_range_degradation"
        )
        years = int(fixed_value(registry, "battery_degradation_years_to_2026"))
        effective_range = (
            nominal_range * winter_factor * (1 - degradation) ** years
        )
    elif candidate["energy"] == "hydrogen":
        winter_factor = fixed_value(registry, "hydrogen_winter_range_factor")
        effective_range = nominal_range * winter_factor
    else:
        raise ValueError(f"未知能源类型: {candidate['energy']}")
    return payload_t, effective_range


def build_chain_requirements(
    schedule: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    event_payload = events[["event_id", "total_weight_kg"]].rename(
        columns={"event_id": "task_id"}
    )
    frame = schedule.merge(
        event_payload,
        on="task_id",
        how="left",
        validate="many_to_one",
    )
    if frame["total_weight_kg"].isna().any():
        raise ValueError("存在无法关联载重的路线任务")
    audit, audit_summary = audit_schedule(frame)
    requirements = (
        frame.groupby(CHAIN_KEY, as_index=False)
        .agg(
            tasks=("task_id", "size"),
            maximum_task_payload_kg=("total_weight_kg", "max"),
            loaded_distance_km=("distance_km", "sum"),
            deadhead_distance_km=("deadhead_to_next_distance_km", "sum"),
        )
        .merge(
            audit[
                [
                    *CHAIN_KEY,
                    "chain_elapsed_hours",
                    "operating_window_compliant",
                ]
            ],
            on=CHAIN_KEY,
            validate="one_to_one",
        )
    )
    requirements["required_distance_km"] = (
        requirements["loaded_distance_km"]
        + requirements["deadhead_distance_km"]
    )
    return requirements, audit_summary


def evaluate_candidates(
    requirements: pd.DataFrame,
    registry: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(
        registry["candidate_vehicles"], start=1
    ):
        candidate_id = f"candidate_{candidate_index:02d}"
        for scenario in ("lower_bound", "upper_bound"):
            payload_t, effective_range = candidate_limits(
                candidate, scenario, registry
            )
            for chain in requirements.itertuples(index=False):
                payload_feasible = chain.maximum_task_payload_kg <= payload_t * 1000
                range_feasible = chain.required_distance_km <= effective_range
                window_feasible = bool(chain.operating_window_compliant)
                rows.append(
                    {
                        "instance_id": chain.instance_id,
                        "vehicle_id": chain.vehicle_id,
                        "candidate_id": candidate_id,
                        "candidate_model": candidate["model"],
                        "energy": candidate["energy"],
                        "scenario": scenario,
                        "payload_limit_t": payload_t,
                        "effective_winter_2026_range_km": effective_range,
                        "maximum_task_payload_kg": chain.maximum_task_payload_kg,
                        "required_distance_km": chain.required_distance_km,
                        "chain_elapsed_hours": chain.chain_elapsed_hours,
                        "operating_window_feasible": window_feasible,
                        "payload_feasible": payload_feasible,
                        "range_feasible_without_recharging": range_feasible,
                        "technical_feasible": (
                            window_feasible
                            and payload_feasible
                            and range_feasible
                        ),
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(
            ["candidate_id", "candidate_model", "energy", "scenario"],
            as_index=False,
        )
        .agg(
            evaluated_chains=("vehicle_id", "size"),
            operating_window_feasible_chains=(
                "operating_window_feasible",
                "sum",
            ),
            payload_feasible_chains=("payload_feasible", "sum"),
            range_feasible_chains=("range_feasible_without_recharging", "sum"),
            technical_feasible_chains=("technical_feasible", "sum"),
        )
    )
    summary["technical_feasible_rate_all_chains"] = (
        summary["technical_feasible_chains"] / summary["evaluated_chains"]
    )
    summary["technical_feasible_rate_window_compliant"] = (
        summary["technical_feasible_chains"]
        / summary["operating_window_feasible_chains"]
    )
    return detail, summary


def main() -> None:
    args = parse_args()
    registry = load_registry()
    orders, waybills = load_data(args.orders)
    events, _ = build_transport_events(orders, waybills)
    schedule = pd.read_csv(args.schedule)
    requirements, audit_summary = build_chain_requirements(schedule, events)
    detail, summary = evaluate_candidates(requirements, registry)
    result = {
        "source_only": ["订单数据.xlsx", "车辆数据.xlsx", "competition_brief"],
        "route_solution": "independently_selected_p90_solution",
        "chains": len(requirements),
        "window_compliant_chains": int(
            requirements["operating_window_compliant"].sum()
        ),
        "scenarios": {
            "lower_bound": "候选车型载重和标称续航区间下界",
            "upper_bound": "候选车型载重和标称续航区间上界",
        },
        "winter_and_degradation_applied": True,
        "en_route_recharging_assumed": False,
        "operating_window_audit": audit_summary,
        "decision_scope": (
            "结果只验证载重、无途中补能续航和运营窗口，不包含碳排放与成本排序。"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        args.output_dir / "candidate_vehicle_feasibility_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_dir / "candidate_vehicle_feasibility_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "candidate_vehicle_feasibility.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
