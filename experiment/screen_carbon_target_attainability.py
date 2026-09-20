#!/usr/bin/env python3
"""Screen whether the 70% carbon target is reachable under zero-emission bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiment.build_carbon_baseline import (
    classify_consumption_parameter,
    fuel_emission_factor_kg_per_kg,
    interval_values,
    load_vehicle_parameters,
    parameter_value,
)
from experiment.validate_parameter_registry import load_registry

CHAIN_KEY = ["instance_id", "vehicle_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛查70%运输减碳目标的理论可达性。")
    base = Path("results/company_transport/final_test")
    parser.add_argument(
        "--schedule",
        type=Path,
        default=Path(
            "processed/company/vrp_final_test/"
            "independently_selected_p90_solution_schedules.csv"
        ),
    )
    parser.add_argument("--vehicles", type=Path, default=Path("车辆数据.xlsx"))
    parser.add_argument(
        "--boundary-detail",
        type=Path,
        default=base / "vehicle_replacement_boundary_detail.csv",
    )
    parser.add_argument(
        "--overlong-classification",
        type=Path,
        default=base / "overlong_task_classification.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=base)
    return parser.parse_args()


def vehicle_emission_factor_bounds(
    vehicles: pd.DataFrame,
    registry: dict[str, Any],
) -> pd.DataFrame:
    diesel_factor = fuel_emission_factor_kg_per_kg(
        parameter_value(registry, "diesel_lower_heating_value"),
        parameter_value(registry, "diesel_carbon_content"),
        parameter_value(registry, "diesel_carbon_oxidation_rate"),
    ) * parameter_value(registry, "diesel_density")
    gasoline_factor = fuel_emission_factor_kg_per_kg(
        parameter_value(registry, "gasoline_lower_heating_value"),
        parameter_value(registry, "gasoline_carbon_content"),
        parameter_value(registry, "gasoline_carbon_oxidation_rate"),
    ) * parameter_value(registry, "gasoline_density")
    light_diesel_min, light_diesel_max = interval_values(
        registry, "diesel_le_2t_fuel_consumption_proxy"
    )
    rows: list[dict[str, object]] = []
    for vehicle in vehicles.itertuples(index=False):
        parameter, status = classify_consumption_parameter(
            vehicle.fuel, vehicle.payload_t
        )
        if status == "covered":
            consumption_min = consumption_max = parameter_value(
                registry, str(parameter)
            )
        elif status == "diesel_le_2t_no_official_default":
            consumption_min, consumption_max = light_diesel_min, light_diesel_max
        else:
            raise ValueError(
                f"最终测试车型缺少可用排放因子: {vehicle.vehicle_type_name}"
            )
        fuel_factor = diesel_factor if vehicle.fuel == "柴油" else gasoline_factor
        rows.append(
            {
                "vehicle_type_name": vehicle.vehicle_type_name,
                "factor_status": status,
                "emission_factor_kgco2_per_km_min": (
                    consumption_min / 100 * fuel_factor
                ),
                "emission_factor_kgco2_per_km_max": (
                    consumption_max / 100 * fuel_factor
                ),
            }
        )
    return pd.DataFrame(rows)


def build_chain_emissions(
    schedule: pd.DataFrame,
    vehicles: pd.DataFrame,
    registry: dict[str, Any],
) -> pd.DataFrame:
    used_vehicle_types = set(schedule["vehicle_type_name"].dropna())
    selected_vehicles = vehicles[
        vehicles["vehicle_type_name"].isin(used_vehicle_types)
    ].copy()
    missing_types = used_vehicle_types - set(selected_vehicles["vehicle_type_name"])
    if missing_types:
        raise ValueError(f"最终测试车型缺少车辆参数: {sorted(missing_types)}")
    factors = vehicle_emission_factor_bounds(selected_vehicles, registry)
    frame = schedule.merge(
        factors, on="vehicle_type_name", how="left", validate="many_to_one"
    )
    factor_columns = [
        "emission_factor_kgco2_per_km_min",
        "emission_factor_kgco2_per_km_max",
    ]
    if frame[factor_columns].isna().any().any():
        raise ValueError("最终测试路线存在无法关联的车辆排放因子")
    type_counts = frame.groupby(CHAIN_KEY)["vehicle_type_name"].nunique()
    if type_counts.gt(1).any():
        raise ValueError("同一车辆链包含多个车型，无法确定空驶排放因子")
    frame["route_distance_km"] = (
        frame["distance_km"] + frame["deadhead_to_next_distance_km"]
    )
    for bound in ("min", "max"):
        frame[f"emissions_kgco2_{bound}"] = (
            frame["route_distance_km"]
            * frame[f"emission_factor_kgco2_per_km_{bound}"]
        )
    return frame.groupby(CHAIN_KEY, as_index=False).agg(
        tasks=("task_id", "size"),
        vehicle_type_name=("vehicle_type_name", "first"),
        factor_status=("factor_status", "first"),
        route_distance_km=("route_distance_km", "sum"),
        emissions_kgco2_min=("emissions_kgco2_min", "sum"),
        emissions_kgco2_max=("emissions_kgco2_max", "sum"),
    )


def build_chain_boundary(
    boundary: pd.DataFrame,
    overlong: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    selected = boundary[
        boundary["scenario"].isin([scenario, "not_applicable"])
    ].copy()
    multiday = selected["analysis_layer"].eq("multiday_task")
    task_to_chain = overlong[["task_id", *CHAIN_KEY]]
    selected_multiday = selected[multiday].drop(columns=CHAIN_KEY).merge(
        task_to_chain, on="task_id", how="left", validate="one_to_one"
    )
    selected = pd.concat([selected[~multiday], selected_multiday], ignore_index=True)
    if selected[CHAIN_KEY].isna().any().any():
        raise ValueError("替换边界存在无法映射到路线链的任务")
    if selected.duplicated(CHAIN_KEY).any():
        raise ValueError("同一场景中的路线链被重复分类")
    return selected[
        [*CHAIN_KEY, "analysis_layer", "replacement_outcome", "evidence_status"]
    ]


def screen_target(
    chain_emissions: pd.DataFrame,
    chain_boundary: pd.DataFrame,
    target: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = chain_emissions.merge(
        chain_boundary, on=CHAIN_KEY, how="left", validate="one_to_one"
    )
    if detail["replacement_outcome"].isna().any():
        raise ValueError("存在未进入替换边界的路线链")
    tiers = {
        "resolved_single_day_zero_emission_bound": {
            "direct_electric",
            "hydrogen_only",
        },
        "plus_conditional_multiday_zero_emission_bound": {
            "direct_electric",
            "hydrogen_only",
            "conditional_electric_with_overnight_replenishment",
            "conditional_hydrogen_with_overnight_replenishment",
        },
    }
    total_min = float(detail["emissions_kgco2_min"].sum())
    total_max = float(detail["emissions_kgco2_max"].sum())
    rows: list[dict[str, object]] = []
    for tier, outcomes in tiers.items():
        replaceable = detail["replacement_outcome"].isin(outcomes)
        avoided_min = float(detail.loc[replaceable, "emissions_kgco2_min"].sum())
        avoided_max = float(detail.loc[replaceable, "emissions_kgco2_max"].sum())
        reduction_at_min_factor = avoided_min / total_min
        reduction_at_max_factor = avoided_max / total_max
        lower = min(reduction_at_min_factor, reduction_at_max_factor)
        upper = max(reduction_at_min_factor, reduction_at_max_factor)
        shortfall_at_min_factor = max(0.0, target * total_min - avoided_min)
        shortfall_at_max_factor = max(0.0, target * total_max - avoided_max)
        rows.append(
            {
                "screening_tier": tier,
                "replaceable_chains": int(replaceable.sum()),
                "baseline_emissions_tco2_min": total_min / 1000,
                "baseline_emissions_tco2_max": total_max / 1000,
                "avoided_emissions_tco2_min": avoided_min / 1000,
                "avoided_emissions_tco2_max": avoided_max / 1000,
                "zero_emission_reduction_rate_min": lower,
                "zero_emission_reduction_rate_max": upper,
                "additional_avoided_tco2_needed_min": min(
                    shortfall_at_min_factor, shortfall_at_max_factor
                )
                / 1000,
                "additional_avoided_tco2_needed_max": max(
                    shortfall_at_min_factor, shortfall_at_max_factor
                )
                / 1000,
                "target": target,
                "target_reachable_under_zero_emission_bound": upper >= target,
            }
        )
    return detail, pd.DataFrame(rows)


def summarize_emissions_by_outcome(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(
            ["scenario", "analysis_layer", "replacement_outcome"],
            as_index=False,
        )
        .agg(
            chains=("vehicle_id", "size"),
            route_distance_km=("route_distance_km", "sum"),
            emissions_tco2_min=(
                "emissions_kgco2_min",
                lambda values: values.sum() / 1000,
            ),
            emissions_tco2_max=(
                "emissions_kgco2_max",
                lambda values: values.sum() / 1000,
            ),
        )
        .sort_values(["scenario", "analysis_layer", "replacement_outcome"])
    )


def main() -> None:
    args = parse_args()
    registry = load_registry()
    target = parameter_value(registry, "carbon_reduction_target")
    schedule = pd.read_csv(args.schedule)
    vehicles = load_vehicle_parameters(args.vehicles)
    chain_emissions = build_chain_emissions(schedule, vehicles, registry)
    boundary = pd.read_csv(args.boundary_detail)
    overlong = pd.read_csv(args.overlong_classification)
    detail_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for scenario in ("lower_bound", "upper_bound"):
        chain_boundary = build_chain_boundary(boundary, overlong, scenario)
        detail, summary = screen_target(chain_emissions, chain_boundary, target)
        detail.insert(0, "scenario", scenario)
        summary.insert(0, "scenario", scenario)
        detail_frames.append(detail)
        summary_frames.append(summary)
    detail = pd.concat(detail_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    outcome_summary = summarize_emissions_by_outcome(detail)
    result = {
        "scope": "final_test_subnetwork_only",
        "route_solution": "independently_selected_p90_solution",
        "chains": len(chain_emissions),
        "target": target,
        "target_source": "competition_brief",
        "new_energy_emissions_assumed_kgco2_per_km": 0.0,
        "assumption_role": "theoretical_upper_bound_only",
        "official_factor_covered_chains": int(
            chain_emissions["factor_status"].eq("covered").sum()
        ),
        "sensitivity_factor_chains": int(
            chain_emissions["factor_status"].ne("covered").sum()
        ),
        "results": summary.to_dict(orient="records"),
        "interpretation": (
            "若零排放理论上界仍低于70%，则当前替换边界必然无法达标。"
            "若上界达到70%，只表示目标可能可达，仍需候选车型电耗、氢耗、"
            "制氢排放和补能设施证据才能形成最终方案。"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(
        args.output_dir / "carbon_target_attainability_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_dir / "carbon_target_attainability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    outcome_summary.to_csv(
        args.output_dir / "carbon_target_emissions_by_outcome.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "carbon_target_attainability.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
