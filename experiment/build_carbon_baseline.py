#!/usr/bin/env python3
"""Build the evidence-covered portion of the 2023 transport carbon baseline."""

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

from experiment.prepare_company_data import (
    build_physical_legs,
    load_source,
    prepare_orders,
    prepare_waybills,
)
from experiment.validate_parameter_registry import load_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算有官方参数覆盖的2023年运输碳排放基线。"
    )
    parser.add_argument("--orders", type=Path, default=Path("订单数据.xlsx"))
    parser.add_argument("--vehicles", type=Path, default=Path("车辆数据.xlsx"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/company_transport/carbon_baseline"),
    )
    return parser.parse_args()


def parameter_value(registry: dict[str, Any], name: str) -> float:
    item = registry["fixed_parameters"][name]
    if item.get("status") != "active":
        raise ValueError(f"参数未启用: {name}")
    return float(item["value"])


def fuel_emission_factor_kg_per_kg(
    lower_heating_value_gj_per_tonne: float,
    carbon_content_tonne_c_per_gj: float,
    oxidation_rate: float,
) -> float:
    return (
        lower_heating_value_gj_per_tonne
        * carbon_content_tonne_c_per_gj
        * oxidation_rate
        * 44
        / 12
    )


def classify_consumption_parameter(
    fuel: object, payload_t: object
) -> tuple[str | None, str]:
    if pd.isna(fuel) or pd.isna(payload_t):
        return None, "missing_vehicle_parameter"
    fuel_name = str(fuel).strip()
    payload = float(payload_t)
    if fuel_name == "汽油" and payload <= 2:
        return "freight_gasoline_le_2t_consumption", "covered"
    if fuel_name == "柴油" and 2 < payload <= 4:
        return "freight_diesel_gt_2_le_4t_consumption", "covered"
    if fuel_name == "柴油" and 4 < payload < 8:
        return "freight_diesel_gt_4_lt_8t_consumption", "covered"
    if fuel_name == "柴油" and 8 <= payload < 20:
        return "freight_diesel_ge_8_lt_20t_consumption", "covered"
    if fuel_name == "柴油" and payload >= 20:
        return "freight_diesel_ge_20t_consumption", "covered"
    if fuel_name == "柴油" and payload <= 2:
        return None, "diesel_le_2t_no_official_default"
    if fuel_name == "纯电动":
        return None, "electricity_consumption_missing"
    return None, "fuel_payload_class_not_covered"


def load_vehicle_parameters(path: Path) -> pd.DataFrame:
    vehicles = pd.read_excel(path, sheet_name="Sheet2")
    required = {"车型种类", "燃油类型", "车辆载重"}
    missing = required - set(vehicles.columns)
    if missing:
        raise ValueError(f"车辆数据缺少字段: {', '.join(sorted(missing))}")
    if vehicles["车型种类"].duplicated().any():
        raise ValueError("车辆数据中的车型种类必须唯一")
    return vehicles[list(required)].rename(
        columns={
            "车型种类": "vehicle_type_name",
            "燃油类型": "fuel",
            "车辆载重": "payload_t",
        }
    )


def build_baseline(
    orders_path: Path, vehicles_path: Path, registry: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    orders_raw, waybills_raw = load_source(orders_path)
    legs, _ = build_physical_legs(
        prepare_orders(orders_raw), prepare_waybills(waybills_raw)
    )
    eligible = legs[legs["analysis_eligible"]].copy()
    vehicle_parameters = load_vehicle_parameters(vehicles_path)
    eligible = eligible.merge(
        vehicle_parameters,
        on="vehicle_type_name",
        how="left",
        validate="many_to_one",
    )

    classifications = eligible.apply(
        lambda row: classify_consumption_parameter(row["fuel"], row["payload_t"]),
        axis=1,
        result_type="expand",
    )
    eligible[["consumption_parameter", "coverage_status"]] = classifications

    fuel_factors = {
        "汽油": fuel_emission_factor_kg_per_kg(
            parameter_value(registry, "gasoline_lower_heating_value"),
            parameter_value(registry, "gasoline_carbon_content"),
            parameter_value(registry, "gasoline_carbon_oxidation_rate"),
        ),
        "柴油": fuel_emission_factor_kg_per_kg(
            parameter_value(registry, "diesel_lower_heating_value"),
            parameter_value(registry, "diesel_carbon_content"),
            parameter_value(registry, "diesel_carbon_oxidation_rate"),
        ),
    }
    densities = {
        "汽油": parameter_value(registry, "gasoline_density"),
        "柴油": parameter_value(registry, "diesel_density"),
    }
    consumption_values = {
        name: parameter_value(registry, name)
        for name in eligible["consumption_parameter"].dropna().unique()
    }
    eligible["fuel_consumption_l_per_100km"] = eligible[
        "consumption_parameter"
    ].map(consumption_values)
    eligible["fuel_emission_factor_kgco2_per_kg"] = eligible["fuel"].map(
        fuel_factors
    )
    eligible["vehicle_emission_factor_kgco2_per_km"] = (
        eligible["fuel_consumption_l_per_100km"]
        * eligible["fuel"].map(densities)
        / 100
        * eligible["fuel_emission_factor_kgco2_per_kg"]
    )
    eligible["emissions_kgco2"] = (
        eligible["distance_km"] * eligible["vehicle_emission_factor_kgco2_per_km"]
    )

    by_group = (
        eligible.groupby(
            ["coverage_status", "fuel", "payload_t", "consumption_parameter"],
            dropna=False,
            as_index=False,
        )
        .agg(
            physical_legs=("vehicle_id", "size"),
            vehicles=("vehicle_id", "nunique"),
            distance_km=("distance_km", "sum"),
            emissions_kgco2=(
                "emissions_kgco2",
                lambda values: values.sum(min_count=1),
            ),
        )
        .sort_values("distance_km", ascending=False)
    )
    by_group["distance_share"] = (
        by_group["distance_km"] / eligible["distance_km"].sum()
    )

    covered = eligible[eligible["coverage_status"] == "covered"]
    total_distance = float(eligible["distance_km"].sum())
    covered_distance = float(covered["distance_km"].sum())
    service_dates = eligible["departed_at"].dt.normalize()
    active_service_days = int(service_dates.nunique())
    annualization_factor = 365 / active_service_days
    observed_covered_emissions = float(covered["emissions_kgco2"].sum() / 1000)
    uncovered = (
        eligible[eligible["coverage_status"] != "covered"]
        .groupby("coverage_status", as_index=False)
        .agg(
            physical_legs=("vehicle_id", "size"),
            distance_km=("distance_km", "sum"),
        )
    )
    summary = {
        "observed_date_min": str(service_dates.min().date()),
        "observed_date_max": str(service_dates.max().date()),
        "observed_months": sorted(
            eligible["departed_at"].dt.to_period("M").astype(str).unique()
        ),
        "active_service_days": active_service_days,
        "annualization_applied": False,
        "method_source": "hubei_transport_carbon_guide_2024",
        "eligible_physical_legs": len(eligible),
        "eligible_distance_km": total_distance,
        "covered_physical_legs": len(covered),
        "covered_distance_km": covered_distance,
        "covered_distance_share": covered_distance / total_distance,
        "observed_covered_emissions_tco2": observed_covered_emissions,
        "annualization_sensitivity": {
            "status": "sensitivity_only",
            "method": "observed covered emissions * 365 / active service days",
            "factor": annualization_factor,
            "covered_emissions_tco2": (
                observed_covered_emissions * annualization_factor
            ),
            "assumption": "观测活跃日的运输强度可代表全年，赛题未直接保证该假设。",
        },
        "is_complete_2023_baseline": len(covered) == len(eligible),
        "uncovered": uncovered.to_dict(orient="records"),
        "interpretation": (
            "observed_covered_emissions_tco2仅表示观测活跃日内由官方缺省参数"
            "覆盖的运输段。年化值只用于敏感性分析，两者均不能作为完整2023年总排放。"
        ),
    }
    return eligible, by_group, summary


def main() -> None:
    args = parse_args()
    for path in (args.orders, args.vehicles):
        if not path.exists():
            raise FileNotFoundError(f"找不到输入文件: {path}")
    _, by_group, summary = build_baseline(
        args.orders, args.vehicles, load_registry()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_group.to_csv(
        args.output_dir / "carbon_baseline_by_parameter_group.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "carbon_baseline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
