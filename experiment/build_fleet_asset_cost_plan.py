#!/usr/bin/env python3
"""Size fleet assets from daily route peaks and summarize supported costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiment.build_carbon_baseline import interval_values
from experiment.validate_parameter_registry import load_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算路线车型方案的最低资产需求与成本。")
    parser.add_argument(
        "--route-plan",
        type=Path,
        default=Path(
            "results/company_transport/final_test/operational_route_vehicle_plan.csv"
        ),
    )
    parser.add_argument(
        "--instances",
        type=Path,
        default=Path("processed/company/vrp_final_test/instances.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/company_transport/final_test"),
    )
    return parser.parse_args()


def candidate_asset_parameters(registry: dict[str, Any]) -> pd.DataFrame:
    electric_subsidy = float(
        registry["fixed_parameters"]["electric_purchase_subsidy"]["value"]
    )
    hydrogen_subsidy = float(
        registry["fixed_parameters"]["hydrogen_purchase_subsidy"]["value"]
    )
    rows = []
    for index, candidate in enumerate(registry["candidate_vehicles"], start=1):
        subsidy = (
            electric_subsidy
            if candidate["energy"] == "electric"
            else hydrogen_subsidy
        )
        rows.append(
            {
                "fleet_type_id": f"candidate_{index:02d}",
                "fleet_model": candidate["model"],
                "energy": candidate["energy"],
                "purchase_cost_cny_per_vehicle": float(
                    candidate["purchase_cost_cny"]
                ),
                "purchase_subsidy_cny_per_vehicle": subsidy,
                "maintenance_cost_cny_per_vehicle_year_min": float(
                    candidate["maintenance_cost_cny_per_year"]
                ),
                "maintenance_cost_cny_per_vehicle_year_max": float(
                    candidate["maintenance_cost_cny_per_year"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_asset_plan(
    route_plan: pd.DataFrame,
    instances: pd.DataFrame,
    registry: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    context = instances[["instance_id", "service_date", "component_id"]]
    routes = route_plan.merge(
        context, on="instance_id", how="left", validate="many_to_one"
    )
    if routes[["service_date", "component_id"]].isna().any().any():
        raise ValueError("路线存在无法关联的服务日期或网络分区")
    selected = routes["assignment_status"].eq("candidate_selected")
    routes["fleet_type_id"] = routes["assigned_candidate_id"].where(
        selected, "existing:" + routes["historical_vehicle_type"]
    )
    routes["fleet_model"] = routes["assigned_model"].where(
        selected, routes["historical_vehicle_type"]
    )
    routes["energy"] = routes["assigned_energy"].where(selected, "diesel")
    routes["is_new_candidate"] = selected
    daily = (
        routes.groupby(
            [
                "service_date",
                "component_id",
                "fleet_type_id",
                "fleet_model",
                "energy",
                "is_new_candidate",
            ],
            as_index=False,
        )
        .agg(route_chains=("vehicle_id", "size"))
    )
    assets = (
        daily.groupby(
            [
                "component_id",
                "fleet_type_id",
                "fleet_model",
                "energy",
                "is_new_candidate",
            ],
            as_index=False,
        )
        .agg(
            required_assets=("route_chains", "max"),
            observed_route_chains=("route_chains", "sum"),
            active_service_days=("service_date", "nunique"),
        )
    )
    candidate_parameters = candidate_asset_parameters(registry)
    assets = assets.merge(
        candidate_parameters,
        on=["fleet_type_id", "fleet_model", "energy"],
        how="left",
        validate="many_to_one",
    )
    maintenance_min, maintenance_max = interval_values(
        registry, "existing_diesel_annual_maintenance_cost"
    )
    existing = ~assets["is_new_candidate"]
    assets.loc[existing, "purchase_cost_cny_per_vehicle"] = 0.0
    assets.loc[existing, "purchase_subsidy_cny_per_vehicle"] = 0.0
    assets.loc[existing, "maintenance_cost_cny_per_vehicle_year_min"] = (
        maintenance_min
    )
    assets.loc[existing, "maintenance_cost_cny_per_vehicle_year_max"] = (
        maintenance_max
    )
    assets["gross_purchase_cost_cny"] = (
        assets["required_assets"] * assets["purchase_cost_cny_per_vehicle"]
    )
    assets["purchase_subsidy_cny"] = (
        assets["required_assets"] * assets["purchase_subsidy_cny_per_vehicle"]
    )
    assets["net_purchase_cost_cny"] = (
        assets["gross_purchase_cost_cny"] - assets["purchase_subsidy_cny"]
    )
    assets["annual_maintenance_cost_cny_min"] = (
        assets["required_assets"]
        * assets["maintenance_cost_cny_per_vehicle_year_min"]
    )
    assets["annual_maintenance_cost_cny_max"] = (
        assets["required_assets"]
        * assets["maintenance_cost_cny_per_vehicle_year_max"]
    )

    baseline_daily = (
        routes.groupby(
            ["service_date", "component_id", "historical_vehicle_type"],
            as_index=False,
        )
        .agg(route_chains=("vehicle_id", "size"))
    )
    baseline_assets = (
        baseline_daily.groupby(
            ["component_id", "historical_vehicle_type"], as_index=False
        )
        .agg(required_assets_floor=("route_chains", "max"))
    )
    return routes, daily, assets, baseline_assets


def summarize_asset_costs(
    routes: pd.DataFrame,
    assets: pd.DataFrame,
    baseline_assets: pd.DataFrame,
    registry: dict[str, Any],
) -> dict[str, object]:
    diesel_energy_min, diesel_energy_max = interval_values(
        registry, "existing_diesel_energy_cost"
    )
    selected = routes["is_new_candidate"]
    selected_energy_cost = float(
        (
            routes.loc[selected, "route_distance_km"]
            * routes.loc[selected, "assigned_energy_cost_cny_per_km"]
        ).sum()
    )
    retained_distance = float(routes.loc[~selected, "route_distance_km"].sum())
    observed_energy_min = selected_energy_cost + retained_distance * diesel_energy_min
    observed_energy_max = selected_energy_cost + retained_distance * diesel_energy_max
    active_days = int(routes["service_date"].nunique())
    annualization_factor = 365 / active_days
    new_assets = assets[assets["is_new_candidate"]]
    gross_purchase = float(new_assets["gross_purchase_cost_cny"].sum())
    subsidy = float(new_assets["purchase_subsidy_cny"].sum())
    net_purchase = float(new_assets["net_purchase_cost_cny"].sum())
    maintenance_min = float(assets["annual_maintenance_cost_cny_min"].sum())
    maintenance_max = float(assets["annual_maintenance_cost_cny_max"].sum())
    annual_energy_min = observed_energy_min * annualization_factor
    annual_energy_max = observed_energy_max * annualization_factor
    return {
        "asset_sizing_method": "sum_of_component_level_daily_peaks_by_fleet_type",
        "observed_active_service_days": active_days,
        "new_energy_assets": int(new_assets["required_assets"].sum()),
        "retained_existing_assets_floor": int(
            assets.loc[~assets["is_new_candidate"], "required_assets"].sum()
        ),
        "all_diesel_baseline_assets_floor": int(
            baseline_assets["required_assets_floor"].sum()
        ),
        "gross_new_vehicle_purchase_cost_cny": gross_purchase,
        "purchase_subsidy_cny": subsidy,
        "net_new_vehicle_purchase_cost_cny": net_purchase,
        "annual_maintenance_cost_cny": {
            "minimum": maintenance_min,
            "maximum": maintenance_max,
        },
        "observed_period_energy_cost_cny": {
            "minimum": observed_energy_min,
            "maximum": observed_energy_max,
        },
        "annualized_sensitivity": {
            "status": "sensitivity_only",
            "factor": annualization_factor,
            "assumption": "23个测试活跃日的日均路线需求可代表全年活跃日。",
            "energy_cost_cny": {
                "minimum": annual_energy_min,
                "maximum": annual_energy_max,
            },
            "first_year_cost_cny": {
                "minimum": net_purchase + maintenance_min + annual_energy_min,
                "maximum": net_purchase + maintenance_max + annual_energy_max,
            },
        },
        "asset_count_limit": (
            "新能源路线均可重排进单日14小时窗口，其资产数按每日峰值计算。"
            "保留柴油路线含跨日和时间异常任务，其资产数及全柴油对照均为下界。"
        ),
    }


def main() -> None:
    args = parse_args()
    registry = load_registry()
    routes, daily, assets, baseline_assets = build_asset_plan(
        pd.read_csv(args.route_plan), pd.read_csv(args.instances), registry
    )
    summary = summarize_asset_costs(routes, assets, baseline_assets, registry)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(
        args.output_dir / "fleet_daily_demand.csv", index=False, encoding="utf-8-sig"
    )
    assets.to_csv(
        args.output_dir / "fleet_asset_cost_plan.csv", index=False, encoding="utf-8-sig"
    )
    baseline_assets.to_csv(
        args.output_dir / "all_diesel_asset_floor.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "fleet_asset_cost_plan.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
