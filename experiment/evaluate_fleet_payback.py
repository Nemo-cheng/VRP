#!/usr/bin/env python3
"""Evaluate simple fleet payback using only registered cost intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiment.build_carbon_baseline import interval_values
from experiment.validate_parameter_registry import load_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算新能源车队简单投资回收期。")
    base = Path("results/company_transport/final_test")
    parser.add_argument(
        "--asset-plan", type=Path, default=base / "fleet_asset_cost_plan.csv"
    )
    parser.add_argument(
        "--baseline-assets", type=Path, default=base / "all_diesel_asset_floor.csv"
    )
    parser.add_argument(
        "--route-cost", type=Path, default=base / "operational_route_vehicle_plan.json"
    )
    parser.add_argument(
        "--asset-summary", type=Path, default=base / "fleet_asset_cost_plan.json"
    )
    parser.add_argument("--output-dir", type=Path, default=base)
    return parser.parse_args()


def evaluate_payback(
    assets: pd.DataFrame,
    baseline_assets: pd.DataFrame,
    route_cost: dict[str, Any],
    asset_summary: dict[str, Any],
    registry: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, object]]:
    maintenance_min, maintenance_max = interval_values(
        registry, "existing_diesel_annual_maintenance_cost"
    )
    baseline_asset_count = int(baseline_assets["required_assets_floor"].sum())
    baseline_maintenance_min = baseline_asset_count * maintenance_min
    baseline_maintenance_max = baseline_asset_count * maintenance_max
    plan_maintenance_min = float(assets["annual_maintenance_cost_cny_min"].sum())
    plan_maintenance_max = float(assets["annual_maintenance_cost_cny_max"].sum())
    annualization_factor = float(
        asset_summary["annualized_sensitivity"]["factor"]
    )
    baseline_energy_min = (
        float(route_cost["baseline_energy_cost_cny"]["minimum"])
        * annualization_factor
    )
    baseline_energy_max = (
        float(route_cost["baseline_energy_cost_cny"]["maximum"])
        * annualization_factor
    )
    plan_energy_min = (
        float(route_cost["plan_energy_cost_cny"]["minimum"])
        * annualization_factor
    )
    plan_energy_max = (
        float(route_cost["plan_energy_cost_cny"]["maximum"])
        * annualization_factor
    )
    baseline_operating_min = baseline_maintenance_min + baseline_energy_min
    baseline_operating_max = baseline_maintenance_max + baseline_energy_max
    plan_operating_min = plan_maintenance_min + plan_energy_min
    plan_operating_max = plan_maintenance_max + plan_energy_max
    annual_saving_min = baseline_operating_min - plan_operating_min
    annual_saving_max = baseline_operating_max - plan_operating_max
    net_investment = float(asset_summary["net_new_vehicle_purchase_cost_cny"])
    gross_investment = float(asset_summary["gross_new_vehicle_purchase_cost_cny"])
    if annual_saving_min <= 0:
        net_payback_worst = None
        gross_payback_worst = None
    else:
        net_payback_worst = net_investment / annual_saving_min
        gross_payback_worst = gross_investment / annual_saving_min
    net_payback_best = net_investment / annual_saving_max
    gross_payback_best = gross_investment / annual_saving_max
    comparison = pd.DataFrame(
        [
            {
                "plan": "all_diesel_asset_floor",
                "annual_maintenance_cost_cny_min": baseline_maintenance_min,
                "annual_maintenance_cost_cny_max": baseline_maintenance_max,
                "annualized_energy_cost_cny_min": baseline_energy_min,
                "annualized_energy_cost_cny_max": baseline_energy_max,
                "annual_operating_cost_cny_min": baseline_operating_min,
                "annual_operating_cost_cny_max": baseline_operating_max,
            },
            {
                "plan": "route_vehicle_replacement_plan",
                "annual_maintenance_cost_cny_min": plan_maintenance_min,
                "annual_maintenance_cost_cny_max": plan_maintenance_max,
                "annualized_energy_cost_cny_min": plan_energy_min,
                "annualized_energy_cost_cny_max": plan_energy_max,
                "annual_operating_cost_cny_min": plan_operating_min,
                "annual_operating_cost_cny_max": plan_operating_max,
            },
        ]
    )
    summary = {
        "status": "sensitivity_only",
        "annualization_factor": annualization_factor,
        "asset_counts_are_lower_bounds": True,
        "baseline_assets_floor": baseline_asset_count,
        "plan_assets_floor": int(assets["required_assets"].sum()),
        "net_new_vehicle_investment_cny": net_investment,
        "gross_new_vehicle_investment_cny": gross_investment,
        "annual_operating_cost_saving_cny": {
            "minimum": annual_saving_min,
            "maximum": annual_saving_max,
        },
        "simple_payback_years_after_subsidy": {
            "best_case": net_payback_best,
            "worst_case": net_payback_worst,
        },
        "simple_payback_years_before_subsidy": {
            "best_case": gross_payback_best,
            "worst_case": gross_payback_worst,
        },
        "excluded_parameters": [
            "discount_rate",
            "vehicle_service_life",
            "residual_value",
            "battery_or_fuel_cell_replacement",
            "charging_or_hydrogen_station_investment",
            "carbon_price",
        ],
        "interpretation": (
            "简单回收期只用于成本区间敏感性。缺少车辆寿命、折现率、残值和"
            "补能基础设施投资时，不能据此形成净现值或最终采购结论。"
        ),
    }
    return comparison, summary


def main() -> None:
    args = parse_args()
    route_cost = json.loads(args.route_cost.read_text(encoding="utf-8"))
    asset_summary = json.loads(args.asset_summary.read_text(encoding="utf-8"))
    comparison, summary = evaluate_payback(
        pd.read_csv(args.asset_plan),
        pd.read_csv(args.baseline_assets),
        route_cost,
        asset_summary,
        load_registry(),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        args.output_dir / "fleet_payback_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "fleet_payback_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
