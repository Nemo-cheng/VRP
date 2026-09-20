import pandas as pd

from experiment.build_fleet_asset_cost_plan import (
    build_asset_plan,
    summarize_asset_costs,
)


def _registry() -> dict:
    return {
        "fixed_parameters": {
            "electric_purchase_subsidy": {"value": 20.0},
            "hydrogen_purchase_subsidy": {"value": 200.0},
        },
        "interval_parameters": {
            "existing_diesel_annual_maintenance_cost": {
                "minimum": 15.0,
                "maximum": 40.0,
                "status": "sensitivity_only",
            },
            "existing_diesel_energy_cost": {
                "minimum": 0.75,
                "maximum": 1.2,
                "status": "sensitivity_only",
            },
        },
        "candidate_vehicles": [
            {
                "model": "electric",
                "energy": "electric",
                "purchase_cost_cny": 120.0,
                "maintenance_cost_cny_per_year": 8.0,
            }
        ],
    }


def test_assets_use_component_level_daily_peak_not_route_sum() -> None:
    routes = pd.DataFrame(
        {
            "instance_id": ["i1", "i1", "i2", "i3"],
            "vehicle_id": ["v1", "v2", "v3", "v4"],
            "assignment_status": ["candidate_selected"] * 4,
            "assigned_candidate_id": ["candidate_01"] * 4,
            "assigned_model": ["electric"] * 4,
            "assigned_energy": ["electric"] * 4,
            "historical_vehicle_type": ["diesel"] * 4,
        }
    )
    instances = pd.DataFrame(
        {
            "instance_id": ["i1", "i2", "i3"],
            "service_date": ["2023-12-01", "2023-12-02", "2023-12-01"],
            "component_id": ["c1", "c1", "c2"],
        }
    )

    _, _, assets, _ = build_asset_plan(routes, instances, _registry())

    assert assets["required_assets"].sum() == 3
    assert assets["net_purchase_cost_cny"].sum() == 300.0


def test_summary_keeps_annualization_as_sensitivity() -> None:
    routes = pd.DataFrame(
        {
            "service_date": ["2023-12-01", "2023-12-02"],
            "is_new_candidate": [True, False],
            "route_distance_km": [100.0, 100.0],
            "assigned_energy_cost_cny_per_km": [0.25, pd.NA],
        }
    )
    assets = pd.DataFrame(
        {
            "is_new_candidate": [True, False],
            "required_assets": [1, 1],
            "gross_purchase_cost_cny": [120.0, 0.0],
            "purchase_subsidy_cny": [20.0, 0.0],
            "net_purchase_cost_cny": [100.0, 0.0],
            "annual_maintenance_cost_cny_min": [8.0, 15.0],
            "annual_maintenance_cost_cny_max": [8.0, 40.0],
        }
    )
    baseline = pd.DataFrame({"required_assets_floor": [2]})

    summary = summarize_asset_costs(routes, assets, baseline, _registry())

    assert summary["observed_period_energy_cost_cny"] == {
        "minimum": 100.0,
        "maximum": 145.0,
    }
    assert summary["annualized_sensitivity"]["status"] == "sensitivity_only"
