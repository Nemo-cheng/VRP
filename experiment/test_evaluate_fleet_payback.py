import pandas as pd
import pytest

from experiment.evaluate_fleet_payback import evaluate_payback


def test_payback_uses_correlated_cost_interval_endpoints() -> None:
    assets = pd.DataFrame(
        {
            "required_assets": [1, 1],
            "annual_maintenance_cost_cny_min": [8.0, 15.0],
            "annual_maintenance_cost_cny_max": [8.0, 40.0],
        }
    )
    baseline_assets = pd.DataFrame({"required_assets_floor": [2]})
    route_cost = {
        "baseline_energy_cost_cny": {"minimum": 100.0, "maximum": 160.0},
        "plan_energy_cost_cny": {"minimum": 70.0, "maximum": 90.0},
    }
    asset_summary = {
        "net_new_vehicle_purchase_cost_cny": 1000.0,
        "gross_new_vehicle_purchase_cost_cny": 1200.0,
        "annualized_sensitivity": {"factor": 2.0},
    }
    registry = {
        "interval_parameters": {
            "existing_diesel_annual_maintenance_cost": {
                "minimum": 15.0,
                "maximum": 40.0,
                "status": "sensitivity_only",
            }
        }
    }

    comparison, summary = evaluate_payback(
        assets, baseline_assets, route_cost, asset_summary, registry
    )

    assert comparison.loc[0, "annual_operating_cost_cny_min"] == 230.0
    assert comparison.loc[1, "annual_operating_cost_cny_min"] == 163.0
    assert summary["annual_operating_cost_saving_cny"] == {
        "minimum": 67.0,
        "maximum": 172.0,
    }
    assert summary["simple_payback_years_after_subsidy"][
        "best_case"
    ] == pytest.approx(1000 / 172)
    assert summary["simple_payback_years_after_subsidy"][
        "worst_case"
    ] == pytest.approx(1000 / 67)
