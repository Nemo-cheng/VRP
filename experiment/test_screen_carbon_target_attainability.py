import pandas as pd

from experiment.screen_carbon_target_attainability import (
    build_chain_boundary,
    build_chain_emissions,
    screen_target,
)


def test_multiday_task_is_mapped_back_to_its_chain() -> None:
    boundary = pd.DataFrame(
        {
            "instance_id": [pd.NA, "i2"],
            "vehicle_id": [pd.NA, "v2"],
            "task_id": ["t1", pd.NA],
            "analysis_layer": ["multiday_task", "deferred_time_anomaly"],
            "scenario": ["lower_bound", "not_applicable"],
            "replacement_outcome": [
                "conditional_hydrogen_with_overnight_replenishment",
                "evidence_pending",
            ],
            "evidence_status": ["conditional", "deferred"],
        }
    )
    overlong = pd.DataFrame(
        {"task_id": ["t1"], "instance_id": ["i1"], "vehicle_id": ["v1"]}
    )

    mapped = build_chain_boundary(boundary, overlong, "lower_bound")

    assert set(map(tuple, mapped[["instance_id", "vehicle_id"]].to_numpy())) == {
        ("i1", "v1"),
        ("i2", "v2"),
    }


def test_zero_emission_bound_only_removes_eligible_chain_emissions() -> None:
    emissions = pd.DataFrame(
        {
            "instance_id": ["i1", "i2", "i3"],
            "vehicle_id": ["v1", "v2", "v3"],
            "emissions_kgco2_min": [70.0, 20.0, 10.0],
            "emissions_kgco2_max": [70.0, 20.0, 10.0],
        }
    )
    boundary = pd.DataFrame(
        {
            "instance_id": ["i1", "i2", "i3"],
            "vehicle_id": ["v1", "v2", "v3"],
            "replacement_outcome": [
                "direct_electric",
                "conditional_hydrogen_with_overnight_replenishment",
                "retain_conventional_or_change_plan",
            ],
            "analysis_layer": ["single", "multiday", "single"],
            "evidence_status": ["resolved", "conditional", "resolved"],
        }
    )

    _, summary = screen_target(emissions, boundary, 0.7)

    strict = summary.iloc[0]
    conditional = summary.iloc[1]
    assert strict["zero_emission_reduction_rate_min"] == 0.7
    assert bool(strict["target_reachable_under_zero_emission_bound"])
    assert strict["additional_avoided_tco2_needed_max"] == 0.0
    assert conditional["zero_emission_reduction_rate_min"] == 0.9


def test_unused_vehicle_without_factor_does_not_block_chain_emissions() -> None:
    schedule = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "vehicle_id": ["v1"],
            "task_id": ["t1"],
            "vehicle_type_name": ["used diesel"],
            "distance_km": [100.0],
            "deadhead_to_next_distance_km": [0.0],
        }
    )
    vehicles = pd.DataFrame(
        {
            "vehicle_type_name": ["used diesel", "unused electric"],
            "fuel": ["柴油", "纯电动"],
            "payload_t": [10.0, 2.0],
        }
    )
    registry = {
        "fixed_parameters": {
            "diesel_lower_heating_value": {"value": 43.33, "status": "active"},
            "diesel_carbon_content": {"value": 0.0202, "status": "active"},
            "diesel_carbon_oxidation_rate": {"value": 0.98, "status": "active"},
            "diesel_density": {"value": 0.8, "status": "active"},
            "gasoline_lower_heating_value": {"value": 44.8, "status": "active"},
            "gasoline_carbon_content": {"value": 0.0189, "status": "active"},
            "gasoline_carbon_oxidation_rate": {"value": 0.98, "status": "active"},
            "gasoline_density": {"value": 0.73, "status": "active"},
            "freight_diesel_ge_8_lt_20t_consumption": {
                "value": 30.7,
                "status": "active",
            },
        },
        "interval_parameters": {
            "diesel_le_2t_fuel_consumption_proxy": {
                "minimum": 12.0,
                "maximum": 20.0,
                "status": "sensitivity_only",
            }
        },
    }

    result = build_chain_emissions(schedule, vehicles, registry)

    assert len(result) == 1
    assert result.iloc[0]["vehicle_type_name"] == "used diesel"
