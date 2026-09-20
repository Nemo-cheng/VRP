import pandas as pd

from experiment.evaluate_multiday_vehicle_feasibility import (
    build_multiday_requirements,
    evaluate_multiday_candidates,
)


def test_build_requirements_excludes_deferred_time_anomalies() -> None:
    classification = pd.DataFrame(
        {
            "task_id": ["long", "waiting"],
            "included_in_multiday_analysis": [True, False],
            "planning_duration_hours": [20.0, pd.NA],
            "required_operating_days": [2, pd.NA],
            "maximum_daily_distance_km": [700.0, pd.NA],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["long", "waiting"],
            "total_weight_kg": [1000.0, 1000.0],
        }
    )

    requirements = build_multiday_requirements(classification, events)

    assert requirements["task_id"].tolist() == ["long"]
    assert requirements.iloc[0]["total_weight_kg"] == 1000.0


def test_range_can_increase_days_but_replenishment_remains_unverified() -> None:
    requirements = pd.DataFrame(
        {
            "task_id": ["long"],
            "origin_site_id": ["a"],
            "destination_site_id": ["b"],
            "distance_km": [1000.0],
            "planning_duration_hours": [20.0],
            "required_operating_days": [2],
            "maximum_daily_distance_km": [700.0],
            "total_weight_kg": [1000.0],
        }
    )
    registry = {
        "fixed_parameters": {
            "electric_winter_range_factor": {"value": 1.0, "status": "active"},
            "annual_battery_range_degradation": {
                "value": 0.0,
                "status": "active",
            },
            "battery_degradation_years_to_2026": {
                "value": 0,
                "status": "active",
            },
        },
        "candidate_vehicles": [
            {
                "model": "test electric",
                "energy": "electric",
                "nominal_range_km": [400.0, 500.0],
                "payload_t": [1.5, 2.0],
            }
        ],
    }

    detail, summary = evaluate_multiday_candidates(requirements, registry)

    lower = detail[detail["scenario"].eq("lower_bound")].iloc[0]
    assert lower["time_minimum_operating_days"] == 2
    assert lower["range_minimum_operating_days"] == 3
    assert lower["conditional_minimum_operating_days"] == 3
    assert lower["additional_days_due_to_range"] == 1
    assert bool(lower["conditional_plan_within_time_and_range"])
    assert not bool(lower["overnight_replenishment_verified"])
    assert summary.loc[
        summary["scenario"].eq("lower_bound"),
        "tasks_requiring_additional_days",
    ].iloc[0] == 1


def test_time_minimum_days_use_balanced_daily_distance() -> None:
    requirements = pd.DataFrame(
        {
            "task_id": ["long"],
            "origin_site_id": ["a"],
            "destination_site_id": ["b"],
            "distance_km": [700.0],
            "planning_duration_hours": [20.0],
            "required_operating_days": [2],
            "maximum_daily_distance_km": [490.0],
            "total_weight_kg": [1000.0],
        }
    )
    registry = {
        "fixed_parameters": {
            "hydrogen_winter_range_factor": {"value": 1.0, "status": "active"},
        },
        "candidate_vehicles": [
            {
                "model": "test hydrogen",
                "energy": "hydrogen",
                "nominal_range_km": [400.0, 400.0],
                "payload_t": [1.5, 1.5],
            }
        ],
    }

    detail, _ = evaluate_multiday_candidates(requirements, registry)

    result = detail.iloc[0]
    assert result["time_minimum_daily_distance_km"] == 350.0
    assert bool(result["range_feasible_at_time_minimum_days"])
    assert result["conditional_minimum_operating_days"] == 2
