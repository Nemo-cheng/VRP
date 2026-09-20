import pandas as pd

from experiment.build_operational_route_vehicle_plan import (
    build_route_vehicle_assignment,
    build_task_level_plan,
    summarize_operating_cost,
)


def _registry() -> dict:
    return {
        "candidate_vehicles": [
            {
                "energy_cost_cny_per_km": 0.5,
                "purchase_cost_cny": 100.0,
                "maintenance_cost_cny_per_year": 10.0,
            },
            {
                "energy_cost_cny_per_km": 0.3,
                "purchase_cost_cny": 200.0,
                "maintenance_cost_cny_per_year": 20.0,
            },
        ],
        "interval_parameters": {
            "existing_diesel_energy_cost": {
                "minimum": 0.75,
                "maximum": 1.2,
                "status": "sensitivity_only",
            }
        },
    }


def test_assignment_selects_lowest_energy_cost_feasible_candidate() -> None:
    detail = pd.DataFrame(
        {
            "instance_id": ["i1", "i1"],
            "vehicle_id": ["v1", "v1"],
            "scenario": ["lower_bound", "lower_bound"],
            "candidate_id": ["candidate_01", "candidate_02"],
            "candidate_model": ["expensive", "cheap"],
            "energy": ["hydrogen", "electric"],
            "technical_feasible": [True, True],
            "operating_window_feasible": [True, True],
        }
    )
    schedule = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "vehicle_id": ["v1"],
            "task_id": ["t1"],
            "vehicle_type_name": ["diesel"],
            "distance_km": [100.0],
            "deadhead_to_next_distance_km": [10.0],
        }
    )

    result = build_route_vehicle_assignment(detail, schedule, _registry())

    assert result.iloc[0]["assigned_candidate_id"] == "candidate_02"
    assert result.iloc[0]["route_distance_km"] == 110.0


def test_cost_summary_keeps_unresolved_chain_on_correlated_diesel_interval() -> None:
    assignment = pd.DataFrame(
        {
            "scenario": ["lower_bound", "lower_bound"],
            "vehicle_id": ["v1", "v2"],
            "assignment_status": [
                "candidate_selected",
                "retain_pending_operating_plan",
            ],
            "assigned_energy": ["electric", "diesel"],
            "assigned_model": ["electric", "retain_existing_vehicle"],
            "assigned_energy_cost_cny_per_km": [0.3, pd.NA],
            "route_distance_km": [100.0, 100.0],
        }
    )

    _, summary = summarize_operating_cost(assignment, _registry())

    assert summary["baseline_energy_cost_cny"] == {
        "minimum": 150.0,
        "maximum": 240.0,
    }
    assert summary["plan_energy_cost_cny"] == {
        "minimum": 105.0,
        "maximum": 150.0,
    }
    assert summary["energy_cost_saving_cny"] == {
        "minimum": 45.0,
        "maximum": 90.0,
    }


def test_task_plan_preserves_route_sequence_and_adds_assignment() -> None:
    schedule = pd.DataFrame(
        {
            "instance_id": ["i1", "i1"],
            "vehicle_id": ["v1", "v1"],
            "sequence": [2, 1],
            "task_id": ["t2", "t1"],
        }
    )
    assignment = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "vehicle_id": ["v1"],
            "scenario": ["lower_bound"],
            "assignment_status": ["candidate_selected"],
            "assigned_candidate_id": ["candidate_01"],
            "assigned_model": ["electric"],
            "assigned_energy": ["electric"],
            "assigned_energy_cost_cny_per_km": [0.25],
        }
    )

    plan = build_task_level_plan(schedule, assignment)

    assert plan["task_id"].tolist() == ["t1", "t2"]
    assert set(plan["assigned_model"]) == {"electric"}
