import pandas as pd

from experiment.build_vehicle_replacement_boundary import (
    build_multiday_boundary,
    build_single_day_boundary,
    summarize_boundary,
)


def test_single_day_boundary_prioritizes_direct_electric() -> None:
    detail = pd.DataFrame(
        {
            "instance_id": ["i1", "i1"],
            "vehicle_id": ["v1", "v1"],
            "scenario": ["lower_bound", "lower_bound"],
            "energy": ["electric", "hydrogen"],
            "operating_window_feasible": [True, True],
            "technical_feasible": [True, True],
            "required_distance_km": [100.0, 100.0],
        }
    )

    boundary = build_single_day_boundary(detail)

    assert len(boundary) == 1
    assert boundary.iloc[0]["replacement_outcome"] == "direct_electric"


def test_noncompliant_single_day_chain_is_not_assigned_to_a_vehicle() -> None:
    detail = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "vehicle_id": ["v1"],
            "scenario": ["lower_bound"],
            "energy": ["hydrogen"],
            "operating_window_feasible": [False],
            "technical_feasible": [False],
            "required_distance_km": [100.0],
        }
    )

    assert build_single_day_boundary(detail).empty


def test_multiday_hydrogen_result_remains_conditional() -> None:
    detail = pd.DataFrame(
        {
            "task_id": ["t1", "t1"],
            "scenario": ["lower_bound", "lower_bound"],
            "energy": ["electric", "hydrogen"],
            "payload_feasible": [True, True],
            "range_feasible_at_time_minimum_days": [False, True],
            "task_distance_km": [700.0, 700.0],
        }
    )

    boundary = build_multiday_boundary(detail)

    result = boundary.iloc[0]
    assert result["replacement_outcome"] == (
        "conditional_hydrogen_with_overnight_replenishment"
    )
    assert bool(result["requires_verified_overnight_replenishment"])
    assert result["evidence_status"] == "conditional_unverified_replenishment"


def test_summary_share_is_calculated_within_layer_and_scenario() -> None:
    detail = pd.DataFrame(
        {
            "analysis_layer": ["single_day_chain"] * 2 + ["multiday_task"],
            "scenario": ["lower_bound"] * 3,
            "replacement_outcome": ["direct_electric", "hydrogen_only", "other"],
            "evidence_status": ["resolved", "resolved", "conditional"],
            "unit_id": ["a", "b", "c"],
            "distance_km": [1.0, 1.0, 1.0],
        }
    )

    summary = summarize_boundary(detail)

    single_day = summary[summary["analysis_layer"].eq("single_day_chain")]
    multiday = summary[summary["analysis_layer"].eq("multiday_task")]
    assert single_day["unit_share_within_layer_scenario"].tolist() == [0.5, 0.5]
    assert multiday.iloc[0]["unit_share_within_layer_scenario"] == 1.0
