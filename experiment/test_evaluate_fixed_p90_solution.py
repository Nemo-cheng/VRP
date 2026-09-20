import pandas as pd
from evaluate_fixed_p90_solution import evaluate_fixed_solution


def test_fixed_solution_uses_supplied_cost_without_reselection() -> None:
    tasks = pd.DataFrame(
        {
            "instance_id": ["i1", "i1"],
            "task_id": ["a", "b"],
            "origin_site_id": ["A", "C"],
            "destination_site_id": ["B", "D"],
            "departed_at": ["2023-12-01 08:00", "2023-12-01 10:00"],
            "arrived_at": ["2023-12-01 09:00", "2023-12-01 11:00"],
            "distance_km": [10.0, 20.0],
        }
    )
    links = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "from_task_id": ["a"],
            "to_task_id": ["b"],
            "deadhead_distance_km": [50.0],
            "deadhead_path": ["B>C"],
            "deadhead_duration_hours_p50": [0.5],
        }
    )
    chains = pd.DataFrame(
        {
            "instance_id": ["i1", "i1"],
            "baseline_variant": ["p90", "p90"],
            "historical_chain_id": ["h1", "h2"],
            "sequence": [1, 1],
            "task_id": ["a", "b"],
        }
    )
    task_types = {"a": {"van"}, "b": {"van"}}

    _, schedule, summary = evaluate_fixed_solution(
        tasks,
        task_types,
        links,
        chains,
        vehicle_cost=60.0,
        time_limit=10.0,
        selection_period="validation",
    )

    assert summary["vehicle_cost_equivalent_km"] == 60.0
    assert summary["recommended_vehicle_count"] == 1
    assert summary["no_final_test_parameter_reselection"]
    assert schedule["vehicle_cost_equivalent_km"].eq(60.0).all()
