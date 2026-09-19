import pandas as pd
from optimize_vehicle_deadhead_tradeoff import (
    evaluate_scenario,
    historical_chain_metrics,
    mark_pareto_frontier,
    prepare_problem,
)


def make_tasks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance_id": ["i1", "i1"],
            "task_id": ["a", "b"],
            "origin_site_id": ["A", "C"],
            "destination_site_id": ["B", "D"],
            "departed_at": ["2023-10-01 08:00", "2023-10-01 10:00"],
            "arrived_at": ["2023-10-01 09:00", "2023-10-01 11:00"],
            "distance_km": [10.0, 20.0],
        }
    )


def make_links() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance_id": ["i1"],
            "from_task_id": ["a"],
            "to_task_id": ["b"],
            "deadhead_distance_km": [50.0],
            "deadhead_path": ["B>C"],
            "deadhead_duration_hours_p50": [0.5],
        }
    )


def test_prepare_problem_keeps_only_explicitly_qualified_instances() -> None:
    tasks = pd.DataFrame(
        {
            "task_id": ["a", "b"],
            "service_date": ["2023-10-01", "2023-10-02"],
            "component_id": ["c1", None],
            "origin_site_id": ["A", "X"],
            "destination_site_id": ["B", "Y"],
        }
    )
    instances = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "service_date": ["2023-10-01"],
            "component_id": ["c1"],
            "qualifies_for_vrp": [True],
        }
    )
    lane_types = pd.DataFrame(
        {
            "origin_site_id": ["A"],
            "destination_site_id": ["B"],
            "vehicle_type_name": ["van"],
        }
    )

    qualified, task_types = prepare_problem(tasks, instances, lane_types)

    assert qualified["task_id"].tolist() == ["a"]
    assert task_types == {"a": {"van"}}


def test_scenario_moves_between_vehicle_and_deadhead_objectives() -> None:
    tasks = make_tasks()
    links = make_links()
    task_types = {"a": {"van"}, "b": {"van"}}

    _, low = evaluate_scenario(tasks, task_types, links, 40.0, 10.0)
    _, high = evaluate_scenario(tasks, task_types, links, 60.0, 10.0)

    assert low["vehicle_count"] == 2
    assert low["internal_deadhead_distance_km"] == 0.0
    assert high["vehicle_count"] == 1
    assert high["internal_deadhead_distance_km"] == 50.0
    assert low["task_service_rate"] == high["task_service_rate"] == 1.0


def test_historical_chain_distance_is_recomputed_from_verified_links() -> None:
    chains = pd.DataFrame(
        {
            "instance_id": ["i1", "i1", "i1"],
            "baseline_variant": ["p50", "p50", "p90"],
            "historical_chain_id": ["c1", "c1", "c2"],
            "sequence": [1, 2, 1],
            "task_id": ["a", "b", "a"],
        }
    )

    result = historical_chain_metrics(chains, make_links())

    assert result["vehicle_count"] == 1
    assert result["selected_task_links"] == 1
    assert result["internal_deadhead_distance_km"] == 50.0


def test_pareto_frontier_removes_dominated_scenario() -> None:
    table = pd.DataFrame(
        {
            "scenario": ["a", "b", "c"],
            "vehicle_count": [10, 10, 9],
            "internal_deadhead_distance_km": [100.0, 120.0, 150.0],
        }
    )

    result = mark_pareto_frontier(table).set_index("scenario")

    assert result.loc["a", "pareto_efficient"]
    assert not result.loc["b", "pareto_efficient"]
    assert result.loc["c", "pareto_efficient"]
