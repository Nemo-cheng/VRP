import pandas as pd
from compare_historical_vehicle_baseline import (
    attach_qualified_instances,
    build_comparison,
    build_historical_chains,
)


def make_tasks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instance_id": ["i1", "i1", "i1", "i1", "i2"],
            "task_id": ["a", "b", "c", "d", "e"],
            "departed_at": pd.to_datetime(
                [
                    "2023-10-01 08:00",
                    "2023-10-01 10:00",
                    "2023-10-01 12:00",
                    "2023-10-01 14:00",
                    "2023-10-02 08:00",
                ]
            ),
            "arrived_at": pd.to_datetime(
                [
                    "2023-10-01 09:00",
                    "2023-10-01 11:00",
                    "2023-10-01 13:00",
                    "2023-10-01 15:00",
                    "2023-10-02 09:00",
                ]
            ),
            "vehicle_type_name": ["van", "van", "truck", "truck", "van"],
            "historical_vehicle": ["v1", "v1", "v1", None, "v1"],
        }
    )


def test_attach_qualified_instances_excludes_unqualified_tasks() -> None:
    tasks = pd.DataFrame(
        {
            "task_id": ["a", "b"],
            "service_date": ["2023-10-01", "2023-10-02"],
            "component_id": ["c1", "c1"],
            "departed_at": ["2023-10-01 08:00", "2023-10-02 08:00"],
            "arrived_at": ["2023-10-01 09:00", "2023-10-02 09:00"],
        }
    )
    instances = pd.DataFrame(
        {
            "instance_id": ["i1", "i2"],
            "service_date": ["2023-10-01", "2023-10-02"],
            "component_id": ["c1", "c1"],
            "qualifies_for_vrp": [True, False],
        }
    )

    result = attach_qualified_instances(tasks, instances)

    assert result["task_id"].tolist() == ["a"]
    assert result["instance_id"].tolist() == ["i1"]


def test_historical_chains_split_on_type_and_infeasible_transition() -> None:
    tasks = make_tasks()
    links = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "from_task_id": ["a"],
            "to_task_id": ["b"],
        }
    )

    chains, metrics, diagnostics = build_historical_chains(tasks, links, "p50")

    assert metrics["verifiable_historical_chain_count"].sum() == 4
    assert chains.groupby("historical_chain_id").size().max() == 2
    assert diagnostics["observed_consecutive_transitions"] == 2
    assert diagnostics["verified_consecutive_transitions"] == 1
    assert diagnostics["transitions_split_by_vehicle_type_change"] == 1
    assert diagnostics["tasks_with_missing_historical_vehicle"] == 1


def test_comparison_uses_matching_p50_and_p90_historical_chains() -> None:
    tasks = make_tasks().iloc[:3].copy()
    p50_links = pd.DataFrame(
        {
            "instance_id": ["i1"],
            "from_task_id": ["a"],
            "to_task_id": ["b"],
        }
    )
    p90_links = pd.DataFrame(columns=["instance_id", "from_task_id", "to_task_id"])
    basic = pd.DataFrame({"instance_id": ["i1"], "vrp_vehicle_count": [2]})
    vehicle_type = pd.DataFrame(
        {"instance_id": ["i1"], "type_compatible_vehicle_count": [2]}
    )
    p50 = pd.DataFrame({"instance_id": ["i1"], "time_dependent_vehicle_count": [2]})
    p90 = pd.DataFrame({"instance_id": ["i1"], "p90_robust_vehicle_count": [3]})

    comparison, chains, summary = build_comparison(
        tasks, p50_links, p90_links, basic, vehicle_type, p50, p90
    )

    assert summary["raw_historical_vehicle_count"] == 1
    assert summary["p50_historical_chain_count"] == 2
    assert summary["p90_historical_chain_count"] == 3
    assert summary["p50_vehicle_reduction_vs_historical_chains"] == 0.0
    assert summary["p90_vehicle_reduction_vs_historical_chains"] == 0.0
    assert len(comparison) == 1
    assert set(chains["baseline_variant"]) == {"p50", "p90"}
