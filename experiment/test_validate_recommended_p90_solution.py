import pandas as pd
from validate_recommended_p90_solution import (
    bootstrap_intervals,
    build_instance_comparison,
    summarize_validation,
)


def make_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    schedule = pd.DataFrame(
        {
            "instance_id": ["i1", "i1", "i2"],
            "vehicle_id": ["v1", "v1", "v1"],
            "task_id": ["a", "b", "c"],
            "deadhead_to_next_distance_km": [5.0, 0.0, 0.0],
        }
    )
    chains = pd.DataFrame(
        {
            "instance_id": ["i1", "i1", "i2"],
            "baseline_variant": ["p90", "p90", "p90"],
            "historical_chain_id": ["h1", "h2", "h3"],
            "sequence": [1, 1, 1],
            "task_id": ["a", "b", "c"],
        }
    )
    links = pd.DataFrame(
        columns=[
            "instance_id",
            "from_task_id",
            "to_task_id",
            "deadhead_distance_km",
        ]
    )
    return schedule, chains, links


def test_builds_paired_instance_comparison() -> None:
    schedule, chains, links = make_data()

    comparison = build_instance_comparison(schedule, chains, links).set_index(
        "instance_id"
    )

    assert comparison.loc["i1", "historical_vehicle_chain_count"] == 2
    assert comparison.loc["i1", "recommended_vehicle_count"] == 1
    assert comparison.loc["i1", "vehicle_reduction_rate"] == 0.5
    assert comparison.loc["i1", "deadhead_reduction_km"] == -5.0
    assert comparison.loc["i2", "both_improved_or_equal"] == False


def test_summary_counts_instance_outcomes() -> None:
    schedule, chains, links = make_data()
    comparison = build_instance_comparison(schedule, chains, links)

    summary = summarize_validation(
        comparison,
        samples=100,
        seed=7,
        solution_label="frozen validation choice",
        independent_final_test=True,
    )

    assert summary["instances"] == 2
    assert summary["solution"] == "frozen validation choice"
    assert "independent final-test estimate" in summary["inference_scope"]
    assert summary["historical_vehicle_chain_count"] == 3
    assert summary["recommended_vehicle_count"] == 2
    assert summary["instances_with_vehicle_reduction"] == 1
    assert summary["instances_with_vehicle_increase"] == 0
    assert summary["instances_with_no_deadhead_increase"] == 1
    assert summary["instances_with_deadhead_increase"] == 1


def test_bootstrap_is_reproducible() -> None:
    schedule, chains, links = make_data()
    comparison = build_instance_comparison(schedule, chains, links)

    first = bootstrap_intervals(comparison, samples=100, seed=11)
    second = bootstrap_intervals(comparison, samples=100, seed=11)

    assert first == second
