from build_dual_solution_report import build_dual_solution_report


def test_report_retains_both_feasible_solutions_without_economic_ranking() -> None:
    table, report = build_dual_solution_report(
        {
            "p90_robust_vehicle_count": 8,
            "internal_deadhead_distance_km": 300.0,
            "task_service_rate": 1.0,
            "route_type_violations": 0,
            "time_overlap_violations": 0,
            "deadhead_endpoint_violations": 0,
            "missing_selected_link_paths": 0,
        },
        {
            "instances": 2,
            "tasks": 20,
            "historical_p90_chain_vehicle_count": 12,
            "historical_p90_chain_deadhead_distance_km": 200.0,
            "recommended_vehicle_count": 9,
            "recommended_deadhead_distance_km": 180.0,
            "vehicle_reduction": 3,
            "deadhead_reduction_km": 20.0,
            "task_service_rate": 1.0,
            "route_type_violations": 0,
            "time_overlap_violations": 0,
            "deadhead_endpoint_violations": 0,
            "missing_selected_link_paths": 0,
        },
    )

    assert table["solution"].tolist() == [
        "minimum_vehicle_p90",
        "historical_deadhead_controlled_p90",
    ]
    assert table.loc[0, "vehicle_reduction_vs_history"] == 4
    assert table.loc[0, "deadhead_reduction_vs_history_km"] == -100.0
    assert report["selection_status"] == "both_retained_no_economic_ranking"
    assert report["all_retained_solutions_feasible"]
