import pandas as pd
import pytest

from experiment.summarize_route_distance_effect import summarize_distance_effect


def test_distance_effect_does_not_treat_vehicle_reduction_as_mileage() -> None:
    schedule = pd.DataFrame({"distance_km": [60.0, 40.0]})
    validation = {
        "tasks": 2,
        "historical_vehicle_chain_count": 2,
        "recommended_vehicle_count": 1,
        "aggregate_vehicle_reduction_rate": 0.5,
        "historical_deadhead_distance_km": 10.0,
        "recommended_deadhead_distance_km": 5.0,
        "aggregate_deadhead_reduction_km_95_ci": [-1.0, 11.0],
    }

    result = summarize_distance_effect(schedule, validation)

    assert result["vehicle_chain_reduction_rate"] == pytest.approx(0.5)
    assert result["total_distance_reduction_rate"] == pytest.approx(5 / 110)
    assert result["loaded_distance_changed"] is False
