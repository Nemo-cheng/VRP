import pandas as pd
import pytest

from experiment.evaluate_candidate_vehicle_feasibility import (
    build_chain_requirements,
    candidate_limits,
)
from experiment.validate_parameter_registry import load_registry


def test_electric_lower_bound_applies_winter_and_three_year_degradation() -> None:
    registry = load_registry()
    candidate = registry["candidate_vehicles"][0]

    payload_t, effective_range = candidate_limits(
        candidate, "lower_bound", registry
    )

    assert payload_t == pytest.approx(1.5)
    assert effective_range == pytest.approx(280 * 0.7 * 0.95**3)


def test_hydrogen_upper_bound_applies_winter_factor_only() -> None:
    registry = load_registry()
    candidate = registry["candidate_vehicles"][-1]

    payload_t, effective_range = candidate_limits(
        candidate, "upper_bound", registry
    )

    assert payload_t == pytest.approx(15.0)
    assert effective_range == pytest.approx(450 * 0.9)


def test_chain_requirements_use_retimed_duration() -> None:
    schedule = pd.DataFrame(
        {
            "instance_id": ["i1", "i1"],
            "vehicle_id": ["v1", "v1"],
            "task_id": ["event_1", "event_2"],
            "departed_at": ["2023-12-01 01:00", "2023-12-01 20:00"],
            "arrived_at": ["2023-12-01 03:00", "2023-12-01 22:00"],
            "distance_km": [10.0, 20.0],
            "deadhead_to_next_distance_km": [2.0, 0.0],
            "deadhead_to_next_duration_hours": [1.0, 0.0],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["event_1", "event_2"],
            "total_weight_kg": [100.0, 200.0],
        }
    )

    requirements, _ = build_chain_requirements(schedule, events)

    assert requirements.iloc[0]["chain_operating_hours"] == pytest.approx(5.0)
    assert bool(requirements.iloc[0]["operating_window_compliant"])
