import pytest

from experiment.evaluate_candidate_vehicle_feasibility import candidate_limits
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
