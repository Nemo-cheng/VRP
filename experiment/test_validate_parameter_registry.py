from pathlib import Path

from experiment.validate_parameter_registry import load_registry, validate_registry


def test_project_parameter_registry_is_valid() -> None:
    registry = load_registry()

    assert validate_registry(registry) == []
    assert len(registry["candidate_vehicles"]) == 6


def test_active_parameter_without_source_is_rejected(tmp_path: Path) -> None:
    registry = load_registry()
    registry["fixed_parameters"]["carbon_reduction_target"].pop("source")

    errors = validate_registry(registry)

    assert "carbon_reduction_target: missing source" in errors


def test_unresolved_parameter_cannot_be_marked_active() -> None:
    registry = load_registry()
    registry["blocked_parameters"]["existing_vehicle_emission_factors_uncovered"][
        "status"
    ] = "active"

    errors = validate_registry(registry)

    assert any("must remain blocked" in error for error in errors)
