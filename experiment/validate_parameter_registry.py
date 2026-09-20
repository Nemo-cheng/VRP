"""Validate that experiment parameters have explicit evidence and status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY = Path(__file__).with_name("experiment_parameters.json")


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    sources = registry.get("sources", {})

    def require_source(name: str, item: dict[str, Any]) -> None:
        source = item.get("source")
        if not source:
            errors.append(f"{name}: missing source")
        elif source not in sources:
            errors.append(f"{name}: unknown source {source}")

    for name, item in registry.get("fixed_parameters", {}).items():
        require_source(name, item)
        if item.get("status") != "active":
            errors.append(f"{name}: fixed parameter must be active")

    for name, item in registry.get("interval_parameters", {}).items():
        require_source(name, item)
        if item.get("minimum") is None or item.get("maximum") is None:
            errors.append(f"{name}: interval bounds are required")
        elif item["minimum"] > item["maximum"]:
            errors.append(f"{name}: minimum exceeds maximum")
        if item.get("status") != "sensitivity_only":
            errors.append(f"{name}: interval parameter must be sensitivity_only")

    for name, item in registry.get("sensitivity_parameters", {}).items():
        require_source(name, item)
        if item.get("value") is None:
            errors.append(f"{name}: sensitivity value is required")
        if item.get("status") != "sensitivity_only":
            errors.append(f"{name}: sensitivity parameter must be sensitivity_only")

    for index, vehicle in enumerate(registry.get("candidate_vehicles", [])):
        name = f"candidate_vehicles[{index}]"
        require_source(name, vehicle)
        for field in (
            "model",
            "energy",
            "nominal_range_km",
            "payload_t",
            "purchase_cost_cny",
            "energy_cost_cny_per_km",
            "maintenance_cost_cny_per_year",
        ):
            if field not in vehicle:
                errors.append(f"{name}: missing {field}")

    for name, item in registry.get("trained_parameters", {}).items():
        training_source = item.get("training_source")
        if training_source not in sources:
            errors.append(f"{name}: unknown training source {training_source}")
        if not item.get("selection_rule"):
            errors.append(f"{name}: missing selection rule")

    for name, item in registry.get("blocked_parameters", {}).items():
        if item.get("status") != "blocked":
            errors.append(f"{name}: unresolved parameter must remain blocked")
        if not item.get("reason"):
            errors.append(f"{name}: missing blocked reason")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="校验实验参数的来源和启用状态。")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()

    errors = validate_registry(load_registry(args.registry))
    if errors:
        for error in errors:
            print(error)
        return 1

    print("parameter registry valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
