from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from evaluate_results import evaluate_all
from run_unified_experiment import (
    DEFAULT_INSTANCE,
    IMAGE,
    ROOT,
    prepare_variants,
    run_solver,
)


def build_image() -> None:
    context = ROOT / "external_data" / "EVRP-TW-SPD-HMA"
    dockerfile = ROOT / "experiment" / "Dockerfile.runtime"
    subprocess.run(
        ["docker", "build", "-t", IMAGE, "-f", str(dockerfile), str(context)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    instance_path = args.instance.resolve()
    variants = prepare_variants(instance_path, ROOT / "experiment" / "instances")
    if args.prepare_only:
        print("Prepared:", ", ".join(str(path) for path in variants.values()))
        return
    if not args.skip_build:
        build_image()

    for model, variant in variants.items():
        print(f"Running {model} with {args.time_limit}s limit")
        run_solver(model, variant, args.time_limit, args.seed)

    rows = evaluate_all(instance_path, args.time_limit)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
