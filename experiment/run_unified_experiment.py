from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCE = (
    ROOT
    / "external_data"
    / "EVRP-TW-SPD-HMA"
    / "data"
    / "jd_instances"
    / "jd200_1.txt"
)
IMAGE = "jd-unified-evrp-experiment:latest"


@dataclass
class Instance:
    headers: dict[str, float | int | str]
    nodes: dict[int, dict[str, float | int | str]]
    distance: dict[tuple[int, int], float]
    travel_time: dict[tuple[int, int], float]
    depot: int


def parse_instance(path: Path) -> Instance:
    headers: dict[str, float | int | str] = {}
    nodes: dict[int, dict[str, float | int | str]] = {}
    distance: dict[tuple[int, int], float] = {}
    travel_time: dict[tuple[int, int], float] = {}
    depot = 0
    section = "header"

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "NODE_SECTION":
                section = "nodes"
                continue
            if line == "DISTANCETIME_SECTION":
                section = "edges"
                continue
            if line == "DEPOT_SECTION":
                section = "depot"
                continue

            if section == "header" and ":" in line:
                key, value = (part.strip() for part in line.split(":", 1))
                if key in {"DIMENSION", "VEHICLES"}:
                    headers[key] = int(value)
                elif key in {
                    "DISPATCHINGCOST",
                    "UNITCOST",
                    "CAPACITY",
                    "ELECTRIC_POWER",
                    "CONSUMPTION_RATE",
                    "RECHARGING_RATE",
                }:
                    headers[key] = float(value)
                else:
                    headers[key] = value
                continue

            if section == "nodes":
                if line.startswith("ID,"):
                    continue
                parts = line.split(",")
                node_id = int(parts[0])
                nodes[node_id] = {
                    "type": parts[1],
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                    "delivery": float(parts[4]),
                    "pickup": float(parts[5]),
                    "ready": float(parts[6]),
                    "due": float(parts[7]),
                    "service": float(parts[8]),
                }
                continue

            if section == "edges":
                if line.startswith("ID,"):
                    continue
                parts = line.split(",")
                origin, destination = int(parts[1]), int(parts[2])
                distance[(origin, destination)] = float(parts[3])
                travel_time[(origin, destination)] = float(parts[4])
                continue

            if section == "depot":
                depot = int(line)

    return Instance(headers, nodes, distance, travel_time, depot)


def prepare_variants(source: Path, output_dir: Path) -> dict[str, Path]:
    instance = parse_instance(source)
    source_lines = source.read_text(encoding="utf-8").splitlines()
    customers = [node for node in instance.nodes.values() if node["type"] == "c"]
    max_edge_distance = max(instance.distance.values())
    max_edge_time = max(instance.travel_time.values())
    total_service = sum(float(node["service"]) for node in customers)
    customer_count = len(customers)
    horizon = math.ceil((customer_count + 2) * max_edge_time + total_service)
    consumption = float(instance.headers["CONSUMPTION_RATE"])
    relaxed_battery = (customer_count + 2) * max_edge_distance * consumption * 1.1

    settings = {
        "vrp_spd": {"relax_time": True, "relax_battery": True},
        "vrp_tw_spd": {"relax_time": False, "relax_battery": True},
        "evrp_tw_spd": {"relax_time": False, "relax_battery": False},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Path] = {}

    for model, options in settings.items():
        rendered: list[str] = []
        section = "header"
        for line in source_lines:
            stripped = line.strip()
            if stripped == "NODE_SECTION":
                section = "nodes"
                rendered.append(line)
                continue
            if stripped == "DISTANCETIME_SECTION":
                section = "edges"
                rendered.append(line)
                continue
            if stripped == "DEPOT_SECTION":
                section = "depot"
                rendered.append(line)
                continue

            if stripped.startswith("NAME") and ":" in stripped:
                rendered.append(f"NAME : jd200_1_{model}")
            elif options["relax_battery"] and stripped.startswith("ELECTRIC_POWER"):
                rendered.append(f"ELECTRIC_POWER : {relaxed_battery:.12f}")
            elif (
                options["relax_time"]
                and section == "nodes"
                and stripped
                and not stripped.startswith("ID,")
            ):
                parts = line.split(",")
                if len(parts) == 9:
                    parts[6] = "0"
                    parts[7] = str(horizon)
                    rendered.append(",".join(parts))
                else:
                    rendered.append(line)
            else:
                rendered.append(line)

        output_path = output_dir / f"jd200_1_{model}.txt"
        output_path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
        variants[model] = output_path

    manifest = {
        "source": str(source),
        "shared_data": [
            "customers",
            "depot",
            "charging_stations",
            "delivery",
            "pickup",
            "distance_matrix",
            "travel_time_matrix",
            "vehicle_capacity",
        ],
        "relaxed_time_horizon": horizon,
        "relaxed_battery_capacity": relaxed_battery,
        "models": settings,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return variants


def build_image() -> None:
    context = ROOT / "external_data" / "EVRP-TW-SPD-HMA"
    dockerfile = ROOT / "experiment" / "Dockerfile"
    subprocess.run(
        ["docker", "build", "-t", IMAGE, "-f", str(dockerfile), str(context)],
        check=True,
    )


def solver_args(problem: str, output: str, time_limit: int, seed: int) -> list[str]:
    return [
        "--problem", problem,
        "--pruning",
        "--output", output,
        "--time", str(time_limit),
        "--runs", "1",
        "--g_1", "20",
        "--pop_size", "4",
        "--init", "rcrs",
        "--cross_repair", "regret",
        "--parent_selection", "circle",
        "--replacement", "one_on_one",
        "--O_1_eval",
        "--two_opt",
        "--two_opt_star",
        "--or_opt", "2",
        "--two_exchange", "2",
        "--elo", "1",
        "--related_removal",
        "--removal_lower", "0.05",
        "--removal_upper", "0.05",
        "--regret_insertion",
        "--individual_search",
        "--population_search",
        "--parallel_insertion",
        "--aggressive_local_search",
        "--station_range", "0.1",
        "--subproblem_range", "2",
        "--random_seed", str(seed),
    ]


def run_solver(model: str, variant: Path, time_limit: int, seed: int) -> float:
    raw_dir = ROOT / "results" / "raw"
    log_dir = ROOT / "results" / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    workspace_mount = f"{ROOT}:/workspace"
    problem = f"/workspace/{variant.relative_to(ROOT).as_posix()}"
    output = "/workspace/results/raw/"
    command = ["docker", "run", "--rm", "-v", workspace_mount, IMAGE]
    command.extend(solver_args(problem, output, time_limit, seed))

    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    (log_dir / f"{model}.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{model} failed. See {log_dir / f'{model}.log'}")
    return elapsed


ROUTE_RE = re.compile(
    r"route\s+\d+,\s+node_num\s+\d+,\s+cost\s+([\d.]+),\s+nodes:\s+(.*)"
)
NODE_RE = re.compile(r"(\d+)(?:\(([-\d.]+),\s*([-\d.]+)\))?")


def parse_solution(
    path: Path,
) -> tuple[list[list[tuple[int, float | None, float | None]]], float]:
    routes: list[list[tuple[int, float | None, float | None]]] = []
    total_cost = math.nan
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROUTE_RE.match(line.strip())
        if match:
            route = []
            for node_match in NODE_RE.finditer(match.group(2)):
                route.append(
                    (
                        int(node_match.group(1)),
                        float(node_match.group(2)) if node_match.group(2) else None,
                        float(node_match.group(3)) if node_match.group(3) else None,
                    )
                )
            routes.append(route)
        elif line.startswith("Total cost:"):
            total_cost = float(line.split(":", 1)[1].strip())
    if not routes or math.isnan(total_cost):
        raise ValueError(f"Cannot parse solution: {path}")
    return routes, total_cost


def evaluate_solution(
    model: str,
    solution_path: Path,
    original: Instance,
    runtime: float,
) -> dict[str, object]:
    routes, total_cost = parse_solution(solution_path)
    capacity = float(original.headers["CAPACITY"])
    battery_capacity = float(original.headers["ELECTRIC_POWER"])
    consumption = float(original.headers["CONSUMPTION_RATE"])
    recharge_rate = float(original.headers["RECHARGING_RATE"])
    dispatch_cost = float(original.headers["DISPATCHINGCOST"])
    unit_cost = float(original.headers["UNITCOST"])

    total_distance = max(0.0, (total_cost - dispatch_cost * len(routes)) / unit_cost)
    total_travel_time = 0.0
    total_waiting_time = 0.0
    charging_visits = 0
    minimum_battery = battery_capacity
    capacity_feasible = True
    time_feasible = True
    battery_feasible = True
    served_customers: set[int] = set()

    for route in routes:
        current_time = float(original.nodes[original.depot]["ready"])
        current_battery = battery_capacity
        customer_ids = [
            node_id
            for node_id, _, _ in route
            if original.nodes[node_id]["type"] == "c"
        ]
        served_customers.update(customer_ids)
        current_load = sum(
            float(original.nodes[node_id]["delivery"]) for node_id in customer_ids
        )
        capacity_feasible &= current_load <= capacity + 1e-6

        for index in range(1, len(route)):
            previous = route[index - 1][0]
            node_id, arrival_charge, departure_charge = route[index]
            edge = (previous, node_id)
            edge_distance = original.distance[edge]
            edge_time = original.travel_time[edge]
            current_time += edge_time
            total_travel_time += edge_time
            current_battery -= edge_distance * consumption
            minimum_battery = min(minimum_battery, current_battery)
            if current_battery < -1e-3:
                battery_feasible = False

            node = original.nodes[node_id]
            if current_time > float(node["due"]) + 1e-3:
                time_feasible = False
            if current_time < float(node["ready"]):
                total_waiting_time += float(node["ready"]) - current_time
                current_time = float(node["ready"])

            if node["type"] == "f":
                charging_visits += 1
                if arrival_charge is not None and departure_charge is not None:
                    charged = max(0.0, departure_charge - arrival_charge)
                    current_time += charged * recharge_rate
                    current_battery = departure_charge
                else:
                    current_battery = battery_capacity
            elif node["type"] == "c":
                current_load -= float(node["delivery"])
                current_load += float(node["pickup"])
                capacity_feasible &= current_load <= capacity + 1e-6
                current_time += float(node["service"])

    expected_customers = {
        node_id for node_id, node in original.nodes.items() if node["type"] == "c"
    }
    customer_coverage = len(served_customers) / len(expected_customers)
    full_feasible = (
        capacity_feasible
        and time_feasible
        and battery_feasible
        and customer_coverage == 1.0
    )
    return {
        "model": model,
        "route_count": len(routes),
        "total_cost": round(total_cost, 4),
        "total_distance": round(total_distance, 4),
        "total_travel_time": round(total_travel_time, 4),
        "total_waiting_time": round(total_waiting_time, 4),
        "charging_visits": charging_visits,
        "total_energy": round(total_distance * consumption, 6),
        "minimum_battery_full_evaluation": round(minimum_battery, 6),
        "customer_coverage": round(customer_coverage, 6),
        "capacity_feasible_under_full_model": capacity_feasible,
        "time_feasible_under_full_model": time_feasible,
        "battery_feasible_under_full_model": battery_feasible,
        "feasible_under_full_model": full_feasible,
        "runtime_seconds": round(runtime, 3),
        "solution_file": str(solution_path.relative_to(ROOT)),
    }


def find_solution(model: str, time_limit: int) -> Path:
    filename = f"jd200_1_{model}_timelimit={time_limit}_subproblem=2.txt"
    expected = ROOT / "results" / "raw" / filename
    if not expected.exists():
        raise FileNotFoundError(expected)
    return expected


def write_metrics(rows: list[dict[str, object]]) -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "unified_experiment_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "unified_experiment_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
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

    runtimes = {}
    for model, variant in variants.items():
        print(f"Running {model} with {args.time_limit}s limit")
        runtimes[model] = run_solver(model, variant, args.time_limit, args.seed)

    original = parse_instance(instance_path)
    rows = [
        evaluate_solution(
            model,
            find_solution(model, args.time_limit),
            original,
            runtimes[model],
        )
        for model in variants
    ]
    write_metrics(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
