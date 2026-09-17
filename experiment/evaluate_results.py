from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from run_unified_experiment import Instance, parse_instance


ROOT = Path(__file__).resolve().parents[1]
ROUTE_RE = re.compile(
    r"route\s+\d+,\s+node_num\s+\d+,\s+cost\s+([\d.]+),\s+nodes:\s+(.*)"
)
NODE_RE = re.compile(r"(\d+)(?:\(([-\d.]+),\s*([-\d.]+)\))?")


def parse_solution(path: Path):
    routes = []
    total_cost = math.nan
    solver_seconds = math.nan
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
        elif re.fullmatch(r"[\d.]+,\s*[\d.]+", line.strip()):
            solver_seconds = float(line.split(",", 1)[1].strip())
    if not routes or math.isnan(total_cost):
        raise ValueError(f"无法解析解文件：{path}")
    return routes, total_cost, solver_seconds


def evaluate(model: str, solution_path: Path, instance: Instance) -> dict[str, object]:
    routes, total_cost, solver_seconds = parse_solution(solution_path)
    capacity = float(instance.headers["CAPACITY"])
    consumption = float(instance.headers["CONSUMPTION_RATE"])
    battery = float(instance.headers["ELECTRIC_POWER"])
    max_range = battery / consumption
    recharge_rate = float(instance.headers["RECHARGING_RATE"])
    dispatch_cost = float(instance.headers["DISPATCHINGCOST"])
    unit_cost = float(instance.headers["UNITCOST"])

    total_distance = 0.0
    total_travel_time = 0.0
    total_waiting_time = 0.0
    total_charging_time = 0.0
    charging_visits = 0
    station_passes = 0
    minimum_range = max_range
    customer_visits: Counter[int] = Counter()
    capacity_feasible = True
    time_feasible = True
    battery_feasible = True
    route_structure_feasible = True

    for route in routes:
        route_structure_feasible &= (
            len(route) >= 2
            and route[0][0] == instance.depot
            and route[-1][0] == instance.depot
        )
        route_customers = [
            node_id
            for node_id, _, _ in route
            if instance.nodes[node_id]["type"] == "c"
        ]
        customer_visits.update(route_customers)
        load = sum(float(instance.nodes[node_id]["delivery"]) for node_id in route_customers)
        capacity_feasible &= load <= capacity + 1e-3
        current_time = float(instance.nodes[instance.depot]["ready"])
        remaining_range = max_range

        start_arrival, start_departure = route[0][1], route[0][2]
        if model == "evrp_tw_spd" and start_arrival is not None and start_departure is not None:
            battery_feasible &= abs(start_arrival - max_range) <= 1e-2
            battery_feasible &= abs(start_departure - max_range) <= 1e-2

        for index in range(1, len(route)):
            previous = route[index - 1][0]
            node_id, arrival_range, departure_range = route[index]
            edge = (previous, node_id)
            edge_distance = instance.distance[edge]
            edge_time = instance.travel_time[edge]
            total_distance += edge_distance
            total_travel_time += edge_time
            current_time += edge_time
            remaining_range -= edge_distance
            minimum_range = min(minimum_range, remaining_range)
            battery_feasible &= remaining_range >= -1e-2

            if model == "evrp_tw_spd" and arrival_range is not None:
                battery_feasible &= abs(arrival_range - remaining_range) <= 1e-2

            node = instance.nodes[node_id]
            time_feasible &= current_time <= float(node["due"]) + 1e-2
            if current_time < float(node["ready"]):
                wait = float(node["ready"]) - current_time
                total_waiting_time += wait
                current_time += wait

            if node["type"] == "f":
                station_passes += 1
                if arrival_range is not None and departure_range is not None:
                    battery_feasible &= departure_range + 1e-2 >= arrival_range
                    battery_feasible &= departure_range <= max_range + 1e-2
                    charged_range = max(0.0, departure_range - arrival_range)
                    if charged_range > 1e-2:
                        charging_visits += 1
                        charge_time = charged_range * consumption * recharge_rate
                        total_charging_time += charge_time
                        current_time += charge_time
                    if model == "evrp_tw_spd":
                        remaining_range = departure_range
            elif node["type"] == "c":
                load -= float(node["delivery"])
                load += float(node["pickup"])
                capacity_feasible &= load <= capacity + 1e-3
                current_time += float(node["service"])

    expected_customers = {
        node_id for node_id, node in instance.nodes.items() if node["type"] == "c"
    }
    served_customers = set(customer_visits)
    coverage = len(served_customers & expected_customers) / len(expected_customers)
    unexpected_customer_visits = set(customer_visits) - expected_customers
    duplicate_customer_visits = sum(
        max(0, visits - 1) for visits in customer_visits.values()
    )
    customers_served_once = (
        not unexpected_customer_visits
        and all(customer_visits[node_id] == 1 for node_id in expected_customers)
    )
    solver_distance = (total_cost - dispatch_cost * len(routes)) / unit_cost
    distance_difference = total_distance - solver_distance
    full_model_feasible: bool | None = None
    if model == "evrp_tw_spd":
        full_model_feasible = (
            capacity_feasible
            and time_feasible
            and battery_feasible
            and customers_served_once
            and route_structure_feasible
        )

    return {
        "model": model,
        "route_count": len(routes),
        "total_cost": round(total_cost, 4),
        "total_distance": round(total_distance, 4),
        "distance_check_difference": round(distance_difference, 4),
        "total_travel_time": round(total_travel_time, 4),
        "total_waiting_time": round(total_waiting_time, 4),
        "total_charging_time": round(total_charging_time, 4),
        "station_passes": station_passes,
        "charging_visits": charging_visits,
        "total_energy": round(total_distance * consumption, 6),
        "minimum_remaining_range": round(minimum_range, 4),
        "minimum_battery_energy": round(minimum_range * consumption, 6),
        "customer_coverage": round(coverage, 6),
        "customers_served_once": customers_served_once,
        "duplicate_customer_visits": duplicate_customer_visits,
        "route_structure_feasible": route_structure_feasible,
        "capacity_feasible": capacity_feasible,
        "time_windows_feasible": time_feasible if model != "vrp_spd" else None,
        "battery_feasible": battery_feasible if model == "evrp_tw_spd" else None,
        "solver_model_feasible": True,
        "full_model_feasible": full_model_feasible,
        "solver_seconds": round(solver_seconds, 3),
        "solution_file": str(solution_path.relative_to(ROOT)),
    }


def evaluate_all(instance_path: Path, time_limit: int) -> list[dict[str, object]]:
    instance = parse_instance(instance_path)
    rows = []
    for model in ("vrp_spd", "vrp_tw_spd", "evrp_tw_spd"):
        filename = f"jd200_1_{model}_timelimit={time_limit}_subproblem=2.txt"
        solution_path = ROOT / "results" / "raw" / filename
        rows.append(evaluate(model, solution_path, instance))

    results_dir = ROOT / "results"
    csv_path = results_dir / "unified_experiment_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / "unified_experiment_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows
