from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluate_results import evaluate
from run import build_image
from run_unified_experiment import (
    DEFAULT_INSTANCE,
    IMAGE,
    ROOT,
    parse_instance,
    prepare_variants,
    solver_args,
)


MODELS = ("vrp_spd", "vrp_tw_spd", "evrp_tw_spd")
SUMMARY_METRICS = (
    "route_count",
    "total_cost",
    "total_distance",
    "total_travel_time",
    "total_waiting_time",
    "charging_visits",
    "total_charging_time",
    "total_energy",
)


def run_directory(model: str, seed: int, time_limit: int) -> Path:
    return ROOT / "results" / "runs" / f"time_{time_limit}" / f"seed_{seed}" / model


def solution_path(model: str, seed: int, time_limit: int) -> Path:
    filename = f"jd200_1_{model}_timelimit={time_limit}_subproblem=2.txt"
    return run_directory(model, seed, time_limit) / filename


def run_one(model: str, variant: Path, seed: int, time_limit: int) -> float:
    output_dir = run_directory(model, seed, time_limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_solution = solution_path(model, seed, time_limit)
    metadata_path = output_dir / "run.json"
    if expected_solution.exists():
        print(f"Skipping completed run: {model}, seed={seed}")
        if metadata_path.exists():
            return float(json.loads(metadata_path.read_text(encoding="utf-8"))["wall_seconds"])
        return 0.0

    workspace_mount = f"{ROOT}:/workspace"
    problem = f"/workspace/{variant.relative_to(ROOT).as_posix()}"
    container_output = f"/workspace/{output_dir.relative_to(ROOT).as_posix()}/"
    command = ["docker", "run", "--rm", "-v", workspace_mount, IMAGE]
    command.extend(solver_args(problem, container_output, time_limit, seed))

    print(f"Running {model}, seed={seed}, limit={time_limit}s")
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True)
    wall_seconds = time.perf_counter() - started
    (output_dir / "solver.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    metadata = {
        "model": model,
        "seed": seed,
        "time_limit": time_limit,
        "wall_seconds": wall_seconds,
        "return_code": completed.returncode,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"求解失败：{model}, seed={seed}，请检查 {output_dir / 'solver.log'}")
    if not expected_solution.exists():
        raise FileNotFoundError(f"求解器未生成预期结果：{expected_solution}")
    return wall_seconds


def collect_rows(instance_path: Path, seeds: list[int], time_limit: int) -> list[dict[str, object]]:
    instance = parse_instance(instance_path)
    rows = []
    for seed in seeds:
        for model in MODELS:
            path = solution_path(model, seed, time_limit)
            row = evaluate(model, path, instance)
            row = {"seed": seed, "time_limit": time_limit, **row}
            metadata_path = run_directory(model, seed, time_limit) / "run.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            row["wall_seconds"] = round(float(metadata["wall_seconds"]), 3)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent_change(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (current - baseline) / baseline * 100.0


def paired_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_seed = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[str(row["model"])] = row

    output = []
    comparisons = (
        ("time_window_effect", "vrp_spd", "vrp_tw_spd"),
        ("battery_charging_effect", "vrp_tw_spd", "evrp_tw_spd"),
    )
    for seed, model_rows in sorted(by_seed.items()):
        for comparison, baseline_name, current_name in comparisons:
            baseline = model_rows[baseline_name]
            current = model_rows[current_name]
            item: dict[str, object] = {
                "seed": seed,
                "comparison": comparison,
                "baseline_model": baseline_name,
                "current_model": current_name,
            }
            for metric in SUMMARY_METRICS:
                item[f"{metric}_change_pct"] = round(
                    percent_change(float(current[metric]), float(baseline[metric])), 4
                )
            output.append(item)
    return output


def model_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        item: dict[str, object] = {"model": model, "runs": len(model_rows)}
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in model_rows]
            item[f"{metric}_mean"] = round(statistics.mean(values), 4)
            item[f"{metric}_std"] = round(statistics.pstdev(values), 4)
            item[f"{metric}_best"] = round(min(values), 4)
        output.append(item)
    return output


def write_summary_markdown(
    path: Path,
    summary: list[dict[str, object]],
    deltas: list[dict[str, object]],
    seeds: list[int],
    time_limit: int,
) -> None:
    lines = [
        "# 批量实验汇总",
        "",
        f"随机种子：{', '.join(str(seed) for seed in seeds)}",
        "",
        f"单次求解上限：{time_limit} 秒",
        "",
        "| 模型 | 运行次数 | 车辆数 | 总成本 | 总距离 | 总行驶时间 | 充电次数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {model} | {runs} | {routes:.2f} ± {routes_std:.2f} | "
            "{cost:.2f} ± {cost_std:.2f} | {distance:.2f} ± {distance_std:.2f} | "
            "{travel:.2f} ± {travel_std:.2f} | {charging:.2f} ± {charging_std:.2f} |".format(
                model=row["model"],
                runs=row["runs"],
                routes=row["route_count_mean"],
                routes_std=row["route_count_std"],
                cost=row["total_cost_mean"],
                cost_std=row["total_cost_std"],
                distance=row["total_distance_mean"],
                distance_std=row["total_distance_std"],
                travel=row["total_travel_time_mean"],
                travel_std=row["total_travel_time_std"],
                charging=row["charging_visits_mean"],
                charging_std=row["charging_visits_std"],
            )
        )

    lines.extend(
        [
            "",
            "## 同种子成对变化",
            "",
            "| 对比 | 总成本变化 | 总距离变化 | 车辆数变化 | 行驶时间变化 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for comparison in ("time_window_effect", "battery_charging_effect"):
        selected = [row for row in deltas if row["comparison"] == comparison]
        lines.append(
            "| {comparison} | {cost:.2f}% | {distance:.2f}% | {routes:.2f}% | {travel:.2f}% |".format(
                comparison=comparison,
                cost=statistics.mean(float(row["total_cost_change_pct"]) for row in selected),
                distance=statistics.mean(float(row["total_distance_change_pct"]) for row in selected),
                routes=statistics.mean(float(row["route_count_change_pct"]) for row in selected),
                travel=statistics.mean(float(row["total_travel_time_change_pct"]) for row in selected),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_output_set(
    output_dir: Path,
    rows: list[dict[str, object]],
    seeds: list[int],
    time_limit: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    deltas = paired_deltas(rows)
    summary = model_summary(rows)
    write_csv(output_dir / "batch_metrics.csv", rows)
    write_csv(output_dir / "batch_paired_deltas.csv", deltas)
    (output_dir / "batch_metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_markdown(
        output_dir / "batch_summary.md", summary, deltas, seeds, time_limit
    )


def write_outputs(rows: list[dict[str, object]], seeds: list[int], time_limit: int) -> None:
    results_dir = ROOT / "results"
    write_output_set(results_dir, rows, seeds, time_limit)
    seed_set = "-".join(str(seed) for seed in seeds)
    archive_dir = (
        results_dir / "batches" / f"time_{time_limit}" / f"seeds_{seed_set}"
    )
    write_output_set(archive_dir, rows, seeds, time_limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    instance_path = args.instance.resolve()
    variants = prepare_variants(instance_path, ROOT / "experiment" / "instances")
    if args.prepare_only:
        print("Prepared:", ", ".join(str(path) for path in variants.values()))
        return
    if not args.skip_build:
        build_image()

    for seed in args.seeds:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_one, model, variant, seed, args.time_limit)
                for model, variant in variants.items()
            ]
            for future in futures:
                future.result()

    rows = collect_rows(instance_path, args.seeds, args.time_limit)
    write_outputs(rows, args.seeds, args.time_limit)
    print(json.dumps(model_summary(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
