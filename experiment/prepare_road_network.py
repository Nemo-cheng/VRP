#!/usr/bin/env python3
"""Download matched historical OSM graphs and build sparse benchmark OD matrices."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "networkx>=3.2",
#   "numpy>=1.26,<2.5",
#   "osmnx>=2.0,<3",
#   "pandas>=2.2",
#   "pyarrow>=15",
#   "scipy>=1.13,<1.17",
# ]
# ///

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer

OSM_SNAPSHOT = "2022-10-31T23:59:59Z"
GRAPH_BUFFER_DEGREES = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备代表实例的 OSM 路网和 OD 矩阵。")
    parser.add_argument(
        "--instances",
        type=Path,
        default=Path("processed/benchmark_instances/benchmark_seed_1103.parquet"),
    )
    parser.add_argument(
        "--graph-dir", type=Path, default=Path("processed/osm/graphs")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("processed/osm")
    )
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/road_network")
    )
    return parser.parse_args()


def _transform_lat(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        -100
        + 2 * x
        + 3 * y
        + 0.2 * y**2
        + 0.1 * x * y
        + 0.2 * np.sqrt(np.abs(x))
        + (20 * np.sin(6 * x * np.pi) + 20 * np.sin(2 * x * np.pi))
        * 2
        / 3
        + (20 * np.sin(y * np.pi) + 40 * np.sin(y / 3 * np.pi)) * 2 / 3
        + (160 * np.sin(y / 12 * np.pi) + 320 * np.sin(y * np.pi / 30))
        * 2
        / 3
    )


def _transform_lng(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        300
        + x
        + 2 * y
        + 0.1 * x**2
        + 0.1 * x * y
        + 0.1 * np.sqrt(np.abs(x))
        + (20 * np.sin(6 * x * np.pi) + 20 * np.sin(2 * x * np.pi))
        * 2
        / 3
        + (20 * np.sin(x * np.pi) + 40 * np.sin(x / 3 * np.pi)) * 2 / 3
        + (150 * np.sin(x / 12 * np.pi) + 300 * np.sin(x / 30 * np.pi))
        * 2
        / 3
    )


def gcj02_to_wgs84(
    lng: pd.Series | np.ndarray, lat: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lng_array = np.asarray(lng, dtype=float)
    lat_array = np.asarray(lat, dtype=float)
    x = lng_array - 105.0
    y = lat_array - 35.0
    delta_lat = _transform_lat(x, y)
    delta_lng = _transform_lng(x, y)
    rad_lat = lat_array / 180 * np.pi
    magic = np.sin(rad_lat)
    magic = 1 - 0.00669342162296594323 * magic**2
    sqrt_magic = np.sqrt(magic)
    delta_lat = (
        delta_lat
        * 180
        / ((6378245.0 * (1 - 0.00669342162296594323)) / (magic * sqrt_magic) * np.pi)
    )
    delta_lng = (
        delta_lng
        * 180
        / (6378245.0 / sqrt_magic * np.cos(rad_lat) * np.pi)
    )
    return lng_array * 2 - (lng_array + delta_lng), lat_array * 2 - (
        lat_array + delta_lat
    )


def configure_osmnx() -> None:
    ox.settings.requests_timeout = 300
    ox.settings.use_cache = True
    ox.settings.overpass_settings = (
        f'[out:json][timeout:{{timeout}}]{{maxsize}}[date:"{OSM_SNAPSHOT}"]'
    )


def route_points(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("stop_index", kind="stable")
    first = ordered.iloc[0]
    start_lng = first["accept_gps_lng"]
    start_lat = first["accept_gps_lat"]
    if not (120 <= start_lng <= 122 and 30 <= start_lat <= 32):
        start_lng = first["delivery_gps_lng"]
        start_lat = first["delivery_gps_lat"]
    depot = pd.DataFrame(
        {
            "node_index": [0],
            "node_type": ["start_proxy"],
            "lng": [start_lng],
            "lat": [start_lat],
        }
    )
    customers = pd.DataFrame(
        {
            "node_index": ordered["stop_index"].astype(int).to_numpy(),
            "node_type": "customer",
            "lng": ordered["delivery_gps_lng"].to_numpy(),
            "lat": ordered["delivery_gps_lat"].to_numpy(),
        }
    )
    return pd.concat([depot, customers], ignore_index=True)


def coordinate_versions(points: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw = points.copy()
    converted = points.copy()
    converted_lng, converted_lat = gcj02_to_wgs84(points["lng"], points["lat"])
    converted["lng"] = converted_lng
    converted["lat"] = converted_lat
    return {"raw": raw, "gcj02_to_wgs84": converted}


def graph_bbox(versions: dict[str, pd.DataFrame]) -> tuple[float, float, float, float]:
    all_points = pd.concat(list(versions.values()), ignore_index=True)
    return (
        float(all_points["lng"].min() - GRAPH_BUFFER_DEGREES),
        float(all_points["lat"].min() - GRAPH_BUFFER_DEGREES),
        float(all_points["lng"].max() + GRAPH_BUFFER_DEGREES),
        float(all_points["lat"].max() + GRAPH_BUFFER_DEGREES),
    )


def load_or_download_graph(
    instance_id: str,
    versions: dict[str, pd.DataFrame],
    graph_dir: Path,
) -> nx.MultiDiGraph:
    graph_dir.mkdir(parents=True, exist_ok=True)
    path = graph_dir / f"{instance_id}.graphml"
    if path.exists():
        graph = ox.load_graphml(path)
    else:
        graph = ox.graph_from_bbox(
            graph_bbox(versions),
            network_type="drive_service",
            retain_all=False,
            truncate_by_edge=True,
        )
        ox.save_graphml(graph, path)
    return ox.truncate.largest_component(graph, strongly=True)


def project_points(
    points: pd.DataFrame, target_crs: object
) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    x, y = transformer.transform(points["lng"].to_numpy(), points["lat"].to_numpy())
    return np.asarray(x), np.asarray(y)


def match_version(
    projected_graph: nx.MultiDiGraph, points: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = project_points(points, projected_graph.graph["crs"])
    _, distances = ox.distance.nearest_edges(
        projected_graph, X=x, Y=y, return_dist=True
    )
    nearest_nodes, node_distances = ox.distance.nearest_nodes(
        projected_graph, X=x, Y=y, return_dist=True
    )
    return (
        np.asarray(distances, dtype=float),
        np.asarray(nearest_nodes),
        np.asarray(node_distances, dtype=float),
    )


def build_od_matrix(
    instance_id: str,
    points: pd.DataFrame,
    graph: nx.MultiDiGraph,
    nearest_nodes: np.ndarray,
    node_connector_distances: np.ndarray,
) -> pd.DataFrame:
    rows = []
    unique_origins = np.unique(nearest_nodes)
    distances_by_origin = {
        origin: nx.single_source_dijkstra_path_length(
            graph, int(origin), weight="length"
        )
        for origin in unique_origins
    }
    for origin_position, (origin_index, origin_node) in enumerate(
        zip(points["node_index"], nearest_nodes)
    ):
        lengths = distances_by_origin[origin_node]
        for destination_position, (destination_index, destination_node) in enumerate(
            zip(points["node_index"], nearest_nodes)
        ):
            if origin_index == destination_index:
                distance = 0.0
            else:
                network_distance = lengths.get(destination_node, math.inf)
                distance = (
                    network_distance
                    + node_connector_distances[origin_position]
                    + node_connector_distances[destination_position]
                )
            rows.append(
                {
                    "instance_id": instance_id,
                    "origin_index": int(origin_index),
                    "destination_index": int(destination_index),
                    "distance_m": float(distance),
                }
            )
    return pd.DataFrame(rows)


def prepare_networks(
    instances: pd.DataFrame, graph_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    configure_osmnx()
    audit_rows = []
    matrices = []
    instance_data = {}
    pooled_match_distances: dict[str, list[np.ndarray]] = {
        "raw": [],
        "gcj02_to_wgs84": [],
    }
    for instance_id, frame in instances.groupby("instance_id", sort=True):
        points = route_points(frame)
        versions = coordinate_versions(points)
        graph = load_or_download_graph(instance_id, versions, graph_dir)
        projected = ox.project_graph(graph)
        matches = {}
        for mode, version_points in versions.items():
            edge_distances, nearest_nodes, node_distances = match_version(
                projected, version_points
            )
            matches[mode] = (edge_distances, nearest_nodes, node_distances)
            pooled_match_distances[mode].append(edge_distances)
            audit_rows.append(
                {
                    "instance_id": instance_id,
                    "coordinate_mode": mode,
                    "points": len(version_points),
                    "median_nearest_road_m": float(np.median(edge_distances)),
                    "p90_nearest_road_m": float(np.quantile(edge_distances, 0.9)),
                    "max_nearest_road_m": float(np.max(edge_distances)),
                    "median_node_connector_m": float(np.median(node_distances)),
                    "p90_node_connector_m": float(np.quantile(node_distances, 0.9)),
                    "within_50m_share": float(np.mean(edge_distances <= 50)),
                    "within_100m_share": float(np.mean(edge_distances <= 100)),
                    "graph_nodes": len(graph.nodes),
                    "graph_edges": len(graph.edges),
                }
            )
        instance_data[instance_id] = (graph, versions, matches)
    global_match_medians = {
        mode: float(np.median(np.concatenate(values)))
        for mode, values in pooled_match_distances.items()
    }
    selected_mode = min(global_match_medians, key=global_match_medians.get)
    for instance_id, (graph, versions, matches) in instance_data.items():
        chosen_points = versions[selected_mode]
        _, nearest_nodes, node_distances = matches[selected_mode]
        matrices.append(
            build_od_matrix(
                instance_id,
                chosen_points,
                graph,
                nearest_nodes,
                node_distances,
            )
        )
    audit = pd.DataFrame(audit_rows)
    matrix = pd.concat(matrices, ignore_index=True)
    unreachable = ~np.isfinite(matrix["distance_m"])
    report = {
        "osm_snapshot": OSM_SNAPSHOT,
        "network_type": "drive_service",
        "instances": int(instances["instance_id"].nunique()),
        "coordinate_mode_selection_metric": "Global median nearest-road distance across all benchmark points.",
        "coordinate_mode_global_median_nearest_road_m": global_match_medians,
        "consistent_coordinate_mode": True,
        "selected_coordinate_mode": selected_mode,
        "coordinate_interpretation": "LaDe does not declare its CRS. The selected GCJ02-to-WGS84 conversion is an empirical alignment decision and remains a sensitivity assumption.",
        "graph_connectivity": "Largest strongly connected component for round-trip vehicle routing.",
        "od_pairs": int(len(matrix)),
        "unreachable_od_pairs": int(unreachable.sum()),
        "od_scope": "Start proxy and customer nodes only. Charging stations are added in stage F.",
        "distance_unit": "metres",
        "connector_rule": "Non-diagonal OD distance includes origin and destination straight-line connectors to their matched road nodes.",
    }
    return audit, matrix, report


def main() -> None:
    args = parse_args()
    instances = pd.read_parquet(args.instances)
    audit, matrix, report = prepare_networks(instances, args.graph_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(args.output_dir / "benchmark_od_matrix.parquet", index=False)
    audit.to_csv(args.result_dir / "coordinate_match_audit.csv", index=False)
    (args.result_dir / "road_network_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
