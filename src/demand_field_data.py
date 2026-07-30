"""第二版 MLP 与 NBFNet 共用的无路径监督数据接口。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .graph_io import load_porto_graph
from .region_candidates import chronological_prefix, load_candidate_manifest
from .region_labels import chronological_window
from .workloads import load_porto_queries


DEMAND_FIELD_DATASET_SCHEMA = "aic.gnn_v2.demand_field_dataset.v2"
SPLIT_NAMES = ("train", "validation", "holdout")


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    source: int
    target: int
    length_m: float
    road_type: str


@dataclass(slots=True)
class DemandFieldDataset:
    manifest: dict
    node_ids: np.ndarray
    node_features: np.ndarray
    edge_source: np.ndarray
    edge_target: np.ndarray
    edge_length: np.ndarray
    edge_type: np.ndarray
    history_query_ids: np.ndarray
    history_query_prototype: np.ndarray
    prototype_weight: np.ndarray
    prototype_origin_offsets: np.ndarray
    prototype_origin_nodes: np.ndarray
    prototype_origin_weights: np.ndarray
    prototype_destination_offsets: np.ndarray
    prototype_destination_nodes: np.ndarray
    prototype_destination_weights: np.ndarray
    region_ids: np.ndarray
    region_nodes: np.ndarray
    boundary_offsets: np.ndarray
    boundary_nodes: np.ndarray
    region_features: np.ndarray
    labels: np.ndarray
    split: np.ndarray
    selection_method: np.ndarray

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["node_feature_names"])

    @property
    def region_feature_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["region_feature_names"])

    def split_mask(self, name: str) -> np.ndarray:
        try:
            split_id = SPLIT_NAMES.index(name)
        except ValueError as error:
            raise ValueError(f"unknown split: {name}") from error
        return self.split == split_id


def build_demand_field_dataset(
    *,
    node_csv: Path,
    edge_csv: Path,
    query_csv: Path,
    candidate_manifest_path: Path,
    label_csv: Path,
    label_manifest_path: Path,
    history_fraction: float = 0.35,
    label_start_fraction: float = 0.35,
    label_end_fraction: float = 0.70,
    split_seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    overlap_group_threshold: float = 0.50,
    prototype_count: int = 128,
    prototype_seed: int = 42,
) -> DemandFieldDataset:
    """构建只读取历史 OD、静态道路图和后续收益标签的数据集。"""

    _validate_split_fractions(train_fraction, validation_fraction)
    graph = load_porto_graph(node_csv, edge_csv)
    edge_records = load_edge_records(edge_csv)
    queries = load_porto_queries(query_csv)
    history_queries = chronological_prefix(queries, history_fraction)
    label_window = chronological_window(
        queries,
        label_start_fraction,
        label_end_fraction,
    )
    candidate_manifest, regions = load_candidate_manifest(candidate_manifest_path)
    labels = load_region_labels(label_csv)
    label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
    _validate_sources(
        graph_node_count=graph.node_count,
        graph_edge_count=graph.edge_count,
        history_queries=history_queries,
        label_window=label_window,
        candidate_manifest=candidate_manifest,
        regions=regions,
        labels=labels,
        label_manifest=label_manifest,
        history_fraction=history_fraction,
        label_start_fraction=label_start_fraction,
        label_end_fraction=label_end_fraction,
    )

    node_ids = np.asarray(sorted(graph.adjacency), dtype=np.int64)
    node_to_index = {int(node): index for index, node in enumerate(node_ids)}
    road_types = tuple(sorted({edge.road_type for edge in edge_records}))
    node_features, node_feature_names, normalizers = build_node_features(
        node_ids=node_ids,
        node_to_index=node_to_index,
        edge_records=edge_records,
        history_queries=history_queries,
        road_types=road_types,
    )
    prototypes = build_demand_prototypes(
        history_queries=history_queries,
        coordinates=graph.coordinates,
        node_to_index=node_to_index,
        prototype_count=prototype_count,
        seed=prototype_seed,
    )
    edge_source, edge_target, edge_length, edge_type = build_edge_arrays(
        edge_records,
        node_to_index,
        road_types,
        normalizers["edge_length_log_max"],
    )

    ordered_regions = sorted(regions, key=lambda region: region.region_id)
    region_ids = np.asarray([region.region_id for region in ordered_regions], dtype=np.int32)
    region_nodes = np.asarray(
        [
            [node_to_index[node] for node in sorted(region.nodes)]
            for region in ordered_regions
        ],
        dtype=np.int32,
    )
    boundary_lists = [
        np.asarray(
            [node_to_index[node] for node in sorted(region.boundary_nodes)],
            dtype=np.int32,
        )
        for region in ordered_regions
    ]
    boundary_offsets = np.zeros(len(boundary_lists) + 1, dtype=np.int32)
    for index, boundary in enumerate(boundary_lists, start=1):
        boundary_offsets[index] = boundary_offsets[index - 1] + len(boundary)
    boundary_nodes = np.concatenate(boundary_lists).astype(np.int32, copy=False)
    region_features = pool_region_features(node_features, region_nodes)
    region_feature_names = tuple(
        [f"mean_{name}" for name in node_feature_names]
        + [f"max_{name}" for name in node_feature_names]
    )
    label_values = np.asarray(
        [labels[int(region_id)]["avg_workload_gain"] for region_id in region_ids],
        dtype=np.float32,
    )
    selection_method = np.asarray(
        [region.selection_method for region in ordered_regions],
        dtype="U32",
    )
    split, split_groups = overlap_grouped_candidate_split(
        region_nodes,
        selection_method,
        seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        overlap_threshold=overlap_group_threshold,
    )
    manifest = {
        "schema": DEMAND_FIELD_DATASET_SCHEMA,
        "candidate_sha256": candidate_manifest["candidate_sha256"],
        "label_schema": label_manifest.get("schema"),
        "source_files": {
            "node_csv": str(node_csv.resolve()),
            "edge_csv": str(edge_csv.resolve()),
            "query_csv": str(query_csv.resolve()),
            "candidate_manifest": str(candidate_manifest_path.resolve()),
            "label_csv": str(label_csv.resolve()),
            "label_manifest": str(label_manifest_path.resolve()),
        },
        "source_sha256": {
            "candidate_manifest": _sha256(candidate_manifest_path),
            "label_csv": _sha256(label_csv),
            "label_manifest": _sha256(label_manifest_path),
        },
        "history_window": _window_metadata(history_queries, 0.0, history_fraction),
        "label_window": _window_metadata(
            label_window,
            label_start_fraction,
            label_end_fraction,
        ),
        "formal_label_query_count": len(label_manifest["query_ids"]),
        "formal_label_query_sample_seed": label_manifest.get("query_sample_seed"),
        "graph": {
            "node_count": int(node_ids.size),
            "edge_count": int(edge_source.size),
            "road_type_count": len(road_types),
        },
        "road_types": list(road_types),
        "node_feature_names": list(node_feature_names),
        "region_feature_names": list(region_feature_names),
        "normalizers": normalizers,
        "candidate_count": int(region_ids.size),
        "region_size": int(region_nodes.shape[1]),
        "label_name": "avg_workload_gain",
        "model_input_policy": {
            "uses_history_od_only": True,
            "uses_static_road_graph": True,
            "uses_path_supervision": False,
            "uses_label_window_as_input": False,
            "uses_cost_or_query_runtime_features": False,
            "uses_node_coordinates": False,
            "coordinates_used_for_prototype_grouping_only": True,
            "selection_method_is_model_input": False,
        },
        "demand_prototypes": {
            "count": int(prototypes["prototype_weight"].size),
            "seed": prototype_seed,
            "method": "weighted_kmeans_on_normalized_origin_destination_coordinates",
            "uses_history_od_only": True,
            "uses_static_coordinates": True,
            "uses_shortest_paths": False,
            "origin_membership_count": int(prototypes["prototype_origin_nodes"].size),
            "destination_membership_count": int(
                prototypes["prototype_destination_nodes"].size
            ),
        },
        "split": {
            "axis": "candidate",
            "strategy": "jaccard_overlap_connected_components",
            "seed": split_seed,
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "holdout_fraction": 1.0 - train_fraction - validation_fraction,
            "counts": {
                name: int(np.sum(split == split_id))
                for split_id, name in enumerate(SPLIT_NAMES)
            },
            "overlap_group_threshold": overlap_group_threshold,
            "overlap_group_count": split_groups["group_count"],
            "largest_overlap_group": split_groups["largest_group"],
            "cross_split_max_jaccard_upper_bound": overlap_group_threshold,
            "guarantee": (
                "Every candidate pair with Jaccard similarity greater than or "
                "equal to the threshold belongs to the same split."
            ),
            "warning": (
                "Candidate holdout measures region generalization within H→Y; "
                "it is not the frozen future temporal test."
            ),
        },
    }
    return DemandFieldDataset(
        manifest=manifest,
        node_ids=node_ids,
        node_features=node_features,
        edge_source=edge_source,
        edge_target=edge_target,
        edge_length=edge_length,
        edge_type=edge_type,
        history_query_ids=prototypes["history_query_ids"],
        history_query_prototype=prototypes["history_query_prototype"],
        prototype_weight=prototypes["prototype_weight"],
        prototype_origin_offsets=prototypes["prototype_origin_offsets"],
        prototype_origin_nodes=prototypes["prototype_origin_nodes"],
        prototype_origin_weights=prototypes["prototype_origin_weights"],
        prototype_destination_offsets=prototypes["prototype_destination_offsets"],
        prototype_destination_nodes=prototypes["prototype_destination_nodes"],
        prototype_destination_weights=prototypes["prototype_destination_weights"],
        region_ids=region_ids,
        region_nodes=region_nodes,
        boundary_offsets=boundary_offsets,
        boundary_nodes=boundary_nodes,
        region_features=region_features,
        labels=label_values,
        split=split,
        selection_method=selection_method,
    )


def save_demand_field_dataset(
    dataset: DemandFieldDataset,
    npz_path: Path,
    manifest_path: Path,
) -> dict:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        node_ids=dataset.node_ids,
        node_features=dataset.node_features,
        edge_source=dataset.edge_source,
        edge_target=dataset.edge_target,
        edge_length=dataset.edge_length,
        edge_type=dataset.edge_type,
        history_query_ids=dataset.history_query_ids,
        history_query_prototype=dataset.history_query_prototype,
        prototype_weight=dataset.prototype_weight,
        prototype_origin_offsets=dataset.prototype_origin_offsets,
        prototype_origin_nodes=dataset.prototype_origin_nodes,
        prototype_origin_weights=dataset.prototype_origin_weights,
        prototype_destination_offsets=dataset.prototype_destination_offsets,
        prototype_destination_nodes=dataset.prototype_destination_nodes,
        prototype_destination_weights=dataset.prototype_destination_weights,
        region_ids=dataset.region_ids,
        region_nodes=dataset.region_nodes,
        boundary_offsets=dataset.boundary_offsets,
        boundary_nodes=dataset.boundary_nodes,
        region_features=dataset.region_features,
        labels=dataset.labels,
        split=dataset.split,
        selection_method=dataset.selection_method,
    )
    manifest = dict(dataset.manifest)
    manifest["dataset_file"] = str(npz_path.resolve())
    manifest["dataset_sha256"] = _dataset_digest(dataset)
    manifest["dataset_file_sha256"] = _sha256(npz_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dataset.manifest = manifest
    return manifest


def load_demand_field_dataset(
    npz_path: Path,
    manifest_path: Path,
) -> DemandFieldDataset:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != DEMAND_FIELD_DATASET_SCHEMA:
        raise ValueError(f"unsupported dataset schema: {manifest.get('schema')}")
    if manifest.get("dataset_file_sha256") != _sha256(npz_path):
        raise ValueError("demand-field dataset digest mismatch")
    with np.load(npz_path, allow_pickle=False) as arrays:
        dataset = DemandFieldDataset(
            manifest=manifest,
            node_ids=arrays["node_ids"],
            node_features=arrays["node_features"],
            edge_source=arrays["edge_source"],
            edge_target=arrays["edge_target"],
            edge_length=arrays["edge_length"],
            edge_type=arrays["edge_type"],
            history_query_ids=arrays["history_query_ids"],
            history_query_prototype=arrays["history_query_prototype"],
            prototype_weight=arrays["prototype_weight"],
            prototype_origin_offsets=arrays["prototype_origin_offsets"],
            prototype_origin_nodes=arrays["prototype_origin_nodes"],
            prototype_origin_weights=arrays["prototype_origin_weights"],
            prototype_destination_offsets=arrays["prototype_destination_offsets"],
            prototype_destination_nodes=arrays["prototype_destination_nodes"],
            prototype_destination_weights=arrays["prototype_destination_weights"],
            region_ids=arrays["region_ids"],
            region_nodes=arrays["region_nodes"],
            boundary_offsets=arrays["boundary_offsets"],
            boundary_nodes=arrays["boundary_nodes"],
            region_features=arrays["region_features"],
            labels=arrays["labels"],
            split=arrays["split"],
            selection_method=arrays["selection_method"],
        )
    _validate_loaded_dataset(dataset)
    if manifest.get("dataset_sha256") != _dataset_digest(dataset):
        raise ValueError("demand-field dataset content identity mismatch")
    return dataset


def load_edge_records(path: Path) -> list[EdgeRecord]:
    records: list[EdgeRecord] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            records.append(
                EdgeRecord(
                    source=int(row["source"]),
                    target=int(row["target"]),
                    length_m=float(row["length_m"]),
                    road_type=row.get("highway", "unknown") or "unknown",
                )
            )
    return records


def load_region_labels(path: Path) -> dict[int, dict[str, float]]:
    labels: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            region_id = int(row["region_id"])
            if region_id in labels:
                raise ValueError(f"duplicate region label: {region_id}")
            labels[region_id] = {
                "avg_workload_gain": float(row["avg_workload_gain"]),
                "label_query_count": float(row["label_query_count"]),
                "correctness_rate": float(row["correctness_rate"]),
            }
    return labels


def build_node_features(
    *,
    node_ids: np.ndarray,
    node_to_index: dict[int, int],
    edge_records: list[EdgeRecord],
    history_queries,
    road_types: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...], dict[str, float]]:
    node_count = len(node_ids)
    road_type_to_index = {road_type: index for index, road_type in enumerate(road_types)}
    origin_count = np.zeros(node_count, dtype=np.float64)
    destination_count = np.zeros(node_count, dtype=np.float64)
    for query in history_queries:
        origin_count[node_to_index[query.origin]] += query.count
        destination_count[node_to_index[query.destination]] += query.count

    in_degree = np.zeros(node_count, dtype=np.float64)
    out_degree = np.zeros(node_count, dtype=np.float64)
    in_length_sum = np.zeros(node_count, dtype=np.float64)
    out_length_sum = np.zeros(node_count, dtype=np.float64)
    in_road_count = np.zeros((node_count, len(road_types)), dtype=np.float32)
    out_road_count = np.zeros((node_count, len(road_types)), dtype=np.float32)
    max_edge_length = 0.0
    for edge in edge_records:
        source = node_to_index[edge.source]
        target = node_to_index[edge.target]
        road_type = road_type_to_index[edge.road_type]
        out_degree[source] += 1.0
        in_degree[target] += 1.0
        out_length_sum[source] += edge.length_m
        in_length_sum[target] += edge.length_m
        out_road_count[source, road_type] += 1.0
        in_road_count[target, road_type] += 1.0
        max_edge_length = max(max_edge_length, edge.length_m)

    max_origin_log = max(float(np.log1p(origin_count).max()), 1.0)
    max_destination_log = max(float(np.log1p(destination_count).max()), 1.0)
    max_in_degree = max(float(in_degree.max()), 1.0)
    max_out_degree = max(float(out_degree.max()), 1.0)
    edge_length_log_max = max(math.log1p(max_edge_length), 1.0)
    mean_in_length = np.divide(
        in_length_sum,
        in_degree,
        out=np.zeros_like(in_length_sum),
        where=in_degree > 0,
    )
    mean_out_length = np.divide(
        out_length_sum,
        out_degree,
        out=np.zeros_like(out_length_sum),
        where=out_degree > 0,
    )
    in_road_share = np.divide(
        in_road_count,
        in_degree[:, None],
        out=np.zeros_like(in_road_count),
        where=in_degree[:, None] > 0,
    )
    out_road_share = np.divide(
        out_road_count,
        out_degree[:, None],
        out=np.zeros_like(out_road_count),
        where=out_degree[:, None] > 0,
    )
    scalar_features = np.column_stack(
        [
            np.log1p(origin_count) / max_origin_log,
            np.log1p(destination_count) / max_destination_log,
            in_degree / max_in_degree,
            out_degree / max_out_degree,
            np.log1p(mean_in_length) / edge_length_log_max,
            np.log1p(mean_out_length) / edge_length_log_max,
        ]
    ).astype(np.float32)
    features = np.concatenate(
        [scalar_features, in_road_share, out_road_share],
        axis=1,
    ).astype(np.float32, copy=False)
    feature_names = (
        "history_origin_count",
        "history_destination_count",
        "in_degree",
        "out_degree",
        "mean_in_edge_length",
        "mean_out_edge_length",
        *(f"in_road_type_{road_type}" for road_type in road_types),
        *(f"out_road_type_{road_type}" for road_type in road_types),
    )
    normalizers = {
        "origin_count_log_max": max_origin_log,
        "destination_count_log_max": max_destination_log,
        "in_degree_max": max_in_degree,
        "out_degree_max": max_out_degree,
        "edge_length_log_max": edge_length_log_max,
    }
    return features, feature_names, normalizers


def build_edge_arrays(
    edge_records: list[EdgeRecord],
    node_to_index: dict[int, int],
    road_types: tuple[str, ...],
    edge_length_log_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    road_type_to_index = {road_type: index for index, road_type in enumerate(road_types)}
    edge_source = np.fromiter(
        (node_to_index[edge.source] for edge in edge_records),
        dtype=np.int32,
        count=len(edge_records),
    )
    edge_target = np.fromiter(
        (node_to_index[edge.target] for edge in edge_records),
        dtype=np.int32,
        count=len(edge_records),
    )
    edge_length = np.fromiter(
        (math.log1p(edge.length_m) / edge_length_log_max for edge in edge_records),
        dtype=np.float32,
        count=len(edge_records),
    )
    edge_type = np.fromiter(
        (road_type_to_index[edge.road_type] for edge in edge_records),
        dtype=np.int16,
        count=len(edge_records),
    )
    return edge_source, edge_target, edge_length, edge_type


def build_demand_prototypes(
    *,
    history_queries,
    coordinates: dict[int, tuple[float, float]],
    node_to_index: dict[int, int],
    prototype_count: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """只用历史 OD 端点和静态坐标构造带权起终点集合。"""

    if not history_queries:
        raise ValueError("cannot build demand prototypes from an empty history window")
    if not 0 < prototype_count <= len(history_queries):
        raise ValueError("prototype_count must be between 1 and history query count")
    if prototype_count > np.iinfo(np.int16).max:
        raise ValueError("prototype_count exceeds the stored int16 prototype id range")
    graph_coordinates = np.asarray(list(coordinates.values()), dtype=np.float64)
    minimum = graph_coordinates.min(axis=0)
    span = np.maximum(graph_coordinates.max(axis=0) - minimum, 1e-12)
    points = np.empty((len(history_queries), 4), dtype=np.float64)
    query_weights = np.empty(len(history_queries), dtype=np.float64)
    for index, query in enumerate(history_queries):
        origin = (np.asarray(coordinates[query.origin]) - minimum) / span
        destination = (np.asarray(coordinates[query.destination]) - minimum) / span
        points[index, :2] = origin
        points[index, 2:] = destination
        query_weights[index] = query.count
    assignment = _weighted_kmeans(points, query_weights, prototype_count, seed)
    if np.any(np.bincount(assignment, minlength=prototype_count) == 0):
        raise RuntimeError("demand prototype clustering produced an empty prototype")

    origin_nodes_by_prototype: list[np.ndarray] = []
    origin_weights_by_prototype: list[np.ndarray] = []
    destination_nodes_by_prototype: list[np.ndarray] = []
    destination_weights_by_prototype: list[np.ndarray] = []
    prototype_weight = np.zeros(prototype_count, dtype=np.float64)
    for prototype_id in range(prototype_count):
        origin_counts: Counter[int] = Counter()
        destination_counts: Counter[int] = Counter()
        for query_index in np.flatnonzero(assignment == prototype_id):
            query = history_queries[int(query_index)]
            origin_counts[node_to_index[query.origin]] += query.count
            destination_counts[node_to_index[query.destination]] += query.count
            prototype_weight[prototype_id] += query.count
        origin_nodes = np.asarray(sorted(origin_counts), dtype=np.int32)
        destination_nodes = np.asarray(sorted(destination_counts), dtype=np.int32)
        origin_values = np.asarray(
            [origin_counts[int(node)] for node in origin_nodes],
            dtype=np.float64,
        )
        destination_values = np.asarray(
            [destination_counts[int(node)] for node in destination_nodes],
            dtype=np.float64,
        )
        origin_nodes_by_prototype.append(origin_nodes)
        destination_nodes_by_prototype.append(destination_nodes)
        origin_weights_by_prototype.append(
            (origin_values / origin_values.sum()).astype(np.float32)
        )
        destination_weights_by_prototype.append(
            (destination_values / destination_values.sum()).astype(np.float32)
        )

    origin_offsets = _ragged_offsets(origin_nodes_by_prototype)
    destination_offsets = _ragged_offsets(destination_nodes_by_prototype)
    return {
        "history_query_ids": np.asarray(
            [query.query_id for query in history_queries],
            dtype=np.int64,
        ),
        "history_query_prototype": assignment.astype(np.int16),
        "prototype_weight": (prototype_weight / prototype_weight.sum()).astype(
            np.float32
        ),
        "prototype_origin_offsets": origin_offsets,
        "prototype_origin_nodes": np.concatenate(origin_nodes_by_prototype).astype(
            np.int32,
            copy=False,
        ),
        "prototype_origin_weights": np.concatenate(origin_weights_by_prototype).astype(
            np.float32,
            copy=False,
        ),
        "prototype_destination_offsets": destination_offsets,
        "prototype_destination_nodes": np.concatenate(
            destination_nodes_by_prototype
        ).astype(np.int32, copy=False),
        "prototype_destination_weights": np.concatenate(
            destination_weights_by_prototype
        ).astype(np.float32, copy=False),
    }


def pool_region_features(
    node_features: np.ndarray,
    region_nodes: np.ndarray,
) -> np.ndarray:
    selected = node_features[region_nodes]
    return np.concatenate([selected.mean(axis=1), selected.max(axis=1)], axis=1).astype(
        np.float32,
        copy=False,
    )


def stratified_candidate_split(
    selection_method: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> np.ndarray:
    _validate_split_fractions(train_fraction, validation_fraction)
    split = np.full(len(selection_method), -1, dtype=np.int8)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, method in enumerate(selection_method.tolist()):
        groups[str(method)].append(index)
    rng = random.Random(seed)
    for method in sorted(groups):
        indices = groups[method]
        rng.shuffle(indices)
        train_end = int(len(indices) * train_fraction)
        validation_end = train_end + int(len(indices) * validation_fraction)
        split[indices[:train_end]] = 0
        split[indices[train_end:validation_end]] = 1
        split[indices[validation_end:]] = 2
    if np.any(split < 0):
        raise RuntimeError("candidate split left unassigned rows")
    return split


def overlap_grouped_candidate_split(
    region_nodes: np.ndarray,
    selection_method: np.ndarray,
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    overlap_threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Keep strongly overlapping candidate components inside one split."""

    _validate_split_fractions(train_fraction, validation_fraction)
    if region_nodes.ndim != 2 or len(region_nodes) != len(selection_method):
        raise ValueError("region_nodes and selection_method must align by candidate")
    if not 0.0 < overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be in (0, 1]")
    candidate_count = len(region_nodes)
    parent = list(range(candidate_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    memberships: dict[int, list[int]] = defaultdict(list)
    candidate_sizes = np.empty(candidate_count, dtype=np.int32)
    for candidate_index, nodes in enumerate(region_nodes):
        unique_nodes = set(int(node) for node in nodes)
        candidate_sizes[candidate_index] = len(unique_nodes)
        for node in unique_nodes:
            memberships[node].append(candidate_index)
    pair_intersections: Counter[tuple[int, int]] = Counter()
    for candidate_indices in memberships.values():
        for left_position, left in enumerate(candidate_indices):
            for right in candidate_indices[left_position + 1 :]:
                pair_intersections[(left, right)] += 1
    for (left, right), intersection in pair_intersections.items():
        union_size = int(candidate_sizes[left] + candidate_sizes[right] - intersection)
        if intersection / union_size >= overlap_threshold:
            union(left, right)

    components: dict[int, list[int]] = defaultdict(list)
    for candidate_index in range(candidate_count):
        components[find(candidate_index)].append(candidate_index)
    component_list = list(components.values())
    rng = random.Random(seed)
    rng.shuffle(component_list)
    component_list.sort(key=len, reverse=True)
    split = np.full(candidate_count, -1, dtype=np.int8)
    target = np.asarray(
        [
            candidate_count * train_fraction,
            candidate_count * validation_fraction,
            candidate_count * (1.0 - train_fraction - validation_fraction),
        ],
        dtype=np.float64,
    )
    counts = np.zeros(3, dtype=np.int32)
    for component in component_list:
        remaining = target - counts
        split_id = int(np.argmax(remaining))
        split[np.asarray(component, dtype=np.int32)] = split_id
        counts[split_id] += len(component)
    if np.any(split < 0) or np.any(counts == 0):
        raise RuntimeError("overlap-grouped split did not populate every split")
    return split, {
        "group_count": len(component_list),
        "largest_group": max(map(len, component_list), default=0),
    }


def _validate_sources(
    *,
    graph_node_count: int,
    graph_edge_count: int,
    history_queries,
    label_window,
    candidate_manifest: dict,
    regions,
    labels: dict[int, dict[str, float]],
    label_manifest: dict,
    history_fraction: float,
    label_start_fraction: float,
    label_end_fraction: float,
) -> None:
    errors: list[str] = []
    if history_fraction > label_start_fraction:
        errors.append("history window overlaps the label window")
    if candidate_manifest.get("graph") != {
        "node_count": graph_node_count,
        "edge_count": graph_edge_count,
    }:
        errors.append("candidate manifest graph identity does not match source CSV")
    if candidate_manifest.get("candidate_sha256") != label_manifest.get(
        "candidate_sha256"
    ):
        errors.append("label manifest candidate digest does not match candidates")
    region_ids = {region.region_id for region in regions}
    if set(labels) != region_ids:
        errors.append("label CSV region ids do not match the candidate pool")
    target_ids = {int(value) for value in label_manifest.get("target_region_ids", [])}
    completed_ids = {
        int(value) for value in label_manifest.get("completed_region_ids", [])
    }
    if label_manifest.get("status") != "complete":
        errors.append("formal label manifest is not complete")
    if target_ids != region_ids or completed_ids != region_ids:
        errors.append("label manifest does not complete every candidate")
    if label_manifest.get("target_region_count") != len(regions):
        errors.append("label manifest target count is inconsistent")
    if label_manifest.get("completed_region_count") != len(regions):
        errors.append("label manifest completed count is inconsistent")
    query_ids = [int(value) for value in label_manifest.get("query_ids", [])]
    label_window_ids = {query.query_id for query in label_window}
    history_ids = {query.query_id for query in history_queries}
    if not query_ids or len(query_ids) != len(set(query_ids)):
        errors.append("label manifest query ids are empty or duplicated")
    if not set(query_ids) <= label_window_ids:
        errors.append("formal label queries are not contained in the label window")
    if history_ids & set(query_ids):
        errors.append("history and formal label queries overlap")
    if label_manifest.get("label_start_fraction") != label_start_fraction:
        errors.append("label start fraction does not match")
    if label_manifest.get("label_end_fraction") != label_end_fraction:
        errors.append("label end fraction does not match")
    label_query_counts = {int(row["label_query_count"]) for row in labels.values()}
    if label_query_counts != {len(query_ids)}:
        errors.append("label CSV query counts do not match the manifest")
    if min(row["correctness_rate"] for row in labels.values()) != 1.0:
        errors.append("label CSV contains an inexact candidate")
    if errors:
        raise ValueError("invalid demand-field sources: " + "; ".join(errors))


def _validate_loaded_dataset(dataset: DemandFieldDataset) -> None:
    candidate_count = int(dataset.manifest["candidate_count"])
    node_count = int(dataset.manifest["graph"]["node_count"])
    edge_count = int(dataset.manifest["graph"]["edge_count"])
    if dataset.node_features.shape[0] != node_count:
        raise ValueError("node feature count does not match dataset manifest")
    if dataset.edge_source.size != edge_count or dataset.edge_target.size != edge_count:
        raise ValueError("edge array count does not match dataset manifest")
    if dataset.region_features.shape[0] != candidate_count:
        raise ValueError("region feature count does not match dataset manifest")
    if dataset.labels.size != candidate_count or dataset.split.size != candidate_count:
        raise ValueError("label or split count does not match dataset manifest")
    prototype_count = int(dataset.manifest["demand_prototypes"]["count"])
    if dataset.prototype_weight.size != prototype_count:
        raise ValueError("prototype count does not match dataset manifest")
    if dataset.prototype_origin_offsets.size != prototype_count + 1:
        raise ValueError("origin prototype offsets are inconsistent")
    if dataset.prototype_destination_offsets.size != prototype_count + 1:
        raise ValueError("destination prototype offsets are inconsistent")
    if dataset.history_query_ids.size != dataset.history_query_prototype.size:
        raise ValueError("history query prototype mapping is inconsistent")
    if not np.isclose(float(dataset.prototype_weight.sum()), 1.0):
        raise ValueError("prototype weights do not sum to one")
    if not np.isfinite(dataset.node_features).all():
        raise ValueError("node features contain non-finite values")
    if not np.isfinite(dataset.region_features).all():
        raise ValueError("region features contain non-finite values")


def _validate_split_fractions(train_fraction: float, validation_fraction: float) -> None:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must sum to less than 1")


def _weighted_kmeans(
    points: np.ndarray,
    weights: np.ndarray,
    cluster_count: int,
    seed: int,
    *,
    max_iterations: int = 40,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = np.empty((cluster_count, points.shape[1]), dtype=np.float64)
    first = int(rng.choice(len(points), p=weights / weights.sum()))
    centers[0] = points[first]
    minimum_distance = np.sum((points - centers[0]) ** 2, axis=1)
    for center_index in range(1, cluster_count):
        probability = minimum_distance * weights
        if probability.sum() <= 0.0:
            chosen = int(rng.integers(0, len(points)))
        else:
            chosen = int(rng.choice(len(points), p=probability / probability.sum()))
        centers[center_index] = points[chosen]
        distance = np.sum((points - centers[center_index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)

    assignment = np.full(len(points), -1, dtype=np.int32)
    for _ in range(max_iterations):
        updated_assignment = _nearest_centers(points, centers)
        if np.array_equal(updated_assignment, assignment):
            break
        assignment = updated_assignment
        nearest_distance = np.sum((points - centers[assignment]) ** 2, axis=1)
        for center_index in range(cluster_count):
            members = assignment == center_index
            if np.any(members):
                centers[center_index] = np.average(
                    points[members],
                    axis=0,
                    weights=weights[members],
                )
            else:
                farthest = int(np.argmax(nearest_distance * weights))
                centers[center_index] = points[farthest]
                assignment[farthest] = center_index
                nearest_distance[farthest] = 0.0
    return assignment


def _nearest_centers(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    assignment = np.empty(len(points), dtype=np.int32)
    chunk_size = 2048
    for start in range(0, len(points), chunk_size):
        chunk = points[start : start + chunk_size]
        distance = np.sum(
            (chunk[:, None, :] - centers[None, :, :]) ** 2,
            axis=2,
        )
        assignment[start : start + len(chunk)] = np.argmin(distance, axis=1)
    return assignment


def _ragged_offsets(values: list[np.ndarray]) -> np.ndarray:
    offsets = np.zeros(len(values) + 1, dtype=np.int32)
    for index, value in enumerate(values, start=1):
        offsets[index] = offsets[index - 1] + len(value)
    return offsets


def _window_metadata(queries, start_fraction: float, end_fraction: float) -> dict:
    return {
        "start_fraction": start_fraction,
        "end_fraction": end_fraction,
        "query_count": len(queries),
        "first_query_id": queries[0].query_id if queries else None,
        "last_query_id": queries[-1].query_id if queries else None,
        "first_timestamp": queries[0].timestamp if queries else None,
        "last_timestamp": queries[-1].timestamp if queries else None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_digest(dataset: DemandFieldDataset) -> str:
    digest = hashlib.sha256()
    excluded_manifest_keys = {
        "dataset_file",
        "dataset_sha256",
        "dataset_file_sha256",
        "source_files",
    }
    identity_manifest = {
        key: value
        for key, value in dataset.manifest.items()
        if key not in excluded_manifest_keys
    }
    digest.update(
        json.dumps(
            identity_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    array_names = (
        "node_ids",
        "node_features",
        "edge_source",
        "edge_target",
        "edge_length",
        "edge_type",
        "history_query_ids",
        "history_query_prototype",
        "prototype_weight",
        "prototype_origin_offsets",
        "prototype_origin_nodes",
        "prototype_origin_weights",
        "prototype_destination_offsets",
        "prototype_destination_nodes",
        "prototype_destination_weights",
        "region_ids",
        "region_nodes",
        "boundary_offsets",
        "boundary_nodes",
        "region_features",
        "labels",
        "split",
        "selection_method",
    )
    for name in array_names:
        value = np.ascontiguousarray(getattr(dataset, name))
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()
