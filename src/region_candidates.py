"""为第二版需求场实验构建和持久化固定候选区域池。"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .graph_types import NodeId, Query, WeightedDiGraph
from .regions import Region, find_boundary_nodes, grow_bfs_region


CANDIDATE_MANIFEST_SCHEMA = "aic.gnn_v2.region_candidates.v3"


@dataclass(frozen=True, slots=True)
class CandidatePoolConfig:
    candidate_count: int = 1_200
    region_size: int = 512
    seed: int = 42
    overlap_threshold: float = 0.80

    def validate(self) -> None:
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if self.region_size < 2:
            raise ValueError("region_size must be at least 2")
        if not 0.0 <= self.overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be between 0 and 1")


def build_fixed_candidate_pool(
    graph: WeightedDiGraph,
    config: CandidatePoolConfig = CandidatePoolConfig(),
) -> list[Region]:
    """只用静态路网构建全图固定随机候选池。"""

    config.validate()
    if not graph.adjacency:
        return []

    random_seeds = iter(_fixed_random_seeds(graph, config.seed))
    accepted: list[Region] = []
    node_to_region_ids: dict[NodeId, list[int]] = defaultdict(list)
    exact_signatures: set[frozenset[NodeId]] = set()

    while len(accepted) < config.candidate_count:
        try:
            seed_node = next(random_seeds)
        except StopIteration:
            break

        nodes = grow_bfs_region(graph, seed_node, config.region_size)
        if (
            len(nodes) == config.region_size
            and nodes not in exact_signatures
            and not _has_high_overlap(
                nodes,
                accepted,
                node_to_region_ids,
                config.overlap_threshold,
            )
        ):
            boundary_nodes = find_boundary_nodes(graph, nodes)
            if len(boundary_nodes) >= 2:
                region_id = len(accepted)
                accepted.append(
                    Region(
                        region_id=region_id,
                        nodes=nodes,
                        boundary_nodes=boundary_nodes,
                        seed_node=seed_node,
                        selection_method="fixed_random_bfs",
                    )
                )
                exact_signatures.add(nodes)
                for node in nodes:
                    node_to_region_ids[node].append(region_id)
    if len(accepted) < config.candidate_count:
        raise RuntimeError(
            "unable to build requested candidate pool: "
            f"requested={config.candidate_count}, built={len(accepted)}"
        )
    return accepted


def write_candidate_manifest(
    path: Path,
    regions: list[Region],
    config: CandidatePoolConfig,
    *,
    graph_node_count: int,
    graph_edge_count: int,
    source_files: Mapping[str, str] | None = None,
) -> dict:
    """把完整候选节点清单写入稳定 JSON，并返回清单对象。"""

    config.validate()
    candidates = [
        {
            "region_id": region.region_id,
            "seed_node": region.seed_node,
            "selection_method": region.selection_method,
            "node_count": region.node_count,
            "boundary_count": region.boundary_count,
            "nodes": sorted(region.nodes),
            "boundary_nodes": sorted(region.boundary_nodes),
        }
        for region in sorted(regions, key=lambda item: item.region_id)
    ]
    candidate_sha256 = _candidate_digest(candidates)
    manifest = {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "config": {
            "candidate_count": config.candidate_count,
            "region_size": config.region_size,
            "seed": config.seed,
            "overlap_threshold": config.overlap_threshold,
            "seed_policy": "uniform_random_over_road_nodes",
            "uses_history_od": False,
            "uses_static_graph_only": True,
        },
        "graph": {
            "node_count": graph_node_count,
            "edge_count": graph_edge_count,
        },
        "source_files": dict(sorted((source_files or {}).items())),
        "candidate_sha256": candidate_sha256,
        "selection_method_counts": dict(
            sorted(Counter(region.selection_method for region in regions).items())
        ),
        "candidates": candidates,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_candidate_manifest(path: Path) -> tuple[dict, list[Region]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CANDIDATE_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported candidate manifest schema: {manifest.get('schema')}")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidate manifest has no candidates list")
    if _candidate_digest(candidates) != manifest.get("candidate_sha256"):
        raise ValueError("candidate manifest digest mismatch")

    regions = [
        Region(
            region_id=int(item["region_id"]),
            nodes=frozenset(int(node) for node in item["nodes"]),
            boundary_nodes=frozenset(int(node) for node in item["boundary_nodes"]),
            seed_node=int(item["seed_node"]),
            selection_method=str(item["selection_method"]),
        )
        for item in candidates
    ]
    return manifest, regions


def chronological_prefix(queries: list[Query], fraction: float) -> list[Query]:
    """取严格按时间排序后的前缀，用于只依赖历史的候选生成。"""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(queries, key=_query_order_key)
    end = max(1, int(len(ordered) * fraction)) if ordered else 0
    return ordered[:end]


def _fixed_random_seeds(graph: WeightedDiGraph, seed: int) -> list[NodeId]:
    nodes = sorted(graph.adjacency)
    random.Random(seed).shuffle(nodes)
    return nodes


def _has_high_overlap(
    nodes: frozenset[NodeId],
    accepted: list[Region],
    node_to_region_ids: Mapping[NodeId, list[int]],
    threshold: float,
) -> bool:
    if threshold >= 1.0:
        return False
    intersections: Counter[int] = Counter()
    for node in nodes:
        intersections.update(node_to_region_ids.get(node, ()))
    for region_id, intersection in intersections.items():
        other_size = accepted[region_id].node_count
        union = len(nodes) + other_size - intersection
        if union and intersection / union >= threshold:
            return True
    return False


def _candidate_digest(candidates: list[dict]) -> str:
    canonical = json.dumps(
        candidates,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _query_order_key(query: Query) -> tuple[int, int]:
    return (
        query.timestamp if query.timestamp is not None else query.query_id,
        query.query_id,
    )
