"""加载 OD 查询负载。"""

from __future__ import annotations

import csv
from pathlib import Path

from .graph_types import Query


def load_porto_queries(query_csv: Path, limit: int | None = None) -> list[Query]:
    queries: list[Query] = []
    with query_csv.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("snap_usable", "True") != "True":
                continue
            queries.append(
                Query(
                    query_id=int(row["query_id"]),
                    origin=int(row["origin_node"]),
                    destination=int(row["dest_node"]),
                    timestamp=int(row["timestamp"]) if row.get("timestamp") else None,
                )
            )
            if limit is not None and len(queries) >= limit:
                break
    return queries


def split_queries_chronologically(
    queries: list[Query],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> tuple[list[Query], list[Query], list[Query]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    ordered = sorted(
        queries,
        key=lambda query: (
            query.timestamp if query.timestamp is not None else query.query_id,
            query.query_id,
        ),
    )
    train_end = int(len(ordered) * train_ratio)
    validation_end = train_end + int(len(ordered) * validation_ratio)
    return ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:]
