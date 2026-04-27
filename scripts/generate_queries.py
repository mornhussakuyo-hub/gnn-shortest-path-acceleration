from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


@dataclass
class Hotspot:
    id: int
    center_node: int
    center_lon: float
    center_lat: float
    nodes: np.ndarray


def load_coords(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes: list[int] = []
    lon_values: list[float] = []
    lat_values: list[float] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.startswith("v "):
                continue
            _, node, lon, lat = line.split()
            nodes.append(int(node))
            lon_values.append(int(lon) / 1e6)
            lat_values.append(int(lat) / 1e6)

    return (
        np.array(nodes, dtype=np.int64),
        np.array(lon_values, dtype=np.float64),
        np.array(lat_values, dtype=np.float64),
    )


def build_hotspots(
    nodes: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    num_hotspots: int,
    nodes_per_hotspot: int,
    rng: np.random.Generator,
) -> list[Hotspot]:
    center_indexes = rng.choice(len(nodes), size=num_hotspots, replace=False)
    hotspots: list[Hotspot] = []

    for hotspot_id, center_index in enumerate(tqdm(center_indexes, desc="Building hotspots"), start=1):
        distance_square = (lon - lon[center_index]) ** 2 + (lat - lat[center_index]) ** 2
        nearest_count = min(nodes_per_hotspot, len(nodes))
        nearest_indexes = np.argpartition(distance_square, nearest_count - 1)[:nearest_count]

        hotspots.append(
            Hotspot(
                id=hotspot_id,
                center_node=int(nodes[center_index]),
                center_lon=float(lon[center_index]),
                center_lat=float(lat[center_index]),
                nodes=nodes[nearest_indexes],
            )
        )

    return hotspots


def choose_different_node(
    rng: np.random.Generator,
    candidates: np.ndarray,
    forbidden_node: int,
) -> int:
    while True:
        node = int(rng.choice(candidates))
        if node != forbidden_node:
            return node


def generate_queries(
    all_nodes: np.ndarray,
    hotspots: list[Hotspot],
    num_queries: int,
    rng: np.random.Generator,
) -> list[dict[str, int | str]]:
    queries: list[dict[str, int | str]] = []
    query_types = ["hotspot_to_hotspot", "hotspot_to_random", "random"]
    probabilities = [0.70, 0.20, 0.10]

    for query_id in tqdm(range(1, num_queries + 1), desc="Generating queries"):
        query_type = str(rng.choice(query_types, p=probabilities))
        source_hotspot = -1
        target_hotspot = -1

        if query_type == "hotspot_to_hotspot":
            source_hotspot_obj, target_hotspot_obj = rng.choice(hotspots, size=2, replace=False)
            source_hotspot = source_hotspot_obj.id
            target_hotspot = target_hotspot_obj.id
            origin = int(rng.choice(source_hotspot_obj.nodes))
            destination = choose_different_node(rng, target_hotspot_obj.nodes, origin)

        elif query_type == "hotspot_to_random":
            source_hotspot_obj = rng.choice(hotspots)
            source_hotspot = source_hotspot_obj.id
            origin = int(rng.choice(source_hotspot_obj.nodes))
            destination = choose_different_node(rng, all_nodes, origin)

        else:
            origin = int(rng.choice(all_nodes))
            destination = choose_different_node(rng, all_nodes, origin)

        queries.append(
            {
                "query_id": query_id,
                "origin": origin,
                "destination": destination,
                "query_type": query_type,
                "source_hotspot": source_hotspot,
                "target_hotspot": target_hotspot,
                "count": 1,
            }
        )

    return queries


def write_hotspots(path: Path, hotspots: list[Hotspot]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["hotspot_id", "center_node", "center_lon", "center_lat", "num_nodes"],
        )
        writer.writeheader()
        for hotspot in hotspots:
            writer.writerow(
                {
                    "hotspot_id": hotspot.id,
                    "center_node": hotspot.center_node,
                    "center_lon": hotspot.center_lon,
                    "center_lat": hotspot.center_lat,
                    "num_nodes": len(hotspot.nodes),
                }
            )


def write_queries(path: Path, queries: list[dict[str, int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query_id",
                "origin",
                "destination",
                "query_type",
                "source_hotspot",
                "target_hotspot",
                "count",
            ],
        )
        writer.writeheader()
        writer.writerows(queries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coords",
        type=Path,
        default=Path("data/raw/dimacs/small/USA-road-d.NY.co"),
        help="DIMACS .co coordinate file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/query_loads/small"),
        help="Directory for generated query-load files.",
    )
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--num-hotspots", type=int, default=20)
    parser.add_argument("--nodes-per-hotspot", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    nodes, lon, lat = load_coords(args.coords)
    hotspots = build_hotspots(
        nodes,
        lon,
        lat,
        num_hotspots=args.num_hotspots,
        nodes_per_hotspot=args.nodes_per_hotspot,
        rng=rng,
    )

    train_queries = generate_queries(nodes, hotspots, args.train_size, rng)
    test_queries = generate_queries(nodes, hotspots, args.test_size, rng)

    write_hotspots(args.output_dir / "hotspots.csv", hotspots)
    write_queries(args.output_dir / "queries_train.csv", train_queries)
    write_queries(args.output_dir / "queries_test.csv", test_queries)

    print(f"Loaded nodes: {len(nodes):,}")
    print(f"Hotspots: {len(hotspots):,}")
    print(f"Train queries: {len(train_queries):,}")
    print(f"Test queries: {len(test_queries):,}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
