"""Audit the frozen Chicago taxi Parquet and extract a temporal 120k OD sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import awkward as ak
import numpy as np


EXPECTED_MD5 = "e7bde64b9e87f41b27edfa2da7424a23"
SELECTED_COLUMNS = "trip.{sec,km,begin.*,end.*}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/compressed/chicago/chicago-taxi.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chicago/chicago_od_presample_120k.csv"),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("data/processed/chicago/chicago_od_presample_manifest.json"),
    )
    parser.add_argument("--sample-count", type=int, default=120_000)
    parser.add_argument("--min-lon", type=float, default=-89.0)
    parser.add_argument("--max-lon", type=float, default=-86.0)
    parser.add_argument("--min-lat", type=float, default=40.0)
    parser.add_argument("--max-lat", type=float, default=43.0)
    parser.add_argument("--skip-source-md5", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise SystemExit("--sample-count must be positive")
    if not args.min_lon < args.max_lon or not args.min_lat < args.max_lat:
        raise SystemExit("invalid coordinate bounds")

    source_md5 = None if args.skip_source_md5 else _digest(args.input, "md5")
    if source_md5 is not None and source_md5 != EXPECTED_MD5:
        raise SystemExit(
            f"source MD5 mismatch: expected {EXPECTED_MD5}, observed {source_md5}"
        )

    metadata = ak.metadata_from_parquet(str(args.input))
    row_groups = int(metadata["num_row_groups"])
    taxi_rows = int(metadata["num_rows"])
    timestamp_counts: Counter[int] = Counter()
    audit_counts: Counter[str] = Counter()
    taxi_base = 0

    print(
        f"audit pass: taxi_rows={taxi_rows:,}, row_groups={row_groups:,}",
        flush=True,
    )
    for row_group in range(row_groups):
        array = _read_row_group(args.input, row_group)
        batch = _flatten_batch(array, taxi_base)
        taxi_base += len(array)
        valid, reasons = _quality_mask(batch, args)
        audit_counts.update(reasons)
        times, counts = np.unique(batch["begin_time_ms"][valid], return_counts=True)
        timestamp_counts.update(
            {int(timestamp): int(count) for timestamp, count in zip(times, counts)}
        )
        print(
            f"audit row_group={row_group + 1}/{row_groups} "
            f"trips={len(valid):,} valid={int(valid.sum()):,}",
            flush=True,
        )

    valid_count = sum(timestamp_counts.values())
    if valid_count == 0:
        raise SystemExit("no field-valid Chicago trips")
    target_count = min(args.sample_count, valid_count)
    targets = _targets_by_timestamp(timestamp_counts, target_count)

    selected: list[tuple[int, int, int, float, float, float, float, float, float]] = []
    seen_by_timestamp: defaultdict[int, int] = defaultdict(int)
    taxi_base = 0
    print(f"selection pass: field_valid={valid_count:,} target={target_count:,}", flush=True)
    for row_group in range(row_groups):
        array = _read_row_group(args.input, row_group)
        batch = _flatten_batch(array, taxi_base)
        taxi_base += len(array)
        valid, _ = _quality_mask(batch, args)
        valid_indices = np.flatnonzero(valid)
        valid_times = batch["begin_time_ms"][valid_indices]
        chosen = _select_batch_indices(
            valid_indices,
            valid_times,
            targets,
            seen_by_timestamp,
        )
        for index in chosen:
            selected.append(
                (
                    int(batch["begin_time_ms"][index]),
                    int(batch["taxi_id"][index]),
                    int(batch["trip_index"][index]),
                    float(batch["origin_lon"][index]),
                    float(batch["origin_lat"][index]),
                    float(batch["dest_lon"][index]),
                    float(batch["dest_lat"][index]),
                    float(batch["seconds"][index]),
                    float(batch["kilometres"][index]),
                )
            )
        print(
            f"select row_group={row_group + 1}/{row_groups} "
            f"selected_so_far={len(selected):,}",
            flush=True,
        )

    selected.sort(key=lambda row: (row[0], row[1], row[2]))
    if len(selected) != target_count:
        raise RuntimeError(f"selected {len(selected):,}, expected {target_count:,}")
    _write_od_csv(args.output, selected)
    output_sha256 = _digest(args.output, "sha256")
    manifest = {
        "schema": "aic.chicago_od_presample.v1",
        "source": {
            "dataset": "Chicago taxi rides",
            "doi": "10.5281/zenodo.14537442",
            "license": "CC-BY-4.0",
            "path": str(args.input),
            "size_bytes": args.input.stat().st_size,
            "expected_md5": EXPECTED_MD5,
            "observed_md5": source_md5,
            "parquet_taxi_rows": taxi_rows,
            "parquet_row_groups": row_groups,
        },
        "quality_protocol": {
            "coordinate_bounds": {
                "min_lon": args.min_lon,
                "max_lon": args.max_lon,
                "min_lat": args.min_lat,
                "max_lat": args.max_lat,
            },
            "end_not_before_begin": True,
            "negative_seconds_or_kilometres_rejected_when_present": True,
            "selection": "global chronological equal-index sample; stable tie order taxi_id/trip_index",
            "requested_sample_count": args.sample_count,
        },
        "audit_counts": dict(sorted(audit_counts.items())),
        "field_valid_trip_count": valid_count,
        "distinct_valid_timestamps": len(timestamp_counts),
        "selected_trip_count": len(selected),
        "selected_time_range_ms": [selected[0][0], selected[-1][0]],
        "selected_unique_origin_coordinates": len(
            {(row[3], row[4]) for row in selected}
        ),
        "selected_unique_destination_coordinates": len(
            {(row[5], row[6]) for row in selected}
        ),
        "output": str(args.output),
        "output_sha256": output_sha256,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output={args.output}", flush=True)
    print(f"manifest={args.manifest_output}", flush=True)
    print(f"output_sha256={output_sha256}", flush=True)


def _read_row_group(path: Path, row_group: int) -> ak.Array:
    return ak.from_parquet(
        str(path),
        columns=SELECTED_COLUMNS,
        row_groups={row_group},
    )


def _flatten_batch(array: ak.Array, taxi_base: int) -> dict[str, np.ndarray]:
    trip = array.trip
    taxi_ids = ak.broadcast_arrays(
        ak.Array(np.arange(taxi_base, taxi_base + len(array), dtype=np.int64)),
        trip.begin.time,
    )[0]
    trip_indices = ak.local_index(trip.begin.time, axis=1)

    return {
        "taxi_id": _to_numpy(ak.flatten(taxi_ids), -1, np.int64),
        "trip_index": _to_numpy(ak.flatten(trip_indices), -1, np.int64),
        "origin_lon": _to_numpy(ak.flatten(trip.begin.lon), np.nan, np.float64),
        "origin_lat": _to_numpy(ak.flatten(trip.begin.lat), np.nan, np.float64),
        "dest_lon": _to_numpy(ak.flatten(trip.end.lon), np.nan, np.float64),
        "dest_lat": _to_numpy(ak.flatten(trip.end.lat), np.nan, np.float64),
        "begin_time_ms": _timestamps_ms(ak.flatten(trip.begin.time)),
        "end_time_ms": _timestamps_ms(ak.flatten(trip.end.time)),
        "seconds": _to_numpy(ak.flatten(trip.sec), np.nan, np.float64),
        "kilometres": _to_numpy(ak.flatten(trip.km), np.nan, np.float64),
    }


def _to_numpy(values: ak.Array, fill_value: object, dtype: np.dtype) -> np.ndarray:
    return np.asarray(ak.to_numpy(ak.fill_none(values, fill_value)), dtype=dtype)


def _timestamps_ms(values: ak.Array) -> np.ndarray:
    filled = ak.fill_none(values, np.datetime64("NaT", "ms"))
    return np.asarray(ak.to_numpy(filled), dtype="datetime64[ms]").astype(np.int64)


def _quality_mask(
    batch: dict[str, np.ndarray], args: argparse.Namespace
) -> tuple[np.ndarray, Counter[str]]:
    size = len(batch["taxi_id"])
    nat = np.iinfo(np.int64).min
    times_ok = (
        (batch["begin_time_ms"] != nat)
        & (batch["end_time_ms"] != nat)
        & (batch["end_time_ms"] >= batch["begin_time_ms"])
    )
    coordinates_ok = np.ones(size, dtype=bool)
    for field in ("origin_lon", "origin_lat", "dest_lon", "dest_lat"):
        coordinates_ok &= np.isfinite(batch[field])
    bounds_ok = (
        (batch["origin_lon"] >= args.min_lon)
        & (batch["origin_lon"] <= args.max_lon)
        & (batch["dest_lon"] >= args.min_lon)
        & (batch["dest_lon"] <= args.max_lon)
        & (batch["origin_lat"] >= args.min_lat)
        & (batch["origin_lat"] <= args.max_lat)
        & (batch["dest_lat"] >= args.min_lat)
        & (batch["dest_lat"] <= args.max_lat)
    )
    seconds_ok = np.isnan(batch["seconds"]) | (batch["seconds"] >= 0)
    kilometres_ok = np.isnan(batch["kilometres"]) | (batch["kilometres"] >= 0)
    valid = times_ok & coordinates_ok & bounds_ok & seconds_ok & kilometres_ok
    counts = Counter(
        {
            "total_trip_records": size,
            "invalid_time_or_order": int((~times_ok).sum()),
            "nonfinite_coordinates": int((~coordinates_ok).sum()),
            "outside_fixed_bounds": int((coordinates_ok & ~bounds_ok).sum()),
            "negative_seconds": int((~seconds_ok).sum()),
            "negative_kilometres": int((~kilometres_ok).sum()),
            "field_valid": int(valid.sum()),
        }
    )
    return valid, counts


def _targets_by_timestamp(
    timestamp_counts: Counter[int], target_count: int
) -> dict[int, np.ndarray]:
    total = sum(timestamp_counts.values())
    if target_count == total:
        global_targets = np.arange(total, dtype=np.int64)
    else:
        global_targets = np.linspace(0, total - 1, target_count, dtype=np.int64)
    targets: dict[int, np.ndarray] = {}
    cursor = 0
    target_cursor = 0
    for timestamp, count in sorted(timestamp_counts.items()):
        end = cursor + count
        next_cursor = int(np.searchsorted(global_targets, end, side="left"))
        if next_cursor > target_cursor:
            targets[timestamp] = global_targets[target_cursor:next_cursor] - cursor
        target_cursor = next_cursor
        cursor = end
    return targets


def _select_batch_indices(
    valid_indices: np.ndarray,
    valid_times: np.ndarray,
    targets: dict[int, np.ndarray],
    seen: defaultdict[int, int],
) -> np.ndarray:
    if len(valid_indices) == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(valid_times, kind="stable")
    sorted_times = valid_times[order]
    unique_times, starts, counts = np.unique(
        sorted_times, return_index=True, return_counts=True
    )
    chosen: list[np.ndarray] = []
    for timestamp_raw, start, count in zip(unique_times, starts, counts):
        timestamp = int(timestamp_raw)
        before = seen[timestamp]
        desired = targets.get(timestamp)
        if desired is not None:
            lo = int(np.searchsorted(desired, before, side="left"))
            hi = int(np.searchsorted(desired, before + int(count), side="left"))
            if hi > lo:
                local = desired[lo:hi] - before
                chosen.append(valid_indices[order[start + local]])
        seen[timestamp] = before + int(count)
    if not chosen:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chosen)


def _write_od_csv(
    path: Path,
    rows: list[tuple[int, int, int, float, float, float, float, float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "trip_id",
                "timestamp",
                "origin_lon",
                "origin_lat",
                "dest_lon",
                "dest_lat",
                "trip_seconds",
                "trip_kilometres",
            ]
        )
        for begin_ms, taxi_id, trip_index, *values in rows:
            writer.writerow(
                [
                    f"{taxi_id}:{trip_index}",
                    begin_ms // 1000,
                    *values,
                ]
            )


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
