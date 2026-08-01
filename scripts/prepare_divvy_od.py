"""Audit the frozen June 2022 Divvy release and extract a temporal 120k OD sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/compressed/chicago/202206-divvy-tripdata.zip"),
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
    parser.add_argument("--maximum-duration-seconds", type=int, default=86_400)
    parser.add_argument("--min-lon", type=float, default=-89.0)
    parser.add_argument("--max-lon", type=float, default=-86.0)
    parser.add_argument("--min-lat", type=float, default=40.0)
    parser.add_argument("--max-lat", type=float, default=43.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0 or args.maximum_duration_seconds <= 0:
        raise SystemExit("sample count and maximum duration must be positive")

    rows, counts, member_name = _read_valid_rows(args)
    rows.sort(key=lambda row: (row[0], row[1]))
    if not rows:
        raise SystemExit("no field-valid Divvy trips")
    indices = _equal_index_sample(len(rows), min(args.sample_count, len(rows)))
    selected = [rows[index] for index in indices]
    _write_od_csv(args.output, selected)

    origin_coordinates = {(row[3], row[4]) for row in selected}
    destination_coordinates = {(row[5], row[6]) for row in selected}
    manifest = {
        "schema": "aic.divvy_od_presample.v1",
        "source": {
            "dataset": "Divvy public trip history",
            "release": "202206-divvy-tripdata.zip",
            "source_url": (
                "https://divvy-tripdata.s3.amazonaws.com/"
                "202206-divvy-tripdata.zip"
            ),
            "license_url": "https://divvybikes.com/data-license-agreement",
            "path": str(args.input),
            "size_bytes": args.input.stat().st_size,
            "sha256": _sha256(args.input),
            "zip_member": member_name,
        },
        "quality_protocol": {
            "strictly_positive_duration": True,
            "maximum_duration_seconds": args.maximum_duration_seconds,
            "coordinate_bounds": {
                "min_lon": args.min_lon,
                "max_lon": args.max_lon,
                "min_lat": args.min_lat,
                "max_lat": args.max_lat,
            },
            "selection": "global chronological equal-index sample; stable ride_id tie order",
            "requested_sample_count": args.sample_count,
            "uses_membership_or_bike_type_for_filtering": False,
        },
        "audit_counts": dict(sorted(counts.items())),
        "field_valid_trip_count": len(rows),
        "selected_trip_count": len(selected),
        "selected_time_range": [selected[0][0], selected[-1][0]],
        "selected_unique_origin_coordinates": len(origin_coordinates),
        "selected_unique_destination_coordinates": len(destination_coordinates),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"field_valid={len(rows):,} selected={len(selected):,} "
        f"unique_origins={len(origin_coordinates):,} "
        f"unique_destinations={len(destination_coordinates):,}",
        flush=True,
    )
    print(f"output={args.output}", flush=True)
    print(f"manifest={args.manifest_output}", flush=True)


def _read_valid_rows(
    args: argparse.Namespace,
) -> tuple[
    list[tuple[int, str, int, float, float, float, float, int]],
    Counter[str],
    str,
]:
    rows: list[tuple[int, str, int, float, float, float, float, int]] = []
    counts: Counter[str] = Counter()
    with zipfile.ZipFile(args.input) as archive:
        csv_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
            and not name.startswith("__MACOSX/")
            and not Path(name).name.startswith("._")
        ]
        if len(csv_members) != 1:
            raise ValueError(f"expected one CSV in Divvy ZIP, found {csv_members}")
        member_name = csv_members[0]
        with archive.open(member_name) as raw_file:
            import io

            with io.TextIOWrapper(raw_file, encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                required = {
                    "ride_id",
                    "started_at",
                    "ended_at",
                    "start_lat",
                    "start_lng",
                    "end_lat",
                    "end_lng",
                }
                missing = required - set(reader.fieldnames or ())
                if missing:
                    raise ValueError(f"Divvy CSV missing fields: {sorted(missing)}")
                for row in reader:
                    counts["total_trip_records"] += 1
                    parsed, reason = _parse_row(row, args)
                    if parsed is None:
                        counts[reason] += 1
                    else:
                        counts["field_valid"] += 1
                        rows.append(parsed)
    return rows, counts, member_name


def _parse_row(
    row: dict[str, str], args: argparse.Namespace
) -> tuple[tuple[int, str, int, float, float, float, float, int] | None, str]:
    try:
        started = datetime.fromisoformat(row["started_at"])
        ended = datetime.fromisoformat(row["ended_at"])
    except (TypeError, ValueError):
        return None, "invalid_timestamp"
    duration = int((ended - started).total_seconds())
    if not 0 < duration <= args.maximum_duration_seconds:
        return None, "invalid_duration"
    try:
        origin_lat = float(row["start_lat"])
        origin_lon = float(row["start_lng"])
        destination_lat = float(row["end_lat"])
        destination_lon = float(row["end_lng"])
    except (TypeError, ValueError):
        return None, "invalid_coordinates"
    if not all(
        math.isfinite(value)
        for value in (origin_lon, origin_lat, destination_lon, destination_lat)
    ):
        return None, "invalid_coordinates"
    if not (
        args.min_lon <= origin_lon <= args.max_lon
        and args.min_lon <= destination_lon <= args.max_lon
        and args.min_lat <= origin_lat <= args.max_lat
        and args.min_lat <= destination_lat <= args.max_lat
    ):
        return None, "outside_fixed_bounds"
    return (
        int((started - datetime(1970, 1, 1)).total_seconds()),
        row["ride_id"],
        int((ended - datetime(1970, 1, 1)).total_seconds()),
        origin_lon,
        origin_lat,
        destination_lon,
        destination_lat,
        duration,
    ), ""


def _equal_index_sample(size: int, target: int) -> list[int]:
    if not 0 < target <= size:
        raise ValueError("target must be between one and size")
    if target == size:
        return list(range(size))
    return [(index * (size - 1)) // (target - 1) for index in range(target)]


def _write_od_csv(
    path: Path,
    rows: list[tuple[int, str, int, float, float, float, float, int]],
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
        for started, ride_id, _, origin_lon, origin_lat, dest_lon, dest_lat, duration in rows:
            writer.writerow(
                [
                    ride_id,
                    started,
                    origin_lon,
                    origin_lat,
                    dest_lon,
                    dest_lat,
                    duration,
                    "",
                ]
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
