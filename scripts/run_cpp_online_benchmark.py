"""Build, run, validate, and report the two-city C++ online benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.export_cpp_online_benchmark import DEFAULT_METHODS, export_benchmark_input


@dataclass(frozen=True)
class CityConfig:
    name: str
    node_csv: Path
    edge_csv: Path
    query_csv: Path
    candidates: Path
    current_manifest: Path
    future_manifest: Path
    selections: tuple[Path, ...]
    python_summaries: tuple[Path, ...]


CITY_CONFIGS = {
    "porto": CityConfig(
        name="porto",
        node_csv=ROOT_DIR / "data/processed/porto/波尔图道路节点.csv",
        edge_csv=ROOT_DIR / "data/processed/porto/波尔图道路边.csv",
        query_csv=ROOT_DIR / "data/processed/porto/波尔图可用起终点节点查询_200米.csv",
        candidates=ROOT_DIR / "results/gnn_v2/candidate_manifest.json",
        current_manifest=ROOT_DIR / "results/gnn_v2/label_manifest.json",
        future_manifest=ROOT_DIR / "results/gnn_v2/future_window_z0/future_label_manifest.json",
        selections=(
            ROOT_DIR / "results/gnn_v2/multi_region_online/selections.csv",
            ROOT_DIR / "results/gnn_v2/multi_region_online_g4/selections.csv",
        ),
        python_summaries=(
            ROOT_DIR / "results/gnn_v2/multi_region_online/summary.json",
            ROOT_DIR / "results/gnn_v2/multi_region_online_g4/summary.json",
        ),
    ),
    "chicago": CityConfig(
        name="chicago",
        node_csv=ROOT_DIR / "data/processed/chicago/chicago_road_nodes.csv",
        edge_csv=ROOT_DIR / "data/processed/chicago/chicago_road_edges.csv",
        query_csv=ROOT_DIR / "data/processed/chicago/chicago_queries_100k.csv",
        candidates=ROOT_DIR / "results/chicago/gnn_v2/candidate_manifest.json",
        current_manifest=ROOT_DIR / "results/chicago/gnn_v2/label_manifest.json",
        future_manifest=(
            ROOT_DIR / "results/chicago/gnn_v2/future_window_z0/future_label_manifest.json"
        ),
        selections=(
            ROOT_DIR
            / "results/chicago/gnn_v2/multi_region_online_g4_clean_rerun/selections.csv",
        ),
        python_summaries=(
            ROOT_DIR
            / "results/chicago/gnn_v2/multi_region_online_g4_clean_rerun/summary.json",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen two-city C++ benchmark.")
    parser.add_argument("--city", choices=("all", *CITY_CONFIGS), default="all")
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse a complete raw city summary and only validate/report it.",
    )
    return parser.parse_args()


def run_benchmark(
    city: CityConfig,
    *,
    binary: Path,
    build_dir: Path,
    cpu: int,
    warmup: int,
    repetitions: int,
    skip_export: bool,
    reuse_existing: bool,
) -> dict[str, object]:
    input_path = build_dir / f"{city.name}.benchmark.bin"
    if not skip_export:
        input_metadata = export_benchmark_input(
            node_csv=city.node_csv,
            edge_csv=city.edge_csv,
            query_csv=city.query_csv,
            candidate_path=city.candidates,
            current_manifest_path=city.current_manifest,
            future_manifest_path=city.future_manifest,
            selection_paths=city.selections,
            methods=DEFAULT_METHODS,
            k=18,
            output_path=input_path,
        )
    else:
        input_metadata = json.loads(input_path.with_suffix(".json").read_text())

    output_dir = ROOT_DIR / "results/cpp_online_benchmark" / city.name
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_summary = build_dir / f"{city.name}.summary.csv.tmp"
    log_path = build_dir / f"{city.name}.launcher.log"
    command = [
        "taskset",
        "-c",
        str(cpu),
        str(binary),
        "--input",
        str(input_path),
        "--output",
        str(temporary_summary),
        "--warmup",
        str(warmup),
        "--repetitions",
        str(repetitions),
    ]
    summary_path = output_dir / "summary.csv"
    raw_summary = temporary_summary
    complete_existing = False
    if reuse_existing:
        for candidate in (temporary_summary, summary_path):
            if not candidate.exists():
                continue
            with candidate.open(encoding="utf-8", newline="") as file:
                if sum(1 for _ in csv.DictReader(file)) == len(DEFAULT_METHODS) * 2:
                    raw_summary = candidate
                    complete_existing = True
                    break
    if complete_existing:
        print(f"reusing complete raw summary for {city.name}: {raw_summary}", flush=True)
    else:
        print(f"running {city.name}: {' '.join(command)}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=ROOT_DIR,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"{city.name} benchmark failed; see {log_path}")

    rows = _read_summary(raw_summary)
    _validate_rows(rows, city.python_summaries)
    if raw_summary != summary_path:
        raw_summary.replace(summary_path)
    protocol = {
        "schema": "aic.cpp_online_benchmark.v1",
        "city": city.name,
        "k": 18,
        "methods": list(DEFAULT_METHODS),
        "windows": ["current_y", "future_f"],
        "query_count_per_window": 2_000,
        "endpoint_cache_capacity": 0,
        "threads": 1,
        "cpu_affinity": cpu,
        "warmup_rounds": warmup,
        "repetitions": repetitions,
        "timing": (
            "steady_clock per-query latency; baseline/indexed order alternates by "
            "query-id plus repetition parity"
        ),
        "correctness": "all indexed distances must match original graph within 1e-6",
        "python_expansion_replay_tolerance_nodes_per_query": 0.1,
        "compiler": _command_output(["g++", "--version"]).splitlines()[0],
        "compile_flags": "-O3 -DNDEBUG -march=native -std=c++20",
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "input": input_metadata,
        "binary_sha256": _sha256(binary),
        "summary_sha256": _sha256(summary_path),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "report.md", protocol)
    print(f"completed {city.name}: {output_dir / 'report.md'}", flush=True)
    return protocol


def _build(build_dir: Path) -> Path:
    subprocess.run(
        ["cmake", "-S", str(ROOT_DIR / "cpp"), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=ROOT_DIR,
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build_dir), "-j2"], cwd=ROOT_DIR, check=True)
    binary = build_dir / "aic_cpp_online_benchmark"
    subprocess.run([str(binary), "--self-test"], cwd=ROOT_DIR, check=True)
    return binary


def _read_summary(path: Path) -> list[dict[str, object]]:
    integer_columns = {
        "query_count",
        "region_count",
        "shortcut_count",
        "internal_node_count",
        "warmup_rounds",
        "repetitions",
    }
    text_columns = {"method", "window"}
    with path.open(encoding="utf-8", newline="") as file:
        rows = []
        for raw in csv.DictReader(file):
            row: dict[str, object] = {}
            for key, value in raw.items():
                if key in text_columns:
                    row[key] = value
                elif key in integer_columns:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def _validate_rows(rows: list[dict[str, object]], python_paths: tuple[Path, ...]) -> None:
    expected: dict[tuple[str, str], dict[str, object]] = {}
    for path in python_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        for key, windows in summary["runs"].items():
            if not key.endswith(".k18"):
                continue
            method = key.removesuffix(".k18")
            for window, values in windows.items():
                expected[(method, window)] = values

    if len(rows) != len(DEFAULT_METHODS) * 2:
        raise ValueError("C++ summary must contain every frozen method/window pair")
    for row in rows:
        key = (str(row["method"]), str(row["window"]))
        reference = expected.get(key)
        if reference is None:
            raise ValueError(f"missing Python reference for {key}")
        if float(row["correctness_rate"]) != 1.0:
            raise ValueError(f"C++ correctness failed for {key}")
        for cpp_key, python_key in (
            ("shortcut_count", "shortcut_count"),
            ("internal_node_count", "internal_node_count"),
            ("baseline_avg_expanded", "baseline_avg_expanded"),
            ("indexed_avg_expanded", "indexed_avg_expanded"),
        ):
            tolerance = 1e-6 if cpp_key in {"shortcut_count", "internal_node_count"} else 1e-1
            if abs(float(row[cpp_key]) - float(reference[python_key])) > tolerance:
                raise ValueError(
                    f"C++/Python structural mismatch for {key}: {cpp_key} "
                    f"{row[cpp_key]} != {reference[python_key]}"
                )


def _write_report(path: Path, summary: dict[str, object]) -> None:
    rows = summary["rows"]
    lines = [
        f"# {str(summary['city']).title()} C++ 精确在线性能评测",
        "",
        "同一个 C++20/O3 可执行程序内实现原图双向 Dijkstra 与压缩查询；单线程固定 CPU，"
        "缓存关闭，先逐查询校验距离与 Python 展开节点，再预热并重复计时。",
        "",
        "| 方法 | 窗口 | shortcuts | 原图均值 | 压缩均值 | 耗时变化 | P95 变化 | 展开变化 | 扫描边变化 | 正确率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['window']} | {row['shortcut_count']:,} | "
            f"{row['baseline_avg_ms']:.4f} ms | {row['indexed_avg_ms']:.4f} ms | "
            f"{row['elapsed_change_pct']:+.2f}% | {row['p95_change_pct']:+.2f}% | "
            f"{row['expanded_change_pct']:+.2f}% | {row['scanned_edges_change_pct']:+.2f}% | "
            f"{row['correctness_rate']:.6f} |"
        )
    faster = sum(float(row["elapsed_change_pct"]) < 0.0 for row in rows)
    lines.extend(
        [
            "",
            f"共 {len(rows)} 个方法—窗口组合，其中 {faster} 个平均墙钟耗时下降；"
            "所有距离正确率为 100%，shortcut、内部节点精确重放；平均展开节点与 Python 冻结结果"
            "每查询相差不超过 0.1（等长路径陈旧堆项清理造成的 tie 顺序差异）。",
            "",
            f"协议：CPU `{summary['cpu_affinity']}`，单线程，预热 `{summary['warmup_rounds']}` 轮，"
            f"正式重复 `{summary['repetitions']}` 轮；编译参数 `{summary['compile_flags']}`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _command_output(command: list[str]) -> str:
    return subprocess.run(command, text=True, capture_output=True, check=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_cpu() -> int:
    affinity = sorted(os.sched_getaffinity(0))
    if not affinity:
        raise RuntimeError("process has no available CPU")
    return affinity[len(affinity) // 2]


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.repetitions <= 0:
        raise SystemExit("--warmup must be non-negative and --repetitions positive")
    cpu = _default_cpu() if args.cpu is None else args.cpu
    if cpu not in os.sched_getaffinity(0):
        raise SystemExit(f"CPU {cpu} is outside current process affinity")
    build_dir = ROOT_DIR / "build/cpp-online"
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / "aic_cpp_online_benchmark"
    if not args.skip_build:
        binary = _build(build_dir)
    elif not binary.exists():
        raise SystemExit(f"missing benchmark binary: {binary}")
    cities = CITY_CONFIGS.values() if args.city == "all" else (CITY_CONFIGS[args.city],)
    for city in cities:
        run_benchmark(
            city,
            binary=binary,
            build_dir=build_dir,
            cpu=cpu,
            warmup=args.warmup,
            repetitions=args.repetitions,
            skip_export=args.skip_export,
            reuse_existing=args.reuse_existing,
        )


if __name__ == "__main__":
    main()
