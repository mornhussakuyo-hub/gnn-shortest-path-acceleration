"""按控制变量矩阵运行第一版 GNN 大型消融实验。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "gnn_ablation"
TRAIN_SCRIPT = ROOT_DIR / "scripts" / "train_gnn_seed_model.py"
EVALUATE_SCRIPT = ROOT_DIR / "scripts" / "evaluate_gnn_seed_model.py"
NON_LEARNING_SCORE_SCRIPT = ROOT_DIR / "scripts" / "generate_non_learning_seed_scores.py"


@dataclass(frozen=True, slots=True)
class AblationVariant:
    name: str
    score_source: str = "gnn"
    model_type: str = "graph_sage"
    target_mode: str = "midpoint"
    excluded_features: tuple[str, ...] = field(default_factory=tuple)
    diffusion_steps: int = 3
    diffusion_restart: float = 0.4
    endpoint_penalty: float = 2.0
    region_risk_penalty: float = 200.0
    candidate_limit: int = 80_000


@dataclass(slots=True)
class TerminalProgress:
    total_runs: int
    enabled: bool

    def update(
        self,
        processed_runs: int,
        run_id: str,
        stage: str,
        elapsed_seconds: float,
    ) -> None:
        if not self.enabled:
            return
        ratio = processed_runs / max(1, self.total_runs)
        bar_width = 24
        filled = min(bar_width, round(ratio * bar_width))
        bar = "█" * filled + "░" * (bar_width - filled)
        elapsed = _format_duration(elapsed_seconds)
        line = (
            f"[{bar}] {processed_runs:>2}/{self.total_runs} {ratio * 100:5.1f}% "
            f"| {stage} | {run_id} | {elapsed}"
        )
        sys.stdout.write(f"\r\033[2K{line}")
        sys.stdout.flush()

    def message(self, message: str) -> None:
        if self.enabled:
            self.clear()
        print(message, flush=True)

    def clear(self) -> None:
        if self.enabled:
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行可断点续跑的 GNN 第一版大型消融实验。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--gnn-region-count", type=int, default=85)
    parser.add_argument("--region-count", type=int, default=100)
    parser.add_argument("--region-size", type=int, default=512)
    parser.add_argument("--candidate-limit", type=int, default=80_000)
    parser.add_argument(
        "--suite",
        choices=("core", "comprehensive"),
        default="core",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = _variants(args.suite, args.candidate_limit)
    runs = [
        (variant, seed)
        for variant in variants
        for seed in (args.random_seeds if variant.score_source == "gnn" else [None])
    ]
    print(
        f"suite={args.suite} variants={len(variants)} "
        f"seeds={len(args.random_seeds)} total_runs={len(runs)}",
        flush=True,
    )
    if args.dry_run:
        for variant, seed in runs:
            seed_label = seed if seed is not None else "deterministic"
            print(f"{variant.name} seed={seed_label}", flush=True)
        return

    if args.restart and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(args.output_dir / "ablation_manifest.json", variants, args)
    progress = TerminalProgress(
        total_runs=len(runs),
        enabled=sys.stdout.isatty() and not args.no_progress,
    )

    failures: list[str] = []
    reference_dir = args.output_dir / "reference"
    if not _evaluation_complete(reference_dir, args.evaluation_split):
        reference_dir.mkdir(parents=True, exist_ok=True)
        reference_command = [
            sys.executable,
            str(EVALUATE_SCRIPT),
            "--output-dir",
            str(reference_dir),
            "--methods",
            "random",
            "hotspot",
            "--region-count",
            str(args.region_count),
            "--region-size",
            str(args.region_size),
            "--workers",
            str(args.workers),
            "--chunk-size",
            str(args.chunk_size),
            "--evaluation-split",
            args.evaluation_split,
        ]
        if not _run_command(
            reference_command,
            reference_dir / "run.log",
            progress,
            0,
            "reference",
            "公共基线",
        ):
            failures.append("reference")
    else:
        progress.message("skip completed reference baselines")

    for index, (variant, seed) in enumerate(runs, start=1):
        run_id = (
            f"{variant.name}__seed{seed}"
            if seed is not None
            else variant.name
        )
        run_dir = args.output_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if not progress.enabled:
            print(f"[{index}/{len(runs)}] {run_id}", flush=True)
        preparation_summary = _preparation_summary_path(variant, run_dir)
        if not preparation_summary.is_file():
            train_command = _preparation_command(args, variant, seed, run_dir)
            if not _run_command(
                train_command,
                run_dir / "run.log",
                progress,
                index - 1,
                run_id,
                "GPU训练" if variant.score_source == "gnn" else "生成解析分数",
            ):
                failures.append(f"{run_id}:train")
                progress.message(f"failed training: {run_id}")
                continue
        else:
            progress.message(f"skip completed training: {run_id}")

        if not _evaluation_complete(run_dir, args.evaluation_split):
            evaluate_command = _evaluation_command(args, variant, run_dir)
            if not _run_command(
                evaluate_command,
                run_dir / "run.log",
                progress,
                index - 1,
                run_id,
                "选区与精确评测",
            ):
                failures.append(f"{run_id}:evaluate")
                progress.message(f"failed evaluation: {run_id}")
                continue
        else:
            progress.message(f"skip completed evaluation: {run_id}")
        _rebuild_aggregate(args.output_dir, variants, args.evaluation_split)
        progress.update(index, run_id, "完成并落盘", 0.0)

    progress.clear()
    _rebuild_aggregate(args.output_dir, variants, args.evaluation_split)
    if failures:
        failure_path = args.output_dir / "failed_runs.txt"
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise SystemExit(f"有 {len(failures)} 个步骤失败，详见 {failure_path}")
    (args.output_dir / "failed_runs.txt").unlink(missing_ok=True)
    print("all ablation runs completed", flush=True)


def _variants(suite: str, default_candidate_limit: int) -> list[AblationVariant]:
    core_variants = [
        AblationVariant("full"),
        AblationVariant("risk_only", score_source="risk_only"),
        AblationVariant("proxy_only", score_source="proxy_only"),
        AblationVariant(
            "gnn_only_no_endpoint_risk",
            endpoint_penalty=0.0,
            region_risk_penalty=0.0,
        ),
        AblationVariant("mlp_no_message_passing", model_type="mlp"),
        AblationVariant(
            "no_diffused_demand_features",
            excluded_features=("diffused_origin_demand", "diffused_destination_demand"),
        ),
        AblationVariant(
            "no_raw_endpoint_frequency",
            excluded_features=("origin_frequency", "destination_frequency"),
        ),
        AblationVariant("no_endpoint_proxy_penalty", endpoint_penalty=0.0),
        AblationVariant("demand_overlap_target", target_mode="demand_overlap"),
        AblationVariant("diffusion_steps_0", diffusion_steps=0),
        AblationVariant("diffusion_steps_1", diffusion_steps=1),
        AblationVariant("diffusion_steps_5", diffusion_steps=5),
        AblationVariant("region_risk_penalty_0", region_risk_penalty=0.0),
        AblationVariant("region_risk_penalty_50", region_risk_penalty=50.0),
        AblationVariant("region_risk_penalty_100", region_risk_penalty=100.0),
        AblationVariant("region_risk_penalty_400", region_risk_penalty=400.0),
    ]
    if suite == "core":
        return core_variants

    extended_variants = [
        AblationVariant("diffusion_steps_8", diffusion_steps=8),
        AblationVariant("diffusion_steps_12", diffusion_steps=12),
        AblationVariant("diffusion_steps_20", diffusion_steps=20),
        AblationVariant("diffusion_restart_0", diffusion_restart=0.0),
        AblationVariant("diffusion_restart_0_2", diffusion_restart=0.2),
        AblationVariant("diffusion_restart_0_6", diffusion_restart=0.6),
        AblationVariant("diffusion_restart_0_8", diffusion_restart=0.8),
        AblationVariant(
            "no_coordinates",
            excluded_features=("longitude", "latitude"),
        ),
        AblationVariant(
            "no_degree_features",
            excluded_features=("log_out_degree", "log_in_degree"),
        ),
        AblationVariant(
            "no_mean_edge_length",
            excluded_features=("mean_out_edge_length",),
        ),
        AblationVariant("endpoint_penalty_0_5", endpoint_penalty=0.5),
        AblationVariant("endpoint_penalty_1", endpoint_penalty=1.0),
        AblationVariant("endpoint_penalty_4", endpoint_penalty=4.0),
        AblationVariant("endpoint_penalty_8", endpoint_penalty=8.0),
        AblationVariant("region_risk_penalty_800", region_risk_penalty=800.0),
        AblationVariant("candidate_limit_20000", candidate_limit=20_000),
        AblationVariant("candidate_limit_40000", candidate_limit=40_000),
        AblationVariant("candidate_limit_120000", candidate_limit=120_000),
    ]
    for diffusion_steps in (5, 8, 12, 20):
        for diffusion_restart in (0.2, 0.6):
            extended_variants.append(
                AblationVariant(
                    f"diffusion_steps_{diffusion_steps}_restart_"
                    f"{str(diffusion_restart).replace('.', '_')}",
                    diffusion_steps=diffusion_steps,
                    diffusion_restart=diffusion_restart,
                )
            )
    variants = core_variants + extended_variants
    if default_candidate_limit != 80_000:
        variants = [
            AblationVariant(
                **{
                    **asdict(variant),
                    "candidate_limit": (
                        default_candidate_limit
                        if variant.candidate_limit == 80_000
                        else variant.candidate_limit
                    ),
                }
            )
            for variant in variants
        ]
    return variants


def _preparation_command(
    args: argparse.Namespace,
    variant: AblationVariant,
    seed: int | None,
    run_dir: Path,
) -> list[str]:
    if variant.score_source != "gnn":
        return [
            sys.executable,
            str(NON_LEARNING_SCORE_SCRIPT),
            "--output-dir",
            str(run_dir),
            "--score-source",
            variant.score_source,
            "--target-mode",
            variant.target_mode,
            "--diffusion-steps",
            str(variant.diffusion_steps),
            "--diffusion-restart",
            str(variant.diffusion_restart),
            "--endpoint-penalty",
            str(variant.endpoint_penalty),
        ]
    if seed is None:
        raise ValueError("GNN 实验必须提供随机种子。")
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(run_dir),
        "--model-type",
        variant.model_type,
        "--target-mode",
        variant.target_mode,
        "--diffusion-steps",
        str(variant.diffusion_steps),
        "--diffusion-restart",
        str(variant.diffusion_restart),
        "--endpoint-penalty",
        str(variant.endpoint_penalty),
        "--region-endpoint-risk-penalty",
        str(variant.region_risk_penalty),
        "--region-count",
        str(args.gnn_region_count),
        "--region-size",
        str(args.region_size),
        "--candidate-limit",
        str(variant.candidate_limit),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--seed",
        str(seed),
        "--skip-region-selection",
    ]
    if variant.excluded_features:
        command.append("--exclude-features")
        command.extend(variant.excluded_features)
    return command


def _preparation_summary_path(variant: AblationVariant, run_dir: Path) -> Path:
    filename = (
        "training_summary.json"
        if variant.score_source == "gnn"
        else "scoring_summary.json"
    )
    return run_dir / filename


def _evaluation_command(
    args: argparse.Namespace,
    variant: AblationVariant,
    run_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--output-dir",
        str(run_dir),
        "--score-csv",
        str(run_dir / "node_scores.csv"),
        "--methods",
        "gnn",
        "--gnn-region-count",
        str(args.gnn_region_count),
        "--region-size",
        str(args.region_size),
        "--candidate-limit",
        str(variant.candidate_limit),
        "--region-endpoint-risk-penalty",
        str(variant.region_risk_penalty),
        "--workers",
        str(args.workers),
        "--chunk-size",
        str(args.chunk_size),
        "--evaluation-split",
        args.evaluation_split,
    ]


def _run_command(
    command: list[str],
    log_path: Path,
    progress: TerminalProgress,
    processed_runs: int,
    run_id: str,
    stage: str,
) -> bool:
    if not progress.enabled:
        print(shlex.join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n$ {shlex.join(command)}\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        start = time.monotonic()
        while process.poll() is None:
            progress.update(
                processed_runs,
                run_id,
                stage,
                time.monotonic() - start,
            )
            time.sleep(1.0)
    progress.clear()
    return process.returncode == 0


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def _evaluation_complete(output_dir: Path, evaluation_split: str) -> bool:
    summary_path = output_dir / "evaluation_summary.csv"
    if not summary_path.is_file():
        return False
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file), None)
        return row is not None and row.get("evaluation_split") == evaluation_split


def _write_manifest(
    path: Path,
    variants: list[AblationVariant],
    args: argparse.Namespace,
) -> None:
    payload = {
        "python": sys.executable,
        "random_seeds": args.random_seeds,
        "run_count": sum(
            len(args.random_seeds) if variant.score_source == "gnn" else 1
            for variant in variants
        ),
        "workers": args.workers,
        "epochs": args.epochs,
        "patience": args.patience,
        "gnn_region_count": args.gnn_region_count,
        "region_count": args.region_count,
        "region_size": args.region_size,
        "candidate_limit": args.candidate_limit,
        "evaluation_split": args.evaluation_split,
        "suite": args.suite,
        "variants": [asdict(variant) for variant in variants],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rebuild_aggregate(
    output_dir: Path,
    variants: list[AblationVariant],
    evaluation_split: str,
) -> None:
    variant_by_name = {variant.name: variant for variant in variants}
    rows: list[dict[str, object]] = []
    runs_dir = output_dir / "runs"
    if not runs_dir.is_dir():
        return
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        if not _evaluation_complete(run_dir, evaluation_split):
            continue
        if "__seed" in run_dir.name:
            variant_name, seed_text = run_dir.name.rsplit("__seed", 1)
            random_seed: int | str = int(seed_text)
        else:
            variant_name = run_dir.name
            random_seed = ""
        variant = variant_by_name.get(variant_name)
        if variant is None:
            continue
        preparation_path = _preparation_summary_path(variant, run_dir)
        if not preparation_path.is_file():
            continue
        preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
        with (run_dir / "evaluation_summary.csv").open(
            "r", encoding="utf-8", newline=""
        ) as file:
            evaluation = next(csv.DictReader(file))
        rows.append(
            {
                "run_id": run_dir.name,
                "variant": variant_name,
                "seed": random_seed,
                "score_source": variant.score_source,
                "model_type": variant.model_type,
                "target_mode": variant.target_mode,
                "excluded_features": ";".join(variant.excluded_features),
                "diffusion_steps": variant.diffusion_steps,
                "diffusion_restart": variant.diffusion_restart,
                "endpoint_penalty": variant.endpoint_penalty,
                "region_risk_penalty": variant.region_risk_penalty,
                "candidate_limit": variant.candidate_limit,
                "best_epoch": preparation.get("best_epoch", ""),
                "test_correlation": preparation["test_correlation"],
                "gpu_training_seconds": preparation.get("gpu_training_seconds", 0.0),
                "peak_gpu_memory_mb": preparation.get("peak_gpu_memory_mb", 0.0),
                **evaluation,
            }
        )
    if not rows:
        return
    aggregate_path = output_dir / "ablation_runs.csv"
    temporary_path = aggregate_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(aggregate_path)


if __name__ == "__main__":
    main()
