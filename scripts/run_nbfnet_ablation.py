"""Run the GNN-v2 NBFNet ablation matrix with durable logs and resume support."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT_DIR / "scripts" / "train_demand_field_nbfnet.py"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "results" / "gnn_v2" / "nbfnet_ablation"
DEFAULT_DATASET_MANIFEST = (
    ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
)
BASE_SUMMARY = ROOT_DIR / "results" / "gnn_v2" / "nbfnet_base" / "summary.json"
MANIFEST_SCHEMA = "aic.gnn_v2.nbfnet_ablation_runner.v1"
CORE_VARIANTS = (
    "origin_only",
    "destination_only",
    "shared_parameters",
    "undirected",
    "degree_rewired",
    "shuffled_od",
    "fixed_diffusion",
    "graphsage",
    "no_edge_features",
    "no_interactions",
    "last_layer_only",
    "no_ranking",
)
PROPAGATION_VARIANTS = (
    "propagation_deep",
    "propagation_residual",
    "propagation_doubling",
    "propagation_residual_doubling",
)
RUNNER_VARIANTS = CORE_VARIANTS + PROPAGATION_VARIANTS
FULL_SEEDS = (42, 43, 44, 45, 46)
SCREENING_SEEDS = (44,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the core NBFNet ablation matrix serially on one CUDA GPU. "
            "Every run has a durable log and completed runs are skipped."
        )
    )
    parser.add_argument("--mode", choices=("screening", "full"), default="screening")
    parser.add_argument(
        "--variants",
        default=",".join(CORE_VARIANTS),
        help="Comma-separated variants. Defaults to the frozen core matrix.",
    )
    parser.add_argument(
        "--seeds",
        help="Optional comma-separated seed override. screening=44, full=42..46.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Output directory. Defaults to "
            "results/gnn_v2/nbfnet_ablation/<mode>."
        ),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--prototype-batch-size", type=int, default=8)
    parser.add_argument(
        "--expanded-graph-batch-size",
        type=int,
        default=4,
        help=(
            "Prototype batch for undirected and GraphSAGE variants, whose "
            "symmetrized edge set is twice as large."
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--randomization-seed", type=int, default=20260730)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun completed items and allow replacing an existing matrix identity.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed run instead of recording it and continuing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = _parse_variants(args.variants)
    seeds = (
        _parse_int_list(args.seeds, "--seeds")
        if args.seeds
        else list(FULL_SEEDS if args.mode == "full" else SCREENING_SEEDS)
    )
    output_root = (
        args.output_root
        if args.output_root is not None
        else DEFAULT_OUTPUT_ROOT / args.mode
    ).resolve()
    logs_dir = output_root / "logs"
    runs_dir = output_root / "runs"
    manifest_path = output_root / "manifest.json"
    lock_path = output_root / ".runner.lock"
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest = json.loads(
        DEFAULT_DATASET_MANIFEST.read_text(encoding="utf-8")
    )
    identity = {
        "schema": MANIFEST_SCHEMA,
        "mode": args.mode,
        "variants": variants,
        "seeds": seeds,
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "candidate_sha256": dataset_manifest["candidate_sha256"],
        "training": {
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "prototype_batch_size": args.prototype_batch_size,
            "expanded_graph_batch_size": args.expanded_graph_batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "randomization_seed": args.randomization_seed,
        },
    }
    identity["experiment_id"] = _digest(identity)[:16]
    manifest = _load_or_create_manifest(manifest_path, identity, args.force)
    matrix = [(variant, seed) for variant in variants for seed in seeds]
    print(
        f"ablation mode={args.mode} experiment={identity['experiment_id']} "
        f"runs={len(matrix)} output={_display_path(output_root)}",
        flush=True,
    )
    if args.dry_run:
        for position, (variant, seed) in enumerate(matrix, start=1):
            print(
                f"[{position}/{len(matrix)}] {variant} seed={seed} "
                f"prototype_batch_size={_prototype_batch_size(args, variant)}"
            )
        return

    _acquire_lock(lock_path)
    manifest["status"] = "running"
    manifest["updated_at"] = _now()
    _write_json_atomic(manifest_path, manifest)
    failures = 0
    try:
        for position, (variant, seed) in enumerate(matrix, start=1):
            run_id = f"{variant}__seed{seed}"
            run_output = runs_dir / variant / f"seed_{seed}"
            log_path = logs_dir / f"{run_id}.log"
            existing = manifest["runs"].get(run_id)
            if (
                not args.force
                and existing
                and existing.get("status") == "completed"
                and _completed_output_is_valid(run_output, variant, seed, identity)
            ):
                print(
                    f"[{position}/{len(matrix)}] skip completed {run_id}",
                    flush=True,
                )
                continue

            command = _training_command(args, variant, seed, run_output)
            run_record = {
                "run_id": run_id,
                "variant": variant,
                "seed": seed,
                "status": "running",
                "started_at": _now(),
                "completed_at": None,
                "return_code": None,
                "output_dir": _display_path(run_output),
                "log": _display_path(log_path),
                "command": command,
            }
            manifest["runs"][run_id] = run_record
            manifest["updated_at"] = _now()
            _write_json_atomic(manifest_path, manifest)
            print(
                f"[{position}/{len(matrix)}] start {run_id} "
                f"log={_display_path(log_path)}",
                flush=True,
            )
            return_code = _run_and_tee(command, log_path)
            run_record["return_code"] = return_code
            run_record["completed_at"] = _now()
            if return_code == 0 and _completed_output_is_valid(
                run_output, variant, seed, identity
            ):
                run_record["status"] = "completed"
                print(
                    f"[{position}/{len(matrix)}] completed {run_id}",
                    flush=True,
                )
            else:
                run_record["status"] = "failed"
                failures += 1
                print(
                    f"[{position}/{len(matrix)}] failed {run_id} "
                    f"return_code={return_code}",
                    flush=True,
                )
            manifest["updated_at"] = _now()
            _write_json_atomic(manifest_path, manifest)
            _write_matrix_summary(output_root, manifest)
            if failures and args.stop_on_error:
                break
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["updated_at"] = _now()
        _write_json_atomic(manifest_path, manifest)
        print("ablation interrupted; completed runs will be skipped on restart")
        raise SystemExit(130)
    finally:
        lock_path.unlink(missing_ok=True)

    completed = sum(
        record.get("status") == "completed"
        for record in manifest["runs"].values()
    )
    manifest["status"] = (
        "complete"
        if completed == len(matrix) and failures == 0
        else "complete_with_failures"
        if failures
        else "in_progress"
    )
    manifest["updated_at"] = _now()
    _write_json_atomic(manifest_path, manifest)
    _write_matrix_summary(output_root, manifest)
    print(
        f"ablation status={manifest['status']} completed={completed}/{len(matrix)} "
        f"failures={failures}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


def _training_command(
    args: argparse.Namespace,
    variant: str,
    seed: int,
    output_dir: Path,
) -> list[str]:
    prototype_batch_size = _prototype_batch_size(args, variant)
    return [
        # Do not resolve this path: on Linux a venv's python is commonly a
        # symlink, and resolving it bypasses the venv's site-packages.
        os.path.abspath(args.python),
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--seeds",
        str(seed),
        "--hidden-dim",
        str(args.hidden_dim),
        "--layers",
        str(args.layers),
        "--prototype-batch-size",
        str(prototype_batch_size),
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--variant",
        variant,
        "--randomization-seed",
        str(args.randomization_seed),
    ]


def _run_and_tee(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"\n[{_now()}] command={json.dumps(command, ensure_ascii=False)}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        return process.wait()


def _load_or_create_manifest(
    path: Path,
    identity: dict,
    force: bool,
) -> dict:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        existing_identity = manifest.get("identity")
        if existing_identity != identity and not force:
            raise SystemExit(
                f"existing ablation identity differs: {path}; "
                "use another --output-root or pass --force"
            )
        if existing_identity == identity:
            return manifest
    return {
        "schema": MANIFEST_SCHEMA,
        "identity": identity,
        "created_at": _now(),
        "updated_at": _now(),
        "status": "planned",
        "runs": {},
    }


def _completed_output_is_valid(
    output_dir: Path,
    variant: str,
    seed: int,
    identity: dict,
) -> bool:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("variant") == variant
        and summary.get("seeds") == [seed]
        and summary.get("dataset_sha256") == identity["dataset_sha256"]
        and summary.get("candidate_sha256") == identity["candidate_sha256"]
        and summary.get("config", {}).get("hidden_dim")
        == identity["training"]["hidden_dim"]
        and summary.get("config", {}).get("propagation_layers")
        == identity["training"]["layers"]
        and summary.get("config", {}).get("prototype_batch_size")
        == _identity_prototype_batch_size(identity, variant)
        and summary.get("config", {}).get("max_epochs")
        == identity["training"]["max_epochs"]
        and summary.get("config", {}).get("patience")
        == identity["training"]["patience"]
        and summary.get("config", {}).get("randomization_seed")
        == identity["training"]["randomization_seed"]
    )


def _prototype_batch_size(args: argparse.Namespace, variant: str) -> int:
    if variant in {"undirected", "graphsage"}:
        return min(args.prototype_batch_size, args.expanded_graph_batch_size)
    return args.prototype_batch_size


def _identity_prototype_batch_size(identity: dict, variant: str) -> int:
    default = int(identity["training"]["prototype_batch_size"])
    if variant in {"undirected", "graphsage"}:
        return min(
            default,
            int(identity["training"]["expanded_graph_batch_size"]),
        )
    return default


def _write_matrix_summary(output_root: Path, manifest: dict) -> None:
    rows: list[dict[str, object]] = []
    for record in manifest["runs"].values():
        row: dict[str, object] = {
            "variant": record["variant"],
            "seed": record["seed"],
            "status": record["status"],
            "return_code": record.get("return_code"),
            "log": record["log"],
        }
        output_dir = ROOT_DIR / record["output_dir"]
        summary_path = output_dir / "summary.json"
        if record["status"] == "completed" and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            run = summary["runs"][0]
            holdout = run["metrics"]["holdout"]
            row.update(
                {
                    "best_epoch": run["best_epoch"],
                    "epochs_ran": run["epochs_ran"],
                    "training_seconds": run["training_seconds"],
                    "holdout_mae": holdout["mae"],
                    "holdout_spearman": holdout["spearman"],
                    "holdout_ndcg_at_k": holdout["ndcg_at_k"],
                    "holdout_top_k_mean_gain": holdout["top_k_mean_gain"],
                }
            )
        rows.append(row)
    rows.extend(_base_rows())
    rows.sort(key=lambda row: (str(row["variant"]), int(row["seed"])))
    csv_path = output_root / "ablation_summary.csv"
    fieldnames = (
        "variant",
        "seed",
        "status",
        "return_code",
        "best_epoch",
        "epochs_ran",
        "training_seconds",
        "holdout_mae",
        "holdout_spearman",
        "holdout_ndcg_at_k",
        "holdout_top_k_mean_gain",
        "log",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "report.md").write_text(
        _render_matrix_report(rows, manifest),
        encoding="utf-8",
    )


def _base_rows() -> list[dict[str, object]]:
    if not BASE_SUMMARY.exists():
        return []
    summary = json.loads(BASE_SUMMARY.read_text(encoding="utf-8"))
    rows = []
    for run in summary.get("runs", []):
        holdout = run["metrics"]["holdout"]
        rows.append(
            {
                "variant": "base",
                "seed": run["seed"],
                "status": "reference",
                "return_code": 0,
                "best_epoch": run["best_epoch"],
                "epochs_ran": run["epochs_ran"],
                "training_seconds": run["training_seconds"],
                "holdout_mae": holdout["mae"],
                "holdout_spearman": holdout["spearman"],
                "holdout_ndcg_at_k": holdout["ndcg_at_k"],
                "holdout_top_k_mean_gain": holdout["top_k_mean_gain"],
                "log": "",
            }
        )
    return rows


def _render_matrix_report(rows: list[dict[str, object]], manifest: dict) -> str:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if "holdout_spearman" in row:
            grouped.setdefault(str(row["variant"]), []).append(row)
    lines = [
        "# NBFNet 核心消融汇总",
        "",
        f"- 实验 ID：`{manifest['identity']['experiment_id']}`",
        f"- 模式：`{manifest['identity']['mode']}`",
        f"- 状态：`{manifest.get('status', 'in_progress')}`",
        "",
        "| 变体 | 完成种子 | Spearman | NDCG@K | Top-K 收益 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in sorted(grouped):
        variant_rows = grouped[variant]
        spearman = [float(row["holdout_spearman"]) for row in variant_rows]
        ndcg = [float(row["holdout_ndcg_at_k"]) for row in variant_rows]
        top_gain = [float(row["holdout_top_k_mean_gain"]) for row in variant_rows]
        lines.append(
            f"| {variant} | {len(variant_rows)} | "
            f"{statistics.fmean(spearman):.4f} ± {statistics.pstdev(spearman):.4f} | "
            f"{statistics.fmean(ndcg):.4f} ± {statistics.pstdev(ndcg):.4f} | "
            f"{statistics.fmean(top_gain):.3f} ± {statistics.pstdev(top_gain):.3f} |"
        )
    lines.extend(
        (
            "",
            "该表会在每个子实验结束后重写；失败与运行中状态见 `manifest.json` 和日志目录。",
            "",
        )
    )
    return "\n".join(lines)


def _acquire_lock(path: Path) -> None:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            existing_pid = int(payload["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise SystemExit(
                f"invalid ablation runner lock: {path}; inspect it before removal"
            ) from error
        if _process_is_alive(existing_pid):
            raise SystemExit(
                f"another ablation runner is active: pid={existing_pid} lock={path}"
            )
        path.unlink()
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise SystemExit(
            f"another ablation runner may be active: {path}; "
            "remove the lock only after confirming no process is running"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(
            json.dumps({"pid": os.getpid(), "created_at": _now()}, ensure_ascii=False)
            + "\n"
        )


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(RUNNER_VARIANTS))
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("--variants must contain unique variant names")
    if unknown:
        raise ValueError(f"unsupported runner variants: {', '.join(unknown)}")
    return variants


def _parse_int_list(value: str, option: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{option} must be a comma-separated integer list") from error
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{option} must contain unique integers")
    return values


def _digest(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
