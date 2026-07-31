"""Run one preregistered G-line Z0-residual structure screening experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT_DIR / "scripts" / "train_demand_field_nbfnet.py"
DATASET_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
Z0_SUMMARY = (
    ROOT_DIR
    / "results"
    / "gnn_v2"
    / "nbfnet_propagation"
    / "train_free_baselines"
    / "summary.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT_DIR
    / "results"
    / "gnn_v2"
    / "nbfnet_propagation"
    / "gradient_structure_screening"
)
MANIFEST_SCHEMA = "aic.gnn_v2.gradient_structure_screening.v1"
TRAINING_SCHEMA = "aic.gnn_v2.od_conditioned_bidirectional_nbfnet.v4"
STRUCTURES = ("g2", "g3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", choices=STRUCTURES, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--head-warmup-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "reduce_on_plateau"),
        default="none",
    )
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.3)
    parser.add_argument("--lr-scheduler-patience", type=int, default=3)
    parser.add_argument("--lr-scheduler-threshold", type=float, default=1.0e-4)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=5.0e-4)
    parser.add_argument("--prototype-batch-size", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    output_root = args.output_root.resolve()
    run_output = output_root / "runs" / args.structure / f"seed_{args.seed}"
    log_path = output_root / "logs" / f"{args.structure}_seed{args.seed}.log"
    manifest_path = output_root / "manifests" / f"{args.structure}.json"
    lock_path = output_root / f".{args.structure}.lock"
    dataset = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    identity = {
        "schema": MANIFEST_SCHEMA,
        "structure": args.structure,
        "seed": args.seed,
        "dataset_sha256": dataset["dataset_sha256"],
        "variant": "propagation_doubling",
        "layers": 32,
        "hidden_dim": 32,
        "residual_scale": 0.01,
        "training_objective": "rank_first",
        "fixed_prior": "z0",
        "precision": "fp32",
        "learning_rate": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_threshold": args.lr_scheduler_threshold,
        "lr_scheduler_min_lr": args.lr_scheduler_min_lr,
        "weight_decay": 0.0,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "head_warmup_steps": args.head_warmup_steps,
        "prototype_batch_size": args.prototype_batch_size,
        "holdout_withheld": True,
    }
    command = _training_command(args, run_output)
    print(
        f"G-line structure={args.structure} seed={args.seed} output={run_output}",
        flush=True,
    )
    if args.dry_run:
        print(" ".join(command))
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(manifest_path, identity, force=args.force)
    if (
        not args.force
        and manifest.get("status") == "complete"
        and _completed_output_is_valid(run_output, identity)
    ):
        print(f"skip completed {args.structure}", flush=True)
        return
    _acquire_lock(lock_path)
    manifest.update(
        {
            "status": "running",
            "started_at": _now(),
            "completed_at": None,
            "command": command,
            "output_dir": _display_path(run_output),
            "log": _display_path(log_path),
        }
    )
    _write_json_atomic(manifest_path, manifest)
    try:
        return_code = _run_and_tee(command, log_path)
    finally:
        lock_path.unlink(missing_ok=True)
    manifest["return_code"] = return_code
    manifest["completed_at"] = _now()
    if return_code == 0 and _completed_output_is_valid(run_output, identity):
        summary = json.loads((run_output / "summary.json").read_text(encoding="utf-8"))
        manifest["gate"] = _performance_gate(
            summary,
            json.loads(Z0_SUMMARY.read_text(encoding="utf-8")),
        )
        manifest["status"] = "complete"
    else:
        manifest["status"] = "failed"
    _write_json_atomic(manifest_path, manifest)
    print(
        f"G-line status={manifest['status']} gate={manifest.get('gate', {}).get('passed')}",
        flush=True,
    )
    if manifest["status"] == "failed":
        raise SystemExit(1)


def _training_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        os.path.abspath(args.python),
        str(TRAIN_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--seeds",
        str(args.seed),
        "--hidden-dim",
        "32",
        "--layers",
        "32",
        "--prototype-batch-size",
        str(args.prototype_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--lr-scheduler",
        args.lr_scheduler,
        "--lr-scheduler-factor",
        str(args.lr_scheduler_factor),
        "--lr-scheduler-patience",
        str(args.lr_scheduler_patience),
        "--lr-scheduler-threshold",
        str(args.lr_scheduler_threshold),
        "--lr-scheduler-min-lr",
        str(args.lr_scheduler_min_lr),
        "--weight-decay",
        "0",
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--training-objective",
        "rank_first",
        "--fixed-prior",
        "z0",
        "--precision",
        "fp32",
        "--variant",
        "propagation_doubling",
        "--propagation-structure",
        args.structure,
        "--propagation-residual-scale",
        "0.01",
        "--head-warmup-steps",
        str(args.head_warmup_steps),
        "--withhold-holdout",
    ]


def _performance_gate(summary: dict, z0_summary: dict) -> dict[str, object]:
    run = summary["runs"][0]
    validation = run["metrics"]["validation"]
    z0 = z0_summary["z0_metrics"]["validation"]
    spearman_delta = float(validation["spearman"] - z0["spearman"])
    ndcg5_delta = float(
        validation["ranking_at_k"]["5"]["ndcg"]
        - z0["ranking_at_k"]["5"]["ndcg"]
    )
    diagnostics = run["diagnostics"]
    epochs_ran = int(run["epochs_ran"])
    effective_steps = int(diagnostics["effective_optimizer_steps"])
    skipped_steps = int(diagnostics["skipped_optimizer_steps"])
    effective_ratio = effective_steps / epochs_ran if epochs_ran else 0.0
    numerical_checks = {
        "effective_step_ratio": effective_ratio >= 0.95,
        "no_skipped_steps": skipped_steps == 0,
        "finite_gradient_norms": (
            int(diagnostics["nonfinite_gradient_norm_steps"]) == 0
        ),
        "no_five_step_zero_gradient_run": (
            int(diagnostics["maximum_consecutive_zero_gradient_steps_after_clip"])
            < 5
        ),
        "validation_loss_recovers_after_doubling": (
            int(diagnostics["unrecovered_validation_loss_doublings"]) == 0
        ),
        "backbone_updated": (
            int(diagnostics["best_checkpoint_backbone_effective_step_count"]) > 0
        ),
    }
    spearman_channel = spearman_delta >= 0.003 and ndcg5_delta >= -0.005
    ndcg_channel = ndcg5_delta >= 0.010 and spearman_delta >= -0.003
    return {
        "passed": all(numerical_checks.values()) and (
            spearman_channel or ndcg_channel
        ),
        "numerical_passed": all(numerical_checks.values()),
        "performance_passed": spearman_channel or ndcg_channel,
        "checks": numerical_checks,
        "channels": {
            "spearman_channel": spearman_channel,
            "ndcg5_channel": ndcg_channel,
        },
        "metrics": {
            "validation_spearman": validation["spearman"],
            "z0_validation_spearman": z0["spearman"],
            "spearman_delta": spearman_delta,
            "validation_ndcg5": validation["ranking_at_k"]["5"]["ndcg"],
            "z0_validation_ndcg5": z0["ranking_at_k"]["5"]["ndcg"],
            "ndcg5_delta": ndcg5_delta,
            "effective_step_ratio": effective_ratio,
            "best_epoch": run["best_epoch"],
        },
    }


def _completed_output_is_valid(
    output_dir: Path,
    identity: dict[str, object],
) -> bool:
    path = output_dir / "summary.json"
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    protocol = summary.get("training_protocol", {})
    scheduler = protocol.get("lr_scheduler", {})
    config = summary.get("config", {})
    return (
        summary.get("schema") == TRAINING_SCHEMA
        and summary.get("dataset_sha256") == identity["dataset_sha256"]
        and summary.get("training_objective") == "rank_first"
        and summary.get("fixed_prior", {}).get("name") == "z0"
        and config.get("propagation_structure") == identity["structure"]
        and config.get("propagation_layers") == 32
        and protocol.get("head_warmup_steps") == identity["head_warmup_steps"]
        and scheduler.get("name") == identity["lr_scheduler"]
        and scheduler.get("factor") == identity["lr_scheduler_factor"]
        and scheduler.get("patience") == identity["lr_scheduler_patience"]
        and scheduler.get("threshold") == identity["lr_scheduler_threshold"]
        and scheduler.get("min_lr") == identity["lr_scheduler_min_lr"]
        and protocol.get("holdout_withheld") is True
        and "holdout" not in summary.get("aggregate", {})
    )


def _load_or_create_manifest(
    path: Path,
    identity: dict[str, object],
    *,
    force: bool,
) -> dict[str, object]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("identity") == identity and not force:
            return existing
        if not force:
            raise SystemExit(f"manifest identity mismatch: {path}; use --force")
    return {
        "schema": MANIFEST_SCHEMA,
        "identity": identity,
        "status": "created",
        "created_at": _now(),
    }


def _run_and_tee(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def _acquire_lock(path: Path) -> None:
    if path.exists():
        try:
            pid = int(json.loads(path.read_text(encoding="utf-8"))["pid"])
            os.kill(pid, 0)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        else:
            raise SystemExit(f"runner already active with pid={pid}: {path}")
    path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")


def _validate_args(args: argparse.Namespace) -> None:
    numeric = (
        args.max_epochs,
        args.patience,
        args.head_warmup_steps,
        args.learning_rate,
        args.prototype_batch_size,
        args.lr_scheduler_factor,
        args.lr_scheduler_threshold,
        args.lr_scheduler_min_lr,
    )
    if any(value <= 0 for value in numeric):
        raise SystemExit("G-line numeric settings must be positive")
    if args.head_warmup_steps >= args.max_epochs:
        raise SystemExit("head warmup must leave at least one backbone update")
    if args.lr_scheduler != "none":
        if not 0.0 < args.lr_scheduler_factor < 1.0:
            raise SystemExit("scheduler factor must be in (0, 1)")
        if args.lr_scheduler_patience < 0:
            raise SystemExit("scheduler patience must be non-negative")
        if args.lr_scheduler_min_lr > args.learning_rate:
            raise SystemExit("scheduler min lr cannot exceed learning rate")


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
