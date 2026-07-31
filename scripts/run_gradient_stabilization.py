"""Run the preregistered S1 CUDA gradient-stabilization gate with resume support."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DIAGNOSTIC_SCRIPT = ROOT_DIR / "scripts" / "diagnose_demand_field_gradients.py"
DATASET_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
DEFAULT_OUTPUT_ROOT = (
    ROOT_DIR
    / "results"
    / "gnn_v2"
    / "nbfnet_propagation"
    / "gradient_stabilization"
)
MANIFEST_SCHEMA = "aic.gnn_v2.gradient_stabilization_runner.v1"
DIAGNOSTIC_SCHEMA = "aic.gnn_v2.gradient_diagnostics.v2"
STRUCTURES = ("g0", "g1", "g2", "g3")
DEFAULT_DEPTHS = (8, 16, 32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run S1 snapshots and three-step gates on one CUDA GPU."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--structures", default=",".join(STRUCTURES))
    parser.add_argument("--depths", default=",".join(map(str, DEFAULT_DEPTHS)))
    parser.add_argument("--shard-name", default="all")
    parser.add_argument("--variant", default="propagation_doubling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--prototype-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--residual-scale", type=float, default=0.01)
    parser.add_argument("--optimizer-steps", type=int, default=3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp64-norm-limit", type=float, default=1.0e4)
    parser.add_argument("--clip-coefficient-min", type=float, default=1.0e-4)
    parser.add_argument("--hook-amplification-limit", type=float, default=1.0e6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structures = _parse_structures(args.structures)
    depths = _parse_depths(args.depths)
    _validate_args(args, depths)
    output_root = args.output_root.resolve()
    logs_dir = output_root / "logs"
    manifests_dir = output_root / "manifests"
    manifest_path = manifests_dir / f"{args.shard_name}.json"
    summary_path = output_root / f"summary_{args.shard_name}.csv"
    report_path = output_root / f"report_{args.shard_name}.md"
    lock_path = output_root / f".{args.shard_name}.lock"

    dataset = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    identity = {
        "schema": MANIFEST_SCHEMA,
        "shard_name": args.shard_name,
        "structures": structures,
        "depths": depths,
        "dataset_sha256": dataset["dataset_sha256"],
        "variant": args.variant,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "prototype_batch_size": args.prototype_batch_size,
        "learning_rate": args.learning_rate,
        "residual_scale": args.residual_scale,
        "optimizer_steps": args.optimizer_steps,
        "max_grad_norm": args.max_grad_norm,
        "gates": {
            "fp64_norm_limit": args.fp64_norm_limit,
            "clip_coefficient_min": args.clip_coefficient_min,
            "hook_amplification_limit": args.hook_amplification_limit,
        },
    }
    identity["experiment_id"] = _digest(identity)[:16]
    commands = _planned_commands(args, structures, depths, output_root)
    print(
        f"S1 experiment={identity['experiment_id']} shard={args.shard_name} "
        f"structures={','.join(structures)} depths={','.join(map(str, depths))}",
        flush=True,
    )
    if args.dry_run:
        for key, command, _output, _log in commands:
            print(f"{key}: {' '.join(command)}")
        return

    logs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(manifest_path, identity, force=args.force)
    _acquire_lock(lock_path)
    manifest["status"] = "running"
    _write_json_atomic(manifest_path, manifest)
    failures = 0
    try:
        for structure in structures:
            for depth in depths:
                key = _run_key(structure, depth, "snapshot")
                command, run_output, log_path = _command_for(
                    args, structure, depth, "snapshot", output_root
                )
                record = _execute_or_reuse(
                    key=key,
                    command=command,
                    output_dir=run_output,
                    log_path=log_path,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    force=args.force,
                )
                if record["status"] == "completed":
                    summary = _load_summary(run_output / "summary.json")
                    record["gate"] = _snapshot_gate(
                        summary,
                        depth=depth,
                        fp64_norm_limit=args.fp64_norm_limit,
                        clip_coefficient_min=args.clip_coefficient_min,
                        hook_amplification_limit=args.hook_amplification_limit,
                    )
                else:
                    failures += 1
                _write_json_atomic(manifest_path, manifest)
                _write_summary(summary_path, report_path, manifest)
                if failures and args.stop_on_error:
                    break
            if failures and args.stop_on_error:
                break

            depth = max(depths)
            snapshot = manifest["runs"].get(_run_key(structure, depth, "snapshot"), {})
            if not snapshot.get("gate", {}).get("passed", False):
                continue
            key = _run_key(structure, depth, "optimizer_steps")
            command, run_output, log_path = _command_for(
                args, structure, depth, "optimizer_steps", output_root
            )
            record = _execute_or_reuse(
                key=key,
                command=command,
                output_dir=run_output,
                log_path=log_path,
                manifest=manifest,
                manifest_path=manifest_path,
                force=args.force,
            )
            if record["status"] == "completed":
                record["gate"] = _optimizer_steps_gate(
                    _load_summary(run_output / "summary.json"),
                    requested_steps=args.optimizer_steps,
                )
            else:
                failures += 1
            _write_json_atomic(manifest_path, manifest)
            _write_summary(summary_path, report_path, manifest)
    finally:
        lock_path.unlink(missing_ok=True)

    manifest["status"] = "complete_with_failures" if failures else "complete"
    manifest["updated_at"] = _now()
    _write_json_atomic(manifest_path, manifest)
    _write_summary(summary_path, report_path, manifest)
    print(f"S1 status={manifest['status']} summary={summary_path}", flush=True)
    if failures:
        raise SystemExit(1)


def _planned_commands(
    args: argparse.Namespace,
    structures: list[str],
    depths: list[int],
    output_root: Path,
) -> list[tuple[str, list[str], Path, Path]]:
    commands: list[tuple[str, list[str], Path, Path]] = []
    for structure in structures:
        for depth in depths:
            command, output, log = _command_for(
                args, structure, depth, "snapshot", output_root
            )
            commands.append((_run_key(structure, depth, "snapshot"), command, output, log))
        depth = max(depths)
        command, output, log = _command_for(
            args, structure, depth, "optimizer_steps", output_root
        )
        commands.append(
            (_run_key(structure, depth, "optimizer_steps"), command, output, log)
        )
    return commands


def _command_for(
    args: argparse.Namespace,
    structure: str,
    depth: int,
    mode: str,
    output_root: Path,
) -> tuple[list[str], Path, Path]:
    output = output_root / "runs" / structure / f"depth_{depth:02d}" / mode
    log = output_root / "logs" / f"{_run_key(structure, depth, mode)}.log"
    command = [
        os.path.abspath(args.python),
        str(DIAGNOSTIC_SCRIPT),
        "--output-dir",
        str(output),
        "--mode",
        mode,
        "--variant",
        args.variant,
        "--propagation-structure",
        structure,
        "--propagation-residual-scale",
        str(args.residual_scale),
        "--seed",
        str(args.seed),
        "--hidden-dim",
        str(args.hidden_dim),
        "--layers",
        str(depth),
        "--prototype-batch-size",
        str(args.prototype_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--max-grad-norm",
        str(args.max_grad_norm),
        "--hook-depths",
        "doubling",
    ]
    if mode == "optimizer_steps":
        command.extend(("--optimizer-steps", str(args.optimizer_steps)))
    return command, output, log


def _execute_or_reuse(
    *,
    key: str,
    command: list[str],
    output_dir: Path,
    log_path: Path,
    manifest: dict[str, object],
    manifest_path: Path,
    force: bool,
) -> dict[str, object]:
    existing = manifest["runs"].get(key)
    if not force and existing and existing.get("status") == "completed":
        if _completed_output_is_valid(output_dir, command):
            print(f"skip completed {key}", flush=True)
            return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "run_id": key,
        "status": "running",
        "started_at": _now(),
        "completed_at": None,
        "return_code": None,
        "output_dir": _display_path(output_dir),
        "log": _display_path(log_path),
        "command": command,
    }
    manifest["runs"][key] = record
    manifest["updated_at"] = _now()
    _write_json_atomic(manifest_path, manifest)
    print(f"start {key} log={_display_path(log_path)}", flush=True)
    return_code = _run_and_tee(command, log_path)
    record["return_code"] = return_code
    record["completed_at"] = _now()
    record["status"] = (
        "completed"
        if return_code == 0 and _completed_output_is_valid(output_dir, command)
        else "failed"
    )
    print(f"{record['status']} {key}", flush=True)
    return record


def _snapshot_gate(
    summary: dict[str, object],
    *,
    depth: int,
    fp64_norm_limit: float,
    clip_coefficient_min: float,
    hook_amplification_limit: float,
) -> dict[str, object]:
    snapshot = summary["result"]["snapshot"]
    gradients = snapshot["parameter_gradients_unscaled"]["summary"]
    hooks = snapshot["tensor_hooks"]
    amplification = _hook_amplification(hooks, depth)
    checks = {
        "gradient_elements_finite": bool(gradients["all_gradient_elements_finite"]),
        "raw_fp32_norm_finite": bool(gradients["raw_fp32_norm_is_finite"]),
        "fp64_norm_within_limit": (
            float(gradients["finite_l2_norm_fp64"]) <= fp64_norm_limit
        ),
        "clip_coefficient_within_limit": (
            float(snapshot["clipping"]["coefficient"]) >= clip_coefficient_min
        ),
        "hooks_finite": all(bool(row["all_finite"]) for row in hooks.values()),
        "hook_amplification_within_limit": amplification <= hook_amplification_limit,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "raw_l2_norm_fp32": gradients["raw_l2_norm_fp32"],
            "finite_l2_norm_fp64": gradients["finite_l2_norm_fp64"],
            "clip_coefficient": snapshot["clipping"]["coefficient"],
            "hook_amplification": amplification,
        },
    }


def _optimizer_steps_gate(
    summary: dict[str, object],
    *,
    requested_steps: int,
) -> dict[str, object]:
    result = summary["result"]
    rows = result["history"][1:]
    norms = [
        float(row["parameter_gradients_unscaled"]["summary"]["finite_l2_norm_fp64"])
        for row in rows
    ]
    growth_ok = all(
        current <= previous * 100.0
        for previous, current in zip(norms, norms[1:])
        if previous > 0.0
    )
    checks = {
        "all_steps_applied": (
            len(rows) == requested_steps
            and all(row.get("optimizer_step") == "applied" for row in rows)
        ),
        "gradient_elements_finite": all(
            row["parameter_gradients_unscaled"]["summary"][
                "all_gradient_elements_finite"
            ]
            for row in rows
        ),
        "raw_fp32_norm_finite": all(
            row["parameter_gradients_unscaled"]["summary"][
                "raw_fp32_norm_is_finite"
            ]
            for row in rows
        ),
        "clipped_gradient_nonzero": all(
            float(row.get("parameter_gradients_clipped", {}).get("summary", {}).get(
                "finite_l2_norm_fp64", 0.0
            ))
            > 0.0
            for row in rows
        ),
        "parameter_delta_nonzero": all(
            float(row.get("parameter_delta_norm", 0.0)) > 0.0 for row in rows
        ),
        "gradient_growth_within_limit": growth_ok,
        "prediction_not_collapsed": all(
            bool(row["score_prediction"]["all_finite"])
            and float(row["score_prediction"]["std"]) >= 1.0e-8
            for row in rows
        ),
    }
    return {
        "passed": bool(rows) and all(checks.values()),
        "checks": checks,
        "metrics": {
            "applied_steps": result["applied_steps"],
            "fp64_gradient_norms": norms,
            "clip_coefficients": [row["clipping"]["coefficient"] for row in rows],
            "prediction_std": [row["score_prediction"]["std"] for row in rows],
            "parameter_delta_norms": [
                row.get("parameter_delta_norm", 0.0) for row in rows
            ],
        },
    }


def _hook_amplification(hooks: dict[str, dict[str, object]], depth: int) -> float:
    ratios: list[float] = []
    for direction in ("origin", "destination"):
        early = hooks.get(f"{direction}.depth_01.gradient")
        deep = hooks.get(f"{direction}.depth_{depth:02d}.gradient")
        if early is None or deep is None:
            return math.inf
        numerator = float(early["maximum_absolute_finite_value"])
        denominator = float(deep["maximum_absolute_finite_value"])
        if denominator == 0.0:
            ratios.append(1.0 if numerator == 0.0 else math.inf)
        else:
            ratios.append(numerator / denominator)
    return max(ratios)


def _completed_output_is_valid(output_dir: Path, command: list[str]) -> bool:
    path = output_dir / "summary.json"
    if not path.exists():
        return False
    try:
        summary = _load_summary(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    structure = command[command.index("--propagation-structure") + 1]
    layers = int(command[command.index("--layers") + 1])
    mode = command[command.index("--mode") + 1]
    return (
        summary.get("schema") == DIAGNOSTIC_SCHEMA
        and summary.get("mode") == mode
        and summary.get("config", {}).get("propagation_structure") == structure
        and summary.get("config", {}).get("propagation_layers") == layers
    )


def _load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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
            raise SystemExit(
                f"manifest identity mismatch: {path}; use --force to replace this shard"
            )
    return {
        "schema": MANIFEST_SCHEMA,
        "identity": identity,
        "status": "created",
        "created_at": _now(),
        "updated_at": _now(),
        "runs": {},
    }


def _write_summary(
    csv_path: Path,
    report_path: Path,
    manifest: dict[str, object],
) -> None:
    rows: list[dict[str, object]] = []
    for run_id, record in sorted(manifest["runs"].items()):
        gate = record.get("gate", {})
        metrics = gate.get("metrics", {})
        rows.append(
            {
                "run_id": run_id,
                "status": record.get("status"),
                "gate_passed": gate.get("passed", ""),
                "fp64_gradient_norm": metrics.get("finite_l2_norm_fp64", ""),
                "clip_coefficient": metrics.get("clip_coefficient", ""),
                "hook_amplification": metrics.get("hook_amplification", ""),
                "applied_steps": metrics.get("applied_steps", ""),
            }
        )
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# S1 梯度稳定化门汇总",
        "",
        f"- 状态：`{manifest['status']}`",
        f"- 实验 ID：`{manifest['identity']['experiment_id']}`",
        "",
        "| 运行 | 进程状态 | 数值门 | FP64 范数 | 裁剪系数 | Hook 放大 | 有效步 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_id']} | {row['status']} | {row['gate_passed']} | "
            f"{row['fp64_gradient_norm']} | {row['clip_coefficient']} | "
            f"{row['hook_amplification']} | {row['applied_steps']} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _parse_structures(value: str) -> list[str]:
    structures = [item.strip() for item in value.split(",") if item.strip()]
    if not structures or len(structures) != len(set(structures)):
        raise ValueError("structures must be non-empty and unique")
    unknown = sorted(set(structures) - set(STRUCTURES))
    if unknown:
        raise ValueError(f"unknown structures: {', '.join(unknown)}")
    return structures


def _parse_depths(value: str) -> list[int]:
    depths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not depths or len(depths) != len(set(depths)) or any(depth <= 0 for depth in depths):
        raise ValueError("depths must be unique positive integers")
    return sorted(depths)


def _validate_args(args: argparse.Namespace, depths: list[int]) -> None:
    if max(depths) != 32:
        raise SystemExit("S1 requires a maximum depth of 32")
    positive = (
        args.hidden_dim,
        args.prototype_batch_size,
        args.learning_rate,
        args.residual_scale,
        args.optimizer_steps,
        args.max_grad_norm,
        args.fp64_norm_limit,
        args.clip_coefficient_min,
        args.hook_amplification_limit,
    )
    if any(value <= 0 for value in positive):
        raise SystemExit("S1 numeric settings must be positive")
    if not args.shard_name or any(character in args.shard_name for character in "/\\"):
        raise SystemExit("shard-name must be a simple non-empty name")


def _run_key(structure: str, depth: int, mode: str) -> str:
    return f"{structure}_depth{depth:02d}_{mode}"


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
