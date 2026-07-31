"""Run the frozen G3 plateau-scheduler S3/S4 seeds sequentially on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
STRUCTURE_RUNNER = ROOT_DIR / "scripts" / "run_gradient_structure_screening.py"
DEFAULT_OUTPUT_ROOT = (
    ROOT_DIR
    / "results"
    / "gnn_v2"
    / "nbfnet_propagation"
    / "gradient_scheduler_confirmation"
)
SCHEMA = "aic.gnn_v2.gradient_scheduler_confirmation.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    output_root = args.output_root.resolve()
    identity = {
        "schema": SCHEMA,
        "seeds": seeds,
        "structure": "g3",
        "learning_rate": 5.0e-3,
        "max_epochs": 40,
        "head_warmup_steps": 8,
        "lr_scheduler": "reduce_on_plateau",
        "scheduler_monitor": "validation_spearman",
        "scheduler_factor": 0.3,
        "scheduler_patience": 3,
        "scheduler_threshold": 1.0e-4,
        "scheduler_min_lr": 5.0e-4,
        "holdout_withheld": True,
    }
    commands = [_seed_command(args.python, output_root, seed) for seed in seeds]
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"runner_{'_'.join(map(str, seeds))}.json"
    lock_path = output_root / f".runner_{'_'.join(map(str, seeds))}.lock"
    manifest = _load_manifest(manifest_path, identity)
    if manifest.get("status") == "complete":
        print(f"skip completed scheduler seeds={seeds}", flush=True)
        return
    _acquire_lock(lock_path)
    manifest.update(
        {
            "status": "running",
            "started_at": _now(),
            "completed_at": None,
            "commands": commands,
            "runs": [],
        }
    )
    _write_json_atomic(manifest_path, manifest)
    try:
        for seed, command in zip(seeds, commands):
            print(f"scheduler confirmation seed={seed} starting", flush=True)
            return_code = subprocess.run(command, cwd=ROOT_DIR, check=False).returncode
            manifest["runs"].append(
                {"seed": seed, "return_code": return_code, "completed_at": _now()}
            )
            _write_json_atomic(manifest_path, manifest)
            if return_code != 0:
                manifest["status"] = "failed"
                break
        else:
            manifest["status"] = "complete"
    finally:
        lock_path.unlink(missing_ok=True)
    manifest["completed_at"] = _now()
    _write_json_atomic(manifest_path, manifest)
    print(
        f"scheduler confirmation status={manifest['status']} seeds={seeds}",
        flush=True,
    )
    if manifest["status"] != "complete":
        raise SystemExit(1)


def _seed_command(python: Path, output_root: Path, seed: int) -> list[str]:
    return [
        os.path.abspath(python),
        str(STRUCTURE_RUNNER),
        "--python",
        os.path.abspath(python),
        "--structure",
        "g3",
        "--seed",
        str(seed),
        "--learning-rate",
        "0.005",
        "--max-epochs",
        "40",
        "--patience",
        "40",
        "--head-warmup-steps",
        "8",
        "--lr-scheduler",
        "reduce_on_plateau",
        "--lr-scheduler-factor",
        "0.3",
        "--lr-scheduler-patience",
        "3",
        "--lr-scheduler-threshold",
        "0.0001",
        "--lr-scheduler-min-lr",
        "0.0005",
        "--output-root",
        str(output_root / f"seed_{seed}"),
    ]


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise SystemExit("seeds must be comma-separated integers") from error
    if not seeds or len(seeds) != len(set(seeds)):
        raise SystemExit("seeds must contain unique integers")
    return seeds


def _load_manifest(path: Path, identity: dict[str, object]) -> dict[str, object]:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise SystemExit(f"manifest identity mismatch: {path}")
        return manifest
    return {
        "schema": SCHEMA,
        "identity": identity,
        "status": "created",
        "created_at": _now(),
    }


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


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
