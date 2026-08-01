"""Freeze global and budget-aware G4 checkpoints using validation metrics only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


SELECTION_FILES = {
    "global_spearman": "model_global_spearman.pt",
    "budget_safe_spearman": "model_budget_safe_spearman.pt",
    "topgain18_safe": "model_topgain18_safe.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-dir",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each seed directory containing validation_snapshot_catalog.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for seed_dir in args.seed_dir:
        catalog_path = seed_dir / "validation_snapshot_catalog.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        selections = _select_validation_peaks(catalog)
        written: dict[str, dict[str, object]] = {}
        for name, candidate in selections.items():
            source = seed_dir / str(candidate["snapshot"])
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            _validate_snapshot(checkpoint, catalog, candidate)
            checkpoint["residual_gate"] = {
                "alpha": candidate["alpha"],
                "grid": catalog["residual_gate_grid"],
                "selection_split": "validation",
                "selection_policy": name,
                "validation_metrics": {
                    key: value
                    for key, value in candidate.items()
                    if key not in {"snapshot"}
                },
            }
            checkpoint["validation_peak_selection"] = {
                "policy": name,
                "catalog_schema": catalog["schema"],
                "holdout_used": False,
            }
            output = seed_dir / SELECTION_FILES[name]
            torch.save(checkpoint, output)
            written[name] = {
                **candidate,
                "checkpoint": output.name,
                "checkpoint_sha256": _sha256(output),
            }
        summary = {
            "schema": "aic.gnn_v2.g4_validation_peak_selection.v1",
            "dataset_sha256": catalog["dataset_sha256"],
            "candidate_sha256": catalog["candidate_sha256"],
            "selection_split": "validation",
            "holdout_used": False,
            "baseline": _baseline_candidate(catalog),
            "selections": written,
        }
        (seed_dir / "validation_peak_selection.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"seed_dir={seed_dir} "
            + " ".join(
                f"{name}=epoch{item['epoch']}/alpha{item['alpha']:g}/"
                f"rho{item['spearman']:.6f}/gain18"
                f"{item['top_gain_at_k']['18']:.3f}"
                for name, item in written.items()
            ),
            flush=True,
        )


def _select_validation_peaks(
    catalog: dict[str, object],
) -> dict[str, dict[str, object]]:
    if catalog.get("schema") != "aic.gnn_v2.validation_snapshot_catalog.v1":
        raise ValueError("validation snapshot catalog schema mismatch")
    if catalog.get("holdout_evaluated") is not False:
        raise ValueError("peak selection catalog must not contain holdout evaluation")
    candidates = _flatten_candidates(catalog)
    baseline = _baseline_candidate(catalog)
    baseline_spearman = float(baseline["spearman"])
    baseline_gain18 = float(baseline["top_gain_at_k"]["18"])
    global_best = max(candidates, key=_global_key)
    budget_candidates = [
        candidate
        for candidate in candidates
        if float(candidate["spearman"]) >= baseline_spearman - 1.0e-12
        and float(candidate["top_gain_at_k"]["18"])
        >= baseline_gain18 - 1.0e-12
    ]
    topgain_candidates = [
        candidate
        for candidate in candidates
        if float(candidate["spearman"]) >= baseline_spearman - 1.0e-12
    ]
    if not budget_candidates or not topgain_candidates:
        raise ValueError("alpha=0 baseline fallback is missing from the catalog")
    return {
        "global_spearman": global_best,
        "budget_safe_spearman": max(budget_candidates, key=_global_key),
        "topgain18_safe": max(topgain_candidates, key=_topgain_key),
    }


def _flatten_candidates(catalog: dict[str, object]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for entry in catalog["entries"]:
        for gate in entry["gates"]:
            candidates.append(
                {
                    "epoch": int(entry["epoch"]),
                    "snapshot": str(entry["snapshot"]),
                    "optimizer_step_effective": bool(
                        entry["optimizer_step_effective"]
                    ),
                    "alpha": float(gate["alpha"]),
                    "spearman": float(gate["spearman"]),
                    "ndcg_at_k": gate["ndcg_at_k"],
                    "top_gain_at_k": gate["top_gain_at_k"],
                }
            )
    if not candidates:
        raise ValueError("validation snapshot catalog is empty")
    return candidates


def _baseline_candidate(catalog: dict[str, object]) -> dict[str, object]:
    candidates = _flatten_candidates(catalog)
    baselines = [
        candidate
        for candidate in candidates
        if candidate["epoch"] == 0 and candidate["alpha"] == 0.0
    ]
    if len(baselines) != 1:
        raise ValueError("catalog must contain exactly one epoch-zero alpha=0 baseline")
    return baselines[0]


def _global_key(candidate: dict[str, object]) -> tuple[float, float, int, float]:
    return (
        float(candidate["spearman"]),
        float(candidate["top_gain_at_k"]["18"]),
        -int(candidate["epoch"]),
        -float(candidate["alpha"]),
    )


def _topgain_key(candidate: dict[str, object]) -> tuple[float, float, int, float]:
    return (
        float(candidate["top_gain_at_k"]["18"]),
        float(candidate["spearman"]),
        -int(candidate["epoch"]),
        -float(candidate["alpha"]),
    )


def _validate_snapshot(
    checkpoint: dict[str, object],
    catalog: dict[str, object],
    candidate: dict[str, object],
) -> None:
    if checkpoint.get("dataset_sha256") != catalog["dataset_sha256"]:
        raise ValueError("snapshot dataset identity mismatch")
    if checkpoint.get("candidate_sha256") != catalog["candidate_sha256"]:
        raise ValueError("snapshot candidate identity mismatch")
    if checkpoint.get("validation_snapshot", {}).get("epoch") != candidate["epoch"]:
        raise ValueError("snapshot epoch mismatch")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
