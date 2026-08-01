"""Merge G4 frozen-evaluation shards into one auditable city summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


POLICIES = ("global_spearman", "budget_safe_spearman", "topgain18_safe")
K_VALUES = (5, 10, 18)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--z0-summary", type=Path, required=True)
    parser.add_argument("--expected-seeds", default="42,43,44")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--city", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value.strip()) for value in args.expected_seeds.split(","))
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.shard]
    z0 = json.loads(args.z0_summary.read_text(encoding="utf-8"))
    result = aggregate(summaries, z0, seeds, args.city)
    result["source_sha256"] = {
        "shards": {_display(path): _sha256(path) for path in args.shard},
        "z0_summary": _sha256(args.z0_summary),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_report(result), encoding="utf-8"
    )
    print(f"summary={args.output_dir / 'summary.json'}")


def aggregate(
    shards: list[dict], z0: dict, expected_seeds: tuple[int, ...], city: str
) -> dict:
    if not shards:
        raise ValueError("no G4 evaluation shards")
    dataset_sha = shards[0].get("dataset_sha256")
    candidate_sha = shards[0].get("candidate_sha256")
    runs: dict[str, dict] = {}
    for shard in shards:
        if shard.get("schema") != "aic.gnn_v2.g4_frozen_evaluation_shard.v1":
            raise ValueError("G4 shard schema mismatch")
        if shard.get("dataset_sha256") != dataset_sha:
            raise ValueError("G4 shard dataset mismatch")
        if shard.get("candidate_sha256") != candidate_sha:
            raise ValueError("G4 shard candidate mismatch")
        protocol = shard.get("protocol", {})
        if protocol.get("training_performed") is not False:
            raise ValueError("G4 evaluation shard performed training")
        if protocol.get("selection_split") != "validation":
            raise ValueError("G4 shard was not validation-frozen")
        if protocol.get("holdout_or_future_used_for_selection") is not False:
            raise ValueError("G4 shard used locked evaluation data for selection")
        overlap = set(runs) & set(shard.get("runs", {}))
        if overlap:
            raise ValueError(f"duplicate G4 runs: {sorted(overlap)}")
        runs.update(shard["runs"])
    if z0.get("dataset_sha256") != dataset_sha:
        raise ValueError("Z0 and G4 dataset mismatch")
    if z0.get("candidate_sha256") != candidate_sha:
        raise ValueError("Z0 and G4 candidate mismatch")
    expected_keys = {
        f"{policy}.seed_{seed}" for policy in POLICIES for seed in expected_seeds
    }
    if set(runs) != expected_keys:
        raise ValueError(
            f"G4 run matrix mismatch: missing={sorted(expected_keys - set(runs))}, "
            f"extra={sorted(set(runs) - expected_keys)}"
        )
    for key, run in runs.items():
        if run.get("validation_replay", {}).get("passed") is not True:
            raise ValueError(f"validation replay failed: {key}")

    z0_metrics = {
        "validation": z0["z0_current_window_metrics"]["validation"],
        "holdout": z0["z0_current_window_metrics"]["holdout"],
        "future_all": z0["z0_future_window_metrics"]["all_candidates"],
    }
    aggregate_by_policy = {
        policy: _aggregate_policy(
            [runs[f"{policy}.seed_{seed}"] for seed in expected_seeds], z0_metrics
        )
        for policy in POLICIES
    }
    return {
        "schema": "aic.gnn_v2.g4_frozen_evaluation.v1",
        "city": city,
        "dataset_sha256": dataset_sha,
        "candidate_sha256": candidate_sha,
        "expected_seeds": list(expected_seeds),
        "protocol": {
            "main_policy": "global_spearman",
            "secondary_policies": ["budget_safe_spearman", "topgain18_safe"],
            "selection_split": "validation",
            "holdout_or_future_used_for_selection": False,
            "all_negative_and_positive_results_retained": True,
        },
        "z0": z0_metrics,
        "runs": runs,
        "aggregate": aggregate_by_policy,
    }


def _aggregate_policy(runs: list[dict], z0: dict) -> dict:
    result: dict[str, object] = {
        "seeds": [int(run["seed"]) for run in runs],
        "selected_epochs": [
            int(run["residual_gate"]["validation_metrics"]["epoch"])
            for run in runs
        ],
        "selected_alphas": [float(run["residual_gate"]["alpha"]) for run in runs],
    }
    scopes = {
        "validation": [run["current_metrics"]["validation"] for run in runs],
        "holdout": [run["current_metrics"]["holdout"] for run in runs],
        "future_all": [run["future_metrics"]["all_candidates"] for run in runs],
    }
    for scope, metrics in scopes.items():
        scope_result = {
            "spearman": _stats([float(item["spearman"]) for item in metrics]),
            "delta_spearman_vs_z0": _stats(
                [float(item["spearman"]) - float(z0[scope]["spearman"]) for item in metrics]
            ),
            "ranking_at_k": {},
        }
        for k in K_VALUES:
            key = str(k)
            ndcg = [float(item["ranking_at_k"][key]["ndcg"]) for item in metrics]
            gain = [float(item["ranking_at_k"][key]["mean_gain"]) for item in metrics]
            scope_result["ranking_at_k"][key] = {
                "ndcg": _stats(ndcg),
                "delta_ndcg_vs_z0": _stats(
                    [value - float(z0[scope]["ranking_at_k"][key]["ndcg"]) for value in ndcg]
                ),
                "mean_gain": _stats(gain),
                "delta_mean_gain_vs_z0": _stats(
                    [value - float(z0[scope]["ranking_at_k"][key]["mean_gain"]) for value in gain]
                ),
            }
        result[scope] = scope_result
    return result


def _stats(values: list[float]) -> dict[str, float | list[float]]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def render_report(summary: dict) -> str:
    lines = [
        f"# {summary['city']} G4 冻结评测",
        "",
        "三个种子的 epoch 与残差门均只依据 validation 冻结；holdout 和未来窗口未参与模型、"
        "策略或超参数选择。主策略固定为 `global_spearman`。",
        "",
        "| 策略 | Epoch | Alpha | Val Spearman | Holdout Spearman | Future Spearman | "
        "Holdout NDCG@5/10/18 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for policy in POLICIES:
        item = summary["aggregate"][policy]
        validation = item["validation"]["spearman"]
        holdout = item["holdout"]["spearman"]
        future = item["future_all"]["spearman"]
        ndcg = item["holdout"]["ranking_at_k"]
        lines.append(
            f"| {policy} | {item['selected_epochs']} | {item['selected_alphas']} | "
            f"{validation['mean']:.6f} ± {validation['std']:.6f} | "
            f"{holdout['mean']:.6f} ± {holdout['std']:.6f} | "
            f"{future['mean']:.6f} ± {future['std']:.6f} | "
            f"{ndcg['5']['ndcg']['mean']:.6f} / {ndcg['10']['ndcg']['mean']:.6f} / "
            f"{ndcg['18']['ndcg']['mean']:.6f} |"
        )
    z0 = summary["z0"]
    lines.extend(
        [
            "",
            f"Z0 对照：validation / holdout / future Spearman 为 "
            f"`{z0['validation']['spearman']:.6f} / {z0['holdout']['spearman']:.6f} / "
            f"{z0['future_all']['spearman']:.6f}`。",
            "",
        ]
    )
    return "\n".join(lines)


def _display(path: Path) -> str:
    return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
