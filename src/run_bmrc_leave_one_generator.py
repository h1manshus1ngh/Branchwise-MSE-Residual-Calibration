"""Three-benchmark, 2-second, leave-one-generator-out BMRC evaluation.

Uses the frozen DINOv3/STAL likelihood sequence cache from paper_complete_v1.
For each target generator, its fake clips are excluded from head fitting.
Real clips are split by source into train/validation/calibration/test partitions;
only train and validation are used to fit and stop the MSE heads. Test metrics
are computed on the same balanced clips for STALL and BMRC.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from evidential import fit_model


ROOT = Path(__file__).resolve().parents[1]
REPLAY = Path(os.environ.get("BMRC_REPLAY", ROOT / "data/paper_complete_v1"))
PARAMS = Path(os.environ.get("STALL_PARAMS", ROOT / "data/stall_params_vatex_dino_v3.npz"))
OUT = Path(os.environ.get("BMRC_OUTPUT", ROOT / "results/main_lgo"))
SEEDS = (42, 43, 44)
SPLIT_SEED = 20261112
BENCHMARKS = ("VideoFeedback", "GenVideo", "ComGenVid")


def stable_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def score_features(spatial: np.ndarray, temporal: np.ndarray, embeddings: np.ndarray,
                   spat_refs: np.ndarray, temp_refs: np.ndarray,
                   spat_loc_scale: tuple[float, float],
                   temp_loc_scale: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror evidential.FeatureExtractor from cached per-frame likelihoods."""
    branch_values = (np.asarray(spatial), np.asarray(temporal))
    qs = (0.9, 0.1)
    refs = (spat_refs, temp_refs)
    loc_scales = (spat_loc_scale, temp_loc_scale)
    features, percentiles = [], []

    diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=-1)
    frame_norms = np.linalg.norm(embeddings, axis=-1)
    relative_motion = (float(np.median(diffs) / max(float(np.median(frame_norms)), 1e-8))
                       if len(diffs) else 0.0)
    motion_cv = float(np.std(diffs) / max(float(np.mean(diffs)), 1e-8)) if len(diffs) else 0.0
    valid_fraction = float(np.isfinite(temporal).sum() / max(len(temporal), 1))

    for branch, values in enumerate(branch_values):
        q, ref, (loc, scale) = qs[branch], refs[branch], loc_scales[branch]
        finite = values[np.isfinite(values)]
        if not finite.size:
            aggregate = np.nan
            pct = 0.5
        else:
            aggregate = float(np.max(finite) if q > 0.5 else np.min(finite))
            pct = float(np.searchsorted(ref, aggregate, side="right") / len(ref))
        z = (finite - loc) / scale if finite.size else np.array([0.0])
        stats = [float(np.mean(z)), float(np.std(z)),
                 *[float(x) for x in np.quantile(z, [.1, .5, .9])]]
        descriptor = [pct, *stats, float(np.log1p(len(finite))),
                      1.0 if branch == 0 else valid_fraction,
                      relative_motion, motion_cv]
        features.append(descriptor)
        percentiles.append(pct)
    quality = np.asarray([1.0, valid_fraction * min(1.0, relative_motion / 1e-3)],
                         dtype=np.float32)
    return np.asarray(features, dtype=np.float32), np.asarray(percentiles, dtype=np.float32), quality


def reference_statistics() -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    with np.load(PARAMS, allow_pickle=False) as p:
        spat = np.asarray(p["calib_ll_spat"], dtype=np.float64)
        temp = np.asarray(p["calib_ll_temp"], dtype=np.float64)

    def summarize(values: np.ndarray, high: bool) -> tuple[np.ndarray, tuple[float, float]]:
        finite = values[np.isfinite(values)]
        loc = float(np.median(finite))
        scale = max(float(np.quantile(finite, .75) - np.quantile(finite, .25)), 1e-6)
        aggregate = []
        for row in values:
            x = row[np.isfinite(row)]
            if x.size:
                aggregate.append(float(np.max(x) if high else np.min(x)))
        return np.sort(np.asarray(aggregate, dtype=np.float64)), (loc, scale)

    spat_ref, spat_ls = summarize(spat, True)
    temp_ref, temp_ls = summarize(temp, False)
    return spat_ref, temp_ref, spat_ls, temp_ls


def load_rows() -> dict[str, np.ndarray]:
    spat_ref, temp_ref, spat_ls, temp_ls = reference_statistics()
    rows = []
    records_dir = REPLAY / "records"
    features_dir = REPLAY / "features"
    for record_path in sorted(records_dir.glob("*.json")):
        record = json.loads(record_path.read_text())
        if record.get("duration") != 2 or record.get("benchmark") not in BENCHMARKS:
            continue
        feature_path = features_dir / f"{record_path.stem}.npz"
        with np.load(feature_path, allow_pickle=False) as z:
            feat, pct, quality = score_features(
                z["spatial_ll"], z["temporal_ll"], z["final_cls"],
                spat_ref, temp_ref, spat_ls, temp_ls)
        if not np.isfinite(feat).all() or not np.isfinite(pct).all():
            raise ValueError(f"Nonfinite descriptor for {record['id']}")
        rows.append({
            "id": record["id"], "benchmark": record["benchmark"],
            "generator": record["source_model"], "is_real": int(record["subset"] == "real"),
            "features": feat, "percentiles": pct, "quality": quality,
            "stall": float(record["original"]),
        })
    if not rows:
        raise ValueError("No duration-2 records found")
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def split_group(indices: np.ndarray, key: str) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SPLIT_SEED + stable_seed(key))
    order = rng.permutation(indices)
    cuts = [int(.50 * len(order)), int(.65 * len(order)), int(.80 * len(order))]
    parts = np.split(order, cuts)
    return dict(zip(("train", "validation", "calibration", "test"), parts))


def metric_row(y: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(y)), "n_real": int(np.sum(y == 1)), "n_fake": int(np.sum(y == 0)),
        "auroc": float(roc_auc_score(y, scores)),
        "ap_real": float(average_precision_score(y, scores)),
    }


def main() -> None:
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_rows()
    split_rows, pred_rows = [], []

    for benchmark in BENCHMARKS:
        bench_idx = np.flatnonzero(data["benchmark"] == benchmark)
        generators = sorted(set(data["generator"][bench_idx][data["is_real"][bench_idx] == 0].tolist()))
        if len(generators) < 2:
            continue
        real_parts: dict[str, np.ndarray] = {}
        for source in sorted(set(data["generator"][bench_idx][data["is_real"][bench_idx] == 1].tolist())):
            idx = bench_idx[(data["is_real"][bench_idx] == 1) & (data["generator"][bench_idx] == source)]
            real_parts[source] = split_group(idx, f"{benchmark}:real:{source}")

        for target in generators:
            target_fake = bench_idx[(data["is_real"][bench_idx] == 0) & (data["generator"][bench_idx] == target)]
            known = [g for g in generators if g != target]
            fake_parts = {
                g: split_group(bench_idx[(data["is_real"][bench_idx] == 0) &
                                         (data["generator"][bench_idx] == g)],
                               f"{benchmark}:fake:{g}")
                for g in known
            }
            tr = np.concatenate([p["train"] for p in real_parts.values()] +
                                [p["train"] for p in fake_parts.values()])
            va = np.concatenate([p["validation"] for p in real_parts.values()] +
                                [p["validation"] for p in fake_parts.values()])
            real_test = np.concatenate([p["test"] for p in real_parts.values()])
            n_balanced = min(len(real_test), len(target_fake))
            sample_seed = SPLIT_SEED + stable_seed(f"{benchmark}:{target}:balanced")
            rng = np.random.default_rng(sample_seed)
            test_idx = np.sort(np.concatenate([
                rng.choice(real_test, n_balanced, replace=False),
                rng.choice(target_fake, n_balanced, replace=False),
            ]))
            y_test = data["is_real"][test_idx]
            stall_metrics = metric_row(y_test, data["stall"][test_idx])
            for seed in SEEDS:
                model, history = fit_model(
                    data["features"][tr], data["percentiles"][tr], data["quality"][tr],
                    data["is_real"][tr], data["features"][va], data["percentiles"][va],
                    data["quality"][va], data["is_real"][va], seed=seed,
                    epochs=150, objective="mse")
                with torch.no_grad():
                    raw = model(torch.as_tensor(data["features"][test_idx], dtype=torch.float32))
                    residual = np.stack([branch[0].numpy() for branch in raw], axis=1)
                pct = data["percentiles"][test_idx]
                q = data["quality"][test_idx]
                weights = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
                bmrc_score = np.sum(weights * np.clip(pct + residual, 0, 1), axis=1)
                bmrc_metrics = metric_row(y_test, bmrc_score)
                for method, values in (("STALL", stall_metrics), ("BMRC_MSE", bmrc_metrics)):
                    split_rows.append({
                        "benchmark": benchmark, "generator": target,
                        "train_generators": "|".join(known), "seed": seed,
                        "method": method,
                        "evaluation": "balanced_2s_leave_one_generator_out",
                        "best_epoch": (min(history, key=lambda h: h["validation_loss"])["epoch"]
                                       if method == "BMRC_MSE" else ""),
                        **values,
                    })
                for j, score in zip(test_idx, bmrc_score):
                    pred_rows.append({
                        "id": data["id"][j], "benchmark": benchmark, "generator": target,
                        "seed": seed, "label_real": int(data["is_real"][j]),
                        "stall_score": float(data["stall"][j]), "bmrc_score": float(score),
                    })
            pd.DataFrame(split_rows).to_csv(OUT / "per_seed_metrics.partial.csv", index=False)
            print(f"Completed {benchmark}/{target}; train generators={len(known)}, balanced n={n_balanced}",
                  flush=True)

    by_seed = pd.DataFrame(split_rows)
    by_seed.to_csv(OUT / "per_seed_metrics.csv", index=False)
    (OUT / "per_seed_metrics.partial.csv").unlink(missing_ok=True)
    aggregate_rows = []
    for (benchmark, generator), group in by_seed.groupby(["benchmark", "generator"], sort=True):
        base = group[group.method == "STALL"].iloc[0]
        cand = group[group.method == "BMRC_MSE"]
        aggregate_rows.append({
            "benchmark": benchmark, "generator": generator,
            "n_each_class": int(base["n_real"]),
            "stall_auroc": float(base["auroc"]), "stall_ap_real": float(base["ap_real"]),
            "bmrc_auroc_mean": float(cand["auroc"].mean()),
            "bmrc_auroc_sd": float(cand["auroc"].std(ddof=1)),
            "delta_auroc": float(cand["auroc"].mean() - base["auroc"]),
            "bmrc_ap_real_mean": float(cand["ap_real"].mean()),
            "bmrc_ap_real_sd": float(cand["ap_real"].std(ddof=1)),
            "delta_ap_real": float(cand["ap_real"].mean() - base["ap_real"]),
            "training_generators": cand["train_generators"].iloc[0],
        })
    summary = pd.DataFrame(aggregate_rows)
    summary.to_csv(OUT / "per_generator_summary.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT / "balanced_test_predictions.csv", index=False)
    protocol = {
        "duration_seconds": 2,
        "feature_source": str(REPLAY / "features") + " and " + str(REPLAY / "records"),
        "encoder_and_reference": "Frozen DINOv3 ViT-L/16 block-24 CLS; original VATEX real-reference spatial/temporal statistics.",
        "feature_validation": "Stored per-frame likelihoods are recomputed exactly from final_cls and the same STALL reference parameters; descriptors follow the saved BMRC FeatureExtractor.",
        "test_protocol": "Leave one generator out within each benchmark. Split each real source 50/15/15/20; use the final 20% as held-out real test. Evaluate equal counts of held-out real and target-generator fake clips, using the same clips for STALL and BMRC.",
        "training_protocol": "Train branchwise MSE residual heads on the training partitions of all non-target fake generator groups and real sources. Use validation partitions for early stopping. Calibration and held-out partitions do not fit the heads.",
        "seeds": list(SEEDS),
        "metrics": "AUROC and real-positive AP; STALL is evaluated on exactly the same balanced test IDs.",
        "limitations": "This is a local 2-second reconstruction on available benchmark feature rows. Generator coverage can differ from the published STALL table when the required cache rows are absent. Splits and models share benchmark sources; seed SD is not an independent-dataset confidence interval.",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("\nFull per-generator summary:\n", summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
