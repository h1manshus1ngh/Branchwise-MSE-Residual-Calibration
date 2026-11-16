"""Build the compact feature cache used by the BMRC evaluation scripts.

The CSV manifest must point to locally available, licensed videos. Exact
reproduction of the saved tables additionally requires the original clip IDs,
frame indices, benchmark/source labels, and the original reference archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from stall import STALL


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV following manifests/manifest_template.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/paper_complete_v1")
    parser.add_argument("--params", type=Path, default=Path(os.environ.get(
        "STALL_PARAMS", ROOT / "data/stall_params_vatex_dino_v3.npz")))
    parser.add_argument("--dino-repo", type=Path, default=None)
    parser.add_argument("--dino-weights", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not args.params.is_file():
        parser.error(f"STALL reference archive not found: {args.params}")

    args.output.mkdir(parents=True, exist_ok=True)
    records_dir, features_dir = args.output / "records", args.output / "features"
    records_dir.mkdir(exist_ok=True)
    features_dir.mkdir(exist_ok=True)
    params = np.load(args.params, allow_pickle=False)
    detector = STALL(args.device, params, dino_repo=str(args.dino_repo) if args.dino_repo else None,
                     dino_weights=str(args.dino_weights) if args.dino_weights else None)

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    needed = {"id", "benchmark", "source_model", "subset", "duration", "video_path", "frame_indices_json"}
    if not rows or not needed.issubset(rows[0]):
        parser.error(f"Manifest must contain columns: {', '.join(sorted(needed))}")

    for i, row in enumerate(rows, 1):
        clip_id = row["id"].strip()
        if not clip_id:
            raise ValueError(f"Manifest row {i} has an empty id")
        duration = int(row["duration"])
        if duration not in (1, 2):
            raise ValueError(f"Row {i}: duration must be 1 or 2 seconds")
        video_path = Path(row["video_path"]).expanduser()
        if not video_path.is_file():
            raise FileNotFoundError(f"Row {i}: video not found: {video_path}")
        frame_indices = json.loads(row["frame_indices_json"])
        if not isinstance(frame_indices, list) or not frame_indices:
            raise ValueError(f"Row {i}: frame_indices_json must be a nonempty JSON array")

        key = hashlib.sha256(clip_id.encode("utf-8")).hexdigest()
        record_path, feature_path = records_dir / f"{key}.json", features_dir / f"{key}.npz"
        frames = detector.inference(str(video_path), frame_indices=frame_indices)
        embs = np.asarray(frames["embs"][0], dtype=np.float32)
        np.savez_compressed(feature_path,
                            spatial_ll=np.asarray(frames["spat_ll"][0], dtype=np.float32),
                            temporal_ll=np.asarray(frames["temp_ll"][0], dtype=np.float32),
                            final_cls=embs)
        record = {"id": clip_id, "benchmark": row["benchmark"].strip(),
                  "source_model": row["source_model"].strip(), "subset": row["subset"].strip(),
                  "duration": duration, "frame_indices": frame_indices,
                  "original": float(np.asarray(frames["final_score"])[0])}
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"[{i}/{len(rows)}] {clip_id}", flush=True)
if __name__ == "__main__":
    main()
