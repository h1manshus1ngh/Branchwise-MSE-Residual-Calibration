# Branchwise MSE Residual Calibration for Generated-Video Detection

This repository contains the BMRC evaluation code and result summaries. BMRC
uses real-reference appearance and transition likelihoods, then trains one MSE
residual head per branch from labeled real and generated videos. The heads and
quality-weighted fusion are supervised.

## Reported results

The primary result is an equal-generator macro over the available two-second
groups, with three optimization seeds per group. Each test fold holds out one
generator within a benchmark. AUC and AP use real videos as the positive class.

| Benchmark | Groups | STALL AUC / AP-real | BMRC AUC / AP-real | Change |
|---|---:|---:|---:|---:|
| VideoFeedback | 10 | 0.8421 / 0.8570 | 0.8698 / 0.8799 | +0.0277 / +0.0229 |
| GenVideo | 8 | 0.8153 / 0.8103 | 0.8776 / 0.8829 | +0.0623 / +0.0726 |
| ComGenVid | 2 | 0.8422 / 0.8496 | 0.8746 / 0.8694 | +0.0324 / +0.0198 |
| **All groups (equal-generator macro)** | **20** | **0.8314 / 0.8376** | **0.8734 / 0.8800** | **+0.0420 / +0.0425** |

The mean is not a gain on every generator: Fast-SVD and Pika regress in the
VideoFeedback rows. Three seeds measure optimization variability on the same
test clips; they are not independent dataset replications.

The branch ablation uses the same folds and test clips:

| Variant | AUC | AP-real |
|---|---:|---:|
| Reference score, fixed fusion | 0.8314 | 0.8376 |
| Spatial residual head only | 0.8386 | 0.8572 |
| Temporal residual head only | 0.8568 | 0.8507 |
| Equal fusion of corrected branches | 0.8734 | 0.8801 |
| Quality-weighted fusion (BMRC) | 0.8734 | 0.8800 |

The quality-weighted and equal-corrected variants are nearly tied on this
evaluation. Full per-generator and per-seed tables are in
[`results/main_lgo/`](results/main_lgo/) and
[`results/component_ablation/`](results/component_ablation/).

## Interactive video demo

Open [`notebooks/bmrc_interactive_demo.ipynb`](notebooks/bmrc_interactive_demo.ipynb) to select a local video, preview its sampled frames, fit the BMRC residual heads from a labeled feature cache, and inspect the spatial, temporal, and fused scores. This is a qualitative demo, not a benchmark evaluation. It requires local video data, DINOv3 weights, STALL reference statistics, and a labeled BMRC cache; these assets are not distributed in the repository. The notebook does not upload or copy videos or caches.

The demo needs `ipywidgets` and `matplotlib`, which are included in the environment and requirements files. Set `BMRC_TRAIN_CACHE` to either a prepared cache directory (`records/` and `features/`) or a compatible local `.npz` feature archive. Set `BMRC_DEMO_VIDEO_DIR` to a folder of local clips if you want the notebook to list them automatically. The notebook explains how to build a cache and run the demo.

## What is required to reproduce the tables

The CSVs in `results/` document the saved runs. The local feature cache and
benchmark videos are not included. To recompute the exact tables, you
need the same clip IDs, benchmark and source labels, selected frame indices,
video files, DINOv3 checkpoint, and STALL reference-statistics archive used for
the saved run. A new or differently sampled manifest is a new evaluation and
should not be described as an exact reproduction of these numbers.

The cache writer in `src/prepare_cache.py` accepts local video paths through a
manifest. Start from [`manifests/manifest_template.csv`](manifests/manifest_template.csv).
For exact replay, populate it with the original rows and exact frame indices;
for a fresh experiment, use your own licensed videos and report the changed
protocol.

## Environment

The scripts were developed with Python 3.10, PyTorch, NumPy, pandas, and
scikit-learn. Create the base environment and install a PyTorch build that
matches your CUDA driver using the official selector at
[pytorch.org](https://pytorch.org/get-started/locally/):

```bash
conda env create -f environment.yml
conda activate bmrc-video-detection
python -m pip install -r requirements.txt
```

Clone the DINOv3 source and obtain the appropriate pretrained checkpoint under
its terms. Set these variables to local paths:

```bash
export DINO_V3_REPO_DIR=/path/to/dinov3
export DINO_V3_WEIGHTS=/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Obtain the STALL reference archive from the released STALL project and set:

```bash
export STALL_PARAMS=/path/to/stall_params_vatex_dino_v3.npz
```

The archive must contain the STALL spatial/temporal whitening and real-reference
calibration arrays expected by `src/stall.py`. Keep all benchmark videos and
derived caches local; follow the datasets' access terms.

## Build the feature cache

Each row needs a unique `id`, `benchmark`, `source_model`, `subset` (`real` for
authentic clips; another label for generated clips), `duration` in seconds,
`video_path`, and a JSON list of zero-based sampled frame indices. For the main
setting, use the same 16 frames as the original two-second evaluation.

```bash
python src/prepare_cache.py manifests/my_manifest.csv \
  --output data/paper_complete_v1 \
  --params "$STALL_PARAMS" \
  --device cuda
```

The generated feature and record files are ignored by Git. The manifest's IDs,
labels, and frame indices must match the saved experiment to reproduce it
exactly.

## Run training and evaluation

```bash
export BMRC_REPLAY="$PWD/data/paper_complete_v1"
export BMRC_OUTPUT="$PWD/results/main_lgo"
python src/run_bmrc_leave_one_generator.py

export BMRC_ABLATION_OUTPUT="$PWD/results/component_ablation"
python src/run_bmrc_component_ablation.py
```

The main script trains on the other generator groups within the same benchmark,
selects stopping epochs on validation partitions, and compares BMRC and the
fixed reference score on the same balanced held-out clips. It uses seeds 42,
43, and 44. The component script evaluates spatial-only, temporal-only,
equal-corrected, and quality-weighted fusion on those same folds. Both scripts
write per-seed and per-generator CSV files.

## Result files

- `results/main_lgo/report.md`: benchmark and overall macro summary.
- `results/main_lgo/per_generator_summary.csv`: generator-level metrics and
  changes.
- `results/main_lgo/per_seed_metrics.csv`: seed-level metrics and evaluation
  protocol labels.
- `results/component_ablation/`: branch/fusion ablation summaries.
- `results/comgenvid_full_coverage/`: previously saved full-coverage replay
  metrics for ComGenVid; these are separate from the balanced macro table.

## Citation and attribution


Coming soon
