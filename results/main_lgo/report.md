# BMRC leave-one-generator-out results

Results from the cached balanced two-second experiment. See protocol.json and per_seed_metrics.csv for full settings and seed-level rows.

Equal-generator macro AUROC: 0.8314 (STALL) to 0.8734 (BMRC); mean delta +0.0420.
Equal-generator macro AP-real: 0.8376 (STALL) to 0.8800 (BMRC); mean delta +0.0425.

BMRC AUROC declines for VideoFeedback Fast-SVD and Pika. The table covers the local 2-second rows, not every generator in the original STALL paper.

| Benchmark | Groups | STALL macro AUC/AP-real | BMRC macro AUC/AP-real | Delta |
|---|---:|---:|---:|---:|
| ComGenVid | 2 | 0.8422/0.8496 | 0.8746/0.8694 | +0.0324/+0.0198 |
| GenVideo | 8 | 0.8153/0.8103 | 0.8776/0.8829 | +0.0623/+0.0726 |
| VideoFeedback | 10 | 0.8421/0.8570 | 0.8698/0.8799 | +0.0277/+0.0229 |
