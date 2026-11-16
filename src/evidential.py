"""Experimental supervised uncertainty extension; original STALL remains unchanged.

Each independent branch regresses r = real_label - branch_percentile with a
Normal-Inverse-Gamma head (Amini et al., 2020). This is residual regression,
not a claim that a Gaussian likelihood models binary class labels. Corrected
scores are ranking scores, not calibrated class probabilities.
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_NAMES = ["percentile", "ll_mean", "ll_std", "ll_q10", "ll_median", "ll_q90",
                 "log_count", "valid_fraction", "relative_motion", "motion_cv"]


def parameter_fingerprint(detector):
    digest = hashlib.sha256()
    for value in (detector.mu_spat, detector.w_spat, detector.mu_temp, detector.w_temp,
                  detector.calib_spat_sorted, detector.calib_temp_sorted):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


class FeatureExtractor:
    """Quantile aggregation calibrated with the SAME operator on real references.

    Low-motion gating is an explicit heuristic, not learned evidence. No motion
    magnitude calibration is available in the upstream parameter archive.
    """
    def __init__(self, detector, robust=False, quantile=0.1):
        if not 0 <= quantile < 0.5:
            raise ValueError("quantile must be in [0, 0.5)")
        self.detector, self.robust, self.quantile = detector, robust, quantile
        self.calibration = []
        self.location_scale = []
        for values, q in ((detector.calib_ll_spat, 1-quantile),
                          (detector.calib_ll_temp, quantile)):
            values = np.asarray(values, dtype=np.float64)
            finite = values[np.isfinite(values)]
            if not finite.size:
                raise ValueError("Empty finite calibration likelihoods")
            loc = np.median(finite)
            scale = max(np.quantile(finite, .75)-np.quantile(finite, .25), 1e-6)
            self.location_scale.append((loc, scale))
            # Upstream references contain per-video likelihood arrays.
            aggregated = np.array([self.aggregate(row, q) for row in values])
            self.calibration.append(np.sort(aggregated[np.isfinite(aggregated)]))

    def aggregate(self, values, q):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        if not len(values):
            return np.nan
        if self.robust:
            return float(np.quantile(values, q))
        return float(np.max(values) if q > .5 else np.min(values))

    def __call__(self, embs):
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 2:
            embs = embs[None]
        if embs.ndim != 3 or embs.shape[0] != 1 or embs.shape[1] < 1:
            raise ValueError("Expected one nonempty video [1,T,D] or [T,D]")
        if not np.isfinite(embs).all():
            raise ValueError("Embeddings must be finite")
        spat = self.detector.spatial_log_likelihood(embs)[0]
        temp = self.detector.temporal_log_likelihood(embs)[0]
        diffs = np.linalg.norm(np.diff(embs[0], axis=0), axis=-1)
        relative_motion = float(np.median(diffs) / max(np.median(np.linalg.norm(embs[0], axis=-1)), 1e-8)) if len(diffs) else 0.
        motion_cv = float(np.std(diffs) / max(np.mean(diffs), 1e-8)) if len(diffs) else 0.
        valid_fraction = float(np.isfinite(temp).sum() / max(len(temp), 1))
        features, percentiles = [], []
        for b, (values, q) in enumerate(((spat, 1-self.quantile), (temp, self.quantile))):
            finite = values[np.isfinite(values)]
            agg = self.aggregate(finite, q)
            ref = self.calibration[b]
            pct = float(np.searchsorted(ref, agg, side="right") / len(ref)) if len(finite) else 0.5
            loc, scale = self.location_scale[b]
            z = (finite-loc)/scale if len(finite) else np.array([0.])
            stats = [np.mean(z), np.std(z), *np.quantile(z, [.1, .5, .9])]
            features.append([pct, *stats, np.log1p(len(finite)),
                             1. if b == 0 else valid_fraction,
                             relative_motion, motion_cv])
            percentiles.append(pct)
        quality = np.array([1., valid_fraction * min(1., relative_motion/1e-3)], dtype=np.float32)
        # For no temporal information the paper requires spatial-only fallback.
        equal = float(np.mean(percentiles)) if np.isfinite(temp).any() else percentiles[0]
        return {"features": np.asarray(features, dtype=np.float32),
                "percentiles": np.asarray(percentiles, dtype=np.float32),
                "quality": quality, "equal_score": equal}


class ResidualHead(nn.Module):
    def __init__(self, n_features=len(FEATURE_NAMES), hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4))

    def forward(self, x):
        raw = self.net(x)
        gamma = torch.tanh(raw[..., 0])
        nu = F.softplus(raw[..., 1]) + 1e-4
        alpha = F.softplus(raw[..., 2]) + 1.0001
        beta = F.softplus(raw[..., 3]) + 1e-4
        return gamma, nu, alpha, beta


def nig_loss(target, parameters, regularization=0.01):
    """Student-t marginal NLL + |error|*(2*nu+alpha) evidence penalty."""
    gamma, nu, alpha, beta = parameters
    omega = 2 * beta * (1 + nu)
    error = target - gamma
    nll = (.5 * torch.log(torch.pi/nu) - alpha * torch.log(omega)
           + (alpha+.5)*torch.log(nu*error.square()+omega)
           + torch.lgamma(alpha)-torch.lgamma(alpha+.5))
    return nll + regularization * error.abs() * (2*nu+alpha)


class EvidentialFusion(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.heads = nn.ModuleList([ResidualHead(hidden=hidden) for _ in range(2)])
        self.register_buffer("center", torch.zeros(2, len(FEATURE_NAMES)))
        self.register_buffer("scale", torch.ones(2, len(FEATURE_NAMES)))

    def fit_scaler(self, features):
        x = torch.as_tensor(features, dtype=torch.float32)
        self.center.copy_(x.median(dim=0).values)
        self.scale.copy_((torch.quantile(x, .75, dim=0)-torch.quantile(x, .25, dim=0)).clamp_min(.05))

    def forward(self, features):
        x = (features-self.center)/self.scale
        return [self.heads[b](x[:, b]) for b in range(2)]

    @torch.no_grad()
    def predict(self, features, percentiles, quality):
        self.eval()
        x = torch.as_tensor(features, dtype=torch.float32)
        params = self(x)
        gamma = np.stack([p[0].numpy() for p in params], axis=1)
        aleatoric = np.stack([(p[3]/(p[2]-1)).numpy() for p in params], axis=1)
        epistemic = np.stack([(p[3]/(p[1]*(p[2]-1))).numpy() for p in params], axis=1)
        variance = aleatoric + epistemic
        precision = np.asarray(quality) / np.maximum(variance, 1e-8)
        weights = precision / np.maximum(precision.sum(axis=1, keepdims=True), 1e-8)
        corrected = np.clip(np.asarray(percentiles)+gamma, 0, 1)
        score = np.sum(weights*corrected, axis=1)
        # Correlated branches: weighted marginal SD is conservative relative to
        # independence pooling. This is a proxy, calibrated on held-out data.
        std = np.sum(weights*np.sqrt(variance), axis=1)
        return {"score": score, "std": std, "weights": weights,
                "aleatoric": aleatoric, "epistemic": epistemic,
                "epistemic_fused": np.sum(weights*epistemic, axis=1),
                "residual_mean": gamma}


def fit_model(features, percentiles, quality, labels, val_features, val_percentiles,
              val_quality, val_labels, seed=42, epochs=150, batch_size=128,
              learning_rate=1e-3, regularization=.01, hidden=32, objective="nig"):
    """Scaler and optimizer see training only; validation chooses early stopping."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if objective not in ("nig", "mse"):
        raise ValueError("objective must be nig or mse")
    loss_fn = (lambda target, p: nig_loss(target, p, regularization)) if objective == "nig" else (lambda target, p: (target-p[0]).square())
    torch.manual_seed(seed)
    model = EvidentialFusion(hidden)
    model.fit_scaler(features)
    x = torch.as_tensor(features, dtype=torch.float32)
    target = torch.as_tensor(labels[:, None]-percentiles, dtype=torch.float32)
    q = torch.as_tensor(quality, dtype=torch.float32)
    vx = torch.as_tensor(val_features, dtype=torch.float32)
    vt = torch.as_tensor(val_labels[:, None]-val_percentiles, dtype=torch.float32)
    vq = torch.as_tensor(val_quality, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, history, stale = float("inf"), None, [], 0
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(x), generator=generator).split(batch_size):
            optimizer.zero_grad()
            losses = torch.stack([loss_fn(target[idx, b], p)
                                  for b, p in enumerate(model(x[idx]))], dim=1)
            loss = (losses*q[idx]).sum()/q[idx].sum().clamp_min(1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses = torch.stack([loss_fn(vt[:, b], p)
                                  for b, p in enumerate(model(vx))], dim=1)
            val_loss = float((losses*vq).sum()/vq.sum().clamp_min(1))
        history.append({"epoch": epoch+1, "validation_loss": val_loss})
        if np.isfinite(val_loss) and val_loss < best - 1e-5:
            best, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 20:
            break
    if best_state is None:
        raise RuntimeError("Training produced no finite validation loss")
    model.load_state_dict(best_state)
    return model, history


def adaptive_aggregation(original, robust, original_percentiles, robust_percentiles, quality):
    """Use evidential reliability to mix extreme and quantile branch estimates.

    Each candidate already uses its matching real-reference aggregation. This
    avoids comparing a quantile summary against an extreme-score calibration.
    Shared observations mean independent-variance pooling would be overconfident.
    """
    candidates = (original, robust)
    variance = np.stack([p["aleatoric"]+p["epistemic"] for p in candidates], axis=1)
    precision = 1/np.maximum(variance, 1e-8)
    aggregation_weights = precision/precision.sum(axis=1, keepdims=True)
    corrected = np.stack([np.clip(pct+p["residual_mean"], 0, 1)
                          for pct, p in zip((original_percentiles, robust_percentiles), candidates)], axis=1)
    branch_score = (aggregation_weights*corrected).sum(axis=1)
    aleatoric = (aggregation_weights*np.sqrt(np.stack([p["aleatoric"] for p in candidates], axis=1))).sum(axis=1)**2
    epistemic = (aggregation_weights*np.sqrt(np.stack([p["epistemic"] for p in candidates], axis=1))).sum(axis=1)**2
    branch_variance = aleatoric+epistemic
    branch_precision = quality/np.maximum(branch_variance, 1e-8)
    weights = branch_precision/np.maximum(branch_precision.sum(axis=1, keepdims=True), 1e-8)
    return {"score": (weights*branch_score).sum(axis=1),
            "std": (weights*np.sqrt(branch_variance)).sum(axis=1),
            "weights": weights, "aleatoric": aleatoric, "epistemic": epistemic,
            "epistemic_fused": (weights*epistemic).sum(axis=1),
            "aggregation_weights": aggregation_weights, "branch_score": branch_score}
