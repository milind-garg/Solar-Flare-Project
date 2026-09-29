"""
Evaluation metrics for the solar flare forecaster.

Metrics:
  - True Positive Rate (TPR / Recall)
  - False Alarm Rate (FAR = FP / (FP + TN))
  - Lead Time: flare peak time minus first alert time (minutes)
  - AUC-ROC, F1 at multiple thresholds
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve


def predict_proba(
    model,
    X: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu",
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
) -> np.ndarray:
    """Run model inference and return probabilities (sigmoid applied).

    Pass the norm_mean/norm_std saved with the checkpoint when X is raw (un-z-scored);
    omit them when X is already normalized.
    """
    if norm_mean is not None and norm_std is not None:
        X = (X - norm_mean) / (norm_std + 1e-8)
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb = torch.from_numpy(X[i: i + batch_size].astype(np.float32)).to(device)
            out = model(Xb)
            logits = out[0] if isinstance(out, tuple) else out  # multi-class model returns (logit, class_logits)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute TPR, FAR, precision, F1, AUC at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    tpr = tp / (tp + fn + 1e-9)   # recall / sensitivity
    far = fp / (fp + tn + 1e-9)   # false alarm rate = FPR
    precision = tp / (tp + fp + 1e-9)
    f1 = 2 * precision * tpr / (precision + tpr + 1e-9)

    # Space-weather skill scores — the metrics flare-forecasting papers report,
    # both robust to the heavy class imbalance (flares are rare).
    # TSS (True Skill Statistic) = TPR - FPR, range [-1, 1], 0 = no skill.
    tss = tpr - far
    # HSS (Heidke Skill Score) = skill over a random/climatology baseline.
    hss_denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2 * (tp * tn - fn * fp) / (hss_denom + 1e-9)

    fpr, tpr_curve, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr_curve)

    return {
        "threshold": threshold,
        "TPR": float(tpr),
        "FAR": float(far),
        "TSS": float(tss),
        "HSS": float(hss),
        "precision": float(precision),
        "F1": float(f1),
        "AUC": float(roc_auc),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, thresholds=(0.3, 0.5, 0.7)) -> pd.DataFrame:
    """Return metrics DataFrame across multiple thresholds."""
    rows = [compute_metrics(y_true, y_prob, t) for t in thresholds]
    return pd.DataFrame(rows)


def compute_lead_times(
    alert_times: pd.DatetimeIndex,
    flare_peak_times: pd.DatetimeIndex,
    match_window_min: int = 60,
    false_alarm_window_min: int = 60,
) -> dict:
    """
    Compute lead time statistics and false alarm count.

    An alert is a True Positive if a flare peaks within match_window_min after the alert.
    An alert with no flare within false_alarm_window_min is a False Alarm.

    Returns dict with lead_times_min list, median_lead_min, p90_lead_min, n_false_alarms.
    """
    lead_times = []
    n_false_alarms = 0

    for at in alert_times:
        # Find flares that peak after this alert (up to match_window_min ahead)
        window_end = at + pd.Timedelta(minutes=match_window_min)
        candidates = [pt for pt in flare_peak_times if at <= pt <= window_end]

        if candidates:
            earliest_peak = min(candidates)
            lead_min = (earliest_peak - at).total_seconds() / 60.0
            lead_times.append(lead_min)
        else:
            n_false_alarms += 1

    return {
        "lead_times_min": lead_times,
        "median_lead_min": float(np.median(lead_times)) if lead_times else np.nan,
        "p90_lead_min": float(np.percentile(lead_times, 90)) if lead_times else np.nan,
        "n_true_positive_alerts": len(lead_times),
        "n_false_alarms": n_false_alarms,
        "false_alarm_rate_pct": 100 * n_false_alarms / max(len(alert_times), 1),
    }


def evaluation_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_times: pd.DatetimeIndex,
    flare_catalog: pd.DataFrame,
    cfg: dict | None = None,
    device: str = "cpu",
) -> None:
    """Print a full evaluation report: threshold sweep + lead time analysis."""
    if cfg is None:
        import yaml
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)

    threshold = cfg["forecasting"]["alert_threshold"]
    lead_min  = cfg["forecasting"]["lead_time_min"]

    y_prob = predict_proba(model, X_test, device=device)

    print("=== Threshold Sweep ===")
    sweep = threshold_sweep(y_test, y_prob)
    print(sweep.to_string(index=False))

    print(f"\n=== Lead Time Analysis (threshold={threshold}) ===")
    alert_mask = y_prob >= threshold
    alert_times = test_times[alert_mask] if alert_mask.any() else pd.DatetimeIndex([])
    flare_peaks = pd.DatetimeIndex(flare_catalog["peak_time"]) if not flare_catalog.empty else pd.DatetimeIndex([])

    lt = compute_lead_times(alert_times, flare_peaks, match_window_min=lead_min)
    for k, v in lt.items():
        if k != "lead_times_min":
            print(f"  {k}: {v}")


def _demo():
    """Sanity check: a perfect forecast scores TSS=HSS=1; a coin-flip scores ~0."""
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=float)
    perfect = compute_metrics(y, y, threshold=0.5)
    assert abs(perfect["TSS"] - 1.0) < 1e-6, perfect["TSS"]
    assert abs(perfect["HSS"] - 1.0) < 1e-6, perfect["HSS"]

    # All-positive predictions: TPR=1, FPR=1 -> TSS=0 (no skill).
    allpos = compute_metrics(y, np.ones_like(y), threshold=0.5)
    assert abs(allpos["TSS"]) < 1e-6, allpos["TSS"]
    print(f"OK: perfect TSS/HSS=1.0, all-positive TSS={allpos['TSS']:.3f}")


if __name__ == "__main__":
    _demo()
