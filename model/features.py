"""
Build sliding-window feature matrices for the CNN-LSTM forecaster.

Per-timestep features (8 total):
  0  log_flux_short          — SXR 0.5–4 Å (or GOES short)
  1  log_flux_long           — SXR 1–8 Å (primary GOES-equivalent)
  2  log_hxr_total           — HXR broadband (HEL1OS or 0 if unavailable)
  3  dflux_1min              — 1-min derivative of flux_long (rise indicator)
  4  dflux_5min              — 5-min derivative of flux_long (slower precursor)
  5  hardness_ratio          — hxr_total / flux_long (key precursor)
  6  excess_norm             — deviation above background (normalized)
  7  qpp_power               — quasi-periodic pulsation band power (ISRO PS-15)

Input to model: (batch, window_min, 8)
Label: 1 if any flare peaks within next lead_time_min minutes, else 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml


FEATURE_NAMES = [
    "log_flux_short",
    "log_flux_long",
    "log_hxr_total",
    "dflux_1min",
    "dflux_5min",
    "hardness_ratio",
    "excess_norm",
    "qpp_power",        # quasi-periodic pulsation strength (HEL1OS ISRO PS-15 feature)
]
N_FEATURES = len(FEATURE_NAMES)

# Multi-class intensity order (ISRO PS-15). Index = CrossEntropy target class.
# Negative (no-flare) windows use -1, which CrossEntropyLoss(ignore_index=-1) skips.
FLARE_CLASS_ORDER = ["A", "B", "C", "M", "X"]
N_CLASSES = len(FLARE_CLASS_ORDER)


def _class_to_idx(cls: str) -> int:
    """'C3.0' -> 2. Unknown/empty -> -1 (ignored by the loss)."""
    if not cls:
        return -1
    return FLARE_CLASS_ORDER.index(cls[0]) if cls[0] in FLARE_CLASS_ORDER else -1

# log10 floors per channel — below-floor values clip here instead of going to -inf.
# GOES/SoLEXS-flux channels are W/m² (or counts scaled small); HXR is counts/s.
FLUX_FLOOR = 1e-9   # W/m² (or calibrated flux); ~1 nW/m² noise floor
HXR_FLOOR = 1e-1    # counts/s


def load_config(cfg_path: str = "config.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_feature_series(df_sxr: pd.DataFrame, df_hxr: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Merge SXR and HXR preprocessed DataFrames into a feature DataFrame at 1-min cadence.

    Args:
        df_sxr: preprocessed SoLEXS or GOES DataFrame (1s or 1min cadence)
        df_hxr: preprocessed HEL1OS DataFrame (1s cadence), or None

    Returns:
        DataFrame at 1-min cadence with FEATURE_NAMES columns.
    """
    # Resample both to 1-minute means
    sxr = df_sxr.resample("1min").mean()

    if df_hxr is not None:
        hxr = df_hxr.resample("1min").mean()
        # Align on common index
        sxr, hxr = sxr.align(hxr, join="left", fill_value=np.nan)
        hxr_total = hxr.get("hxr_total", pd.Series(0.0, index=sxr.index))
    else:
        hxr_total = pd.Series(0.0, index=sxr.index)

    flux_short = sxr.get("flux_short", pd.Series(np.nan, index=sxr.index)).clip(lower=1e-12)
    flux_long  = sxr.get("flux_long",  pd.Series(np.nan, index=sxr.index)).clip(lower=1e-12)
    bg_long    = sxr.get("bg_long",    pd.Series(np.nan, index=sxr.index)).clip(lower=1e-12)
    hxr_total  = hxr_total.clip(lower=0)

    # log10 with a floor — real dynamic range for both W/m² flux (GOES) and
    # counts/s (SoLEXS/HEL1OS). log1p(1e-6)≈1e-6 gave no scaling for GOES flux.
    log_flux_short = np.log10(flux_short.clip(lower=FLUX_FLOOR))
    log_flux_long  = np.log10(flux_long.clip(lower=FLUX_FLOOR))
    log_hxr_total  = np.log10(hxr_total.clip(lower=HXR_FLOOR))

    # Derivatives (per minute)
    dflux_1 = flux_long.diff(1)
    dflux_5 = flux_long.diff(5) / 5.0

    # Hardness ratio
    hardness = hxr_total / flux_long.replace(0, np.nan)

    # Normalized excess
    excess_norm = (flux_long - bg_long) / bg_long.replace(0, np.nan)

    # QPP feature: compute on NATIVE-cadence HEL1OS to preserve sub-2-min QPPs
    # (resampling to 1 min first would alias any period < 2 min — an ISRO PS-15 target).
    # After computation, resample the QPP series down to the 1-min feature grid.
    if df_hxr is not None:
        native_hxr = (
            df_hxr["hxr_total"] if "hxr_total" in df_hxr.columns else df_hxr.iloc[:, 0]
        ).clip(lower=0)
        if not native_hxr.eq(0).all():
            dt_sec = df_hxr.index.to_series().diff().dt.total_seconds().median()
            native_cadence = float(dt_sec) if pd.notna(dt_sec) and dt_sec > 0 else 1.0
            qpp = qpp_power(native_hxr, cadence_sec=native_cadence).resample("1min").mean().reindex(sxr.index)
        else:
            qpp = pd.Series(0.0, index=sxr.index)
    else:
        qpp = pd.Series(0.0, index=sxr.index)

    feat = pd.DataFrame({
        "log_flux_short": log_flux_short,
        "log_flux_long":  log_flux_long,
        "log_hxr_total":  log_hxr_total,
        "dflux_1min":     dflux_1,
        "dflux_5min":     dflux_5,
        "hardness_ratio": hardness,
        "excess_norm":    excess_norm,
        "qpp_power":      qpp,
    }, index=sxr.index)

    return feat


def build_windows(
    feat: pd.DataFrame,
    flare_times: pd.DatetimeIndex | None,
    window_min: int = 60,
    lead_time_min: int = 30,
    negative_ratio: float = 5.0,
    flare_classes: "list[str] | pd.Series | None" = None,
    return_times: bool = False,
) -> tuple:
    """
    Build (X, y) arrays from a feature DataFrame and known flare peak times.

    Args:
        feat:           Feature DataFrame at 1-min cadence (FEATURE_NAMES columns)
        flare_times:    Known flare peak times (for labeling)
        window_min:     Length of input window (timesteps)
        lead_time_min:  Positive label if flare peaks within next lead_time_min mins
        negative_ratio: Ratio of negative to positive samples (class balance cap)
        flare_classes:  Optional class label ("C3.0", "X1.2", …) per flare in
                        flare_times (same order/length). When given, a third array
                        y_cls is returned for multi-class training.
        return_times:   If True, append a DatetimeIndex of window anchor times to
                        the return tuple. Anchor time = feat.index[i], the timestamp
                        immediately *after* the input window ends (i.e. "now").

    Returns:
        (X, y)                  — default
        (X, y, y_cls)           — when flare_classes is given
        (X, y, times)           — when return_times=True and no flare_classes
        (X, y, y_cls, times)    — when both return_times=True and flare_classes given
    """
    feat = feat[FEATURE_NAMES].copy()

    # Fill NaN with forward-fill then 0
    feat = feat.ffill().fillna(0.0)
    values = feat.values.astype(np.float32)
    times = feat.index

    # Map floored peak-time → most-severe class index (for multi-class labels).
    want_cls = flare_classes is not None
    cls_by_time: dict = {}
    flare_set = set()
    if flare_times is not None:
        floored = flare_times.floor("1min")
        flare_set = set(floored.tolist())
        if want_cls:
            classes = list(flare_classes)
            for t, c in zip(floored, classes):
                idx = _class_to_idx(str(c))
                # keep the most severe class if two flares share a minute
                cls_by_time[t] = max(cls_by_time.get(t, -1), idx)

    X_pos, X_neg, cls_pos = [], [], []
    t_pos, t_neg = [], []

    for i in range(window_min, len(times) - lead_time_min + 1):
        window = values[i - window_min: i]          # (window_min, N_FEATURES)
        if np.isnan(window).mean() > 0.3:           # skip >30% NaN windows
            continue
        window = np.nan_to_num(window, nan=0.0)

        # Label: any flare peak in next lead_time_min minutes?
        future = [t.floor("1min") for t in times[i: i + lead_time_min]]
        hits = [t for t in future if t in flare_set]

        if hits:
            X_pos.append(window)
            t_pos.append(times[i])
            if want_cls:
                cls_pos.append(max((cls_by_time.get(t, -1) for t in hits), default=-1))
        else:
            X_neg.append(window)
            t_neg.append(times[i])

    # Balance: cap negatives at negative_ratio * positives
    n_pos = len(X_pos)
    if n_pos > 0:
        max_neg = int(n_pos * negative_ratio)
        if len(X_neg) > max_neg:
            rng = np.random.default_rng(42)
            neg_idx = rng.choice(len(X_neg), max_neg, replace=False)
            X_neg = [X_neg[i] for i in neg_idx]
            t_neg = [t_neg[i] for i in neg_idx]

    X = np.stack(X_pos + X_neg, axis=0) if X_pos or X_neg else np.empty((0, window_min, N_FEATURES), dtype=np.float32)
    y = np.concatenate([np.ones(len(X_pos), dtype=np.float32), np.zeros(len(X_neg), dtype=np.float32)])

    if not want_cls:
        if return_times:
            return X, y, pd.DatetimeIndex(t_pos + t_neg)
        return X, y

    # Negatives get -1 (ignored by CrossEntropyLoss); positives get their class idx.
    y_cls = np.concatenate([
        np.asarray(cls_pos, dtype=np.int64),
        np.full(len(X_neg), -1, dtype=np.int64),
    ]) if (X_pos or X_neg) else np.empty((0,), dtype=np.int64)

    if return_times:
        return X, y, y_cls, pd.DatetimeIndex(t_pos + t_neg)
    return X, y, y_cls


def qpp_power(
    series: pd.Series,
    cadence_sec: float,
    window_min: int = 30,
    p_lo_sec: float = 120.0,
    p_hi_sec: float = 1800.0,
) -> pd.Series:
    """
    Rolling "quasi-periodic pulsation" strength for a (hard X-ray) light curve —
    an ISRO PS-15 required signature. For each timestep, take the trailing `window_min`
    minutes, detrend, FFT, and return the fraction of spectral power that falls in
    the QPP period band [p_lo_sec, p_hi_sec]. Output in [0, 1]: ~0 for noise/no
    periodicity, high when a strong oscillation sits in the band.

    Note on cadence: QPPs shorter than 2*cadence are invisible (Nyquist). Run this
    on the *native 1 s* HEL1OS series to catch sub-minute QPPs; on the 1-min
    feature series it only sees periods >= ~2 min. This function is called in
    build_feature_series() on the native HEL1OS cadence and resampled to 1min —
    fully integrated as feature #8 (qpp_power) in the model input.
    """
    vals = series.to_numpy(dtype=float)
    win = max(4, int(window_min * 60 / cadence_sec))
    out = np.full(len(vals), np.nan)

    freqs = np.fft.rfftfreq(win, d=cadence_sec)
    band = (freqs >= 1.0 / p_hi_sec) & (freqs <= 1.0 / p_lo_sec)
    if not band.any():
        return pd.Series(out, index=series.index, name="qpp_power")

    for i in range(win, len(vals) + 1):
        seg = vals[i - win: i]
        if np.isnan(seg).any():
            continue
        # Linear detrend removes a flare-rise ramp that would otherwise leak
        # power across all frequency bins and falsely inflate QPP scores.
        t_idx = np.arange(len(seg), dtype=float)
        slope, intercept = np.polyfit(t_idx, seg, 1)
        seg = seg - (slope * t_idx + intercept)
        power = np.abs(np.fft.rfft(seg)) ** 2
        total = power[1:].sum()   # drop DC
        out[i - 1] = float(power[band].sum() / total) if total > 0 else 0.0

    return pd.Series(out, index=series.index, name="qpp_power")


def normalize_features(X_train: np.ndarray, X_val: np.ndarray | None = None) -> tuple:
    """
    Z-score normalize per feature using training set statistics.
    Returns (X_train_norm, X_val_norm, mean, std).
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    if X_val is not None:
        X_val_norm = (X_val - mean) / std
        return X_train_norm, X_val_norm, mean.squeeze(), std.squeeze()
    return X_train_norm, mean.squeeze(), std.squeeze()


def _demo():
    """Self-check qpp_power: a strong in-band oscillation scores high, noise low.

    Must run at native 1-second cadence so the QPP band [120 s, 1800 s] is a
    strict subset of [0, Nyquist].  At 1-min cadence the band spans all non-DC
    bins and both signals trivially score ~1.0.
    """
    cadence = 1.0   # native HEL1OS cadence (1 s)
    n = 3600        # 1 hour
    idx = pd.date_range("2024-02-01", periods=n, freq="1s")
    t = np.arange(n) * cadence

    # 300 s period sits well inside the [120, 1800] s QPP band → high band-sum fraction
    osc = pd.Series(np.sin(2 * np.pi * t / 300.0), index=idx)
    # White noise: power spread uniformly → tiny fraction falls in the narrow QPP band
    noise = pd.Series(np.random.default_rng(0).normal(0, 1, n), index=idx)

    q_osc = qpp_power(osc, cadence).dropna()
    q_noise = qpp_power(noise, cadence).dropna()
    assert q_osc.mean() > 0.5, f"oscillation QPP too low: {q_osc.mean():.3f}"
    assert q_noise.mean() < q_osc.mean(), "noise should score below a clean oscillation"
    print(f"OK: qpp_power osc={q_osc.mean():.3f} > noise={q_noise.mean():.3f}")


if __name__ == "__main__":
    _demo()
