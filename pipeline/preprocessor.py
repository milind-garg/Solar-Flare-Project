"""
Preprocessing for SoLEXS, HEL1OS, and GOES light curves.

Steps:
  1. Resample to uniform cadence (fill short gaps, NaN long gaps)
  2. Estimate background via rolling percentile
  3. Compute noise (rolling MAD)
  4. Clip/mask bad values
  5. Save to Parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(cfg_path: str = "config.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_uniform(df: pd.DataFrame, cadence: str = "1s", gap_fill_limit: int = 30) -> pd.DataFrame:
    """
    Resample to uniform cadence.
    - Forward-fill gaps up to gap_fill_limit samples.
    - Longer gaps become NaN.
    """
    df = df.resample(cadence).mean()
    df = df.ffill(limit=gap_fill_limit)
    return df


# ---------------------------------------------------------------------------
# Background estimation
# ---------------------------------------------------------------------------

def estimate_background(series: pd.Series, window_min: int = 60, cadence_sec: int = 1) -> pd.Series:
    """
    Rolling 60th-percentile background (adapts to solar-cycle drift).
    window_min minutes → window_samples samples.

    Uses a trailing (causal) window so that the background at time t only uses
    data up to and including t — no future data leaks into the nowcasting trigger
    or into excess_norm. center=True would give a slightly smoother estimate for
    post-hoc analysis, but makes the detector non-causal.
    """
    window = window_min * 60 // cadence_sec
    bg = series.rolling(window=window, min_periods=window // 4).quantile(0.60)
    # Forward-fill the leading NaNs where the window hasn't filled yet, then
    # backward-fill in case the very first samples are all NaN.
    return bg.ffill().bfill()


def estimate_noise(series: pd.Series, window_min: int = 10, cadence_sec: int = 1) -> pd.Series:
    """Rolling median absolute deviation (robust noise estimator).

    Trailing window only — same causal constraint as estimate_background.
    """
    window = window_min * 60 // cadence_sec
    median = series.rolling(window, min_periods=window // 4).median()
    mad = (series - median).abs().rolling(window, min_periods=window // 4).median()
    # σ ≈ 1.4826 * MAD for Gaussian noise
    return (mad * 1.4826).ffill().bfill().clip(lower=1e-12)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_solexs(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Preprocess SoLEXS DataFrame (columns: flux_short, flux_long).
    Returns enriched DataFrame with background and noise columns.
    """
    if cfg is None:
        cfg = load_config()

    nc = cfg["nowcasting"]
    bg_win = nc["background_window_min"]

    # Resample to 1-second uniform cadence
    df = resample_uniform(df, cadence="1s", gap_fill_limit=30)

    # Clip negative values (unphysical)
    df = df.clip(lower=0)
    df[df < 1e-12] = np.nan

    for col, bg_col, noise_col in [
        ("flux_short", "bg_short", "noise_short"),
        ("flux_long",  "bg_long",  "noise_long"),
    ]:
        if col not in df.columns:
            continue
        df[bg_col]    = estimate_background(df[col], window_min=bg_win)
        df[noise_col] = estimate_noise(df[col], window_min=10)
        df[f"excess_{col}"] = (df[col] - df[bg_col]).clip(lower=0)

    return df


def preprocess_helios(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Preprocess HEL1OS DataFrame (columns: hxr_total, hxr_soft, hxr_hard).
    """
    if cfg is None:
        cfg = load_config()

    nc = cfg["nowcasting"]
    bg_win = nc["background_window_min"]

    df = resample_uniform(df, cadence="1s", gap_fill_limit=30)
    df = df.clip(lower=0)
    df[df < 1e-12] = np.nan   # suppress numerical-noise near-zero counts

    for col in ["hxr_total", "hxr_soft", "hxr_hard"]:
        if col not in df.columns:
            continue
        df[f"bg_{col}"]    = estimate_background(df[col], window_min=bg_win)
        df[f"noise_{col}"] = estimate_noise(df[col], window_min=10)
        df[f"excess_{col}"] = (df[col] - df[f"bg_{col}"]).clip(lower=0)

    return df


def preprocess_goes(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Preprocess GOES XRS DataFrame (columns: flux_short, flux_long).
    GOES data is already 1-min cadence; resample to 1-min uniform.
    """
    if cfg is None:
        cfg = load_config()

    nc = cfg["nowcasting"]
    bg_win = nc["background_window_min"]

    df = resample_uniform(df, cadence="1min", gap_fill_limit=5)
    df = df.clip(lower=0)
    df[df < 1e-12] = np.nan

    for col, bg_col, noise_col in [
        ("flux_short", "bg_short", "noise_short"),
        ("flux_long",  "bg_long",  "noise_long"),
    ]:
        if col not in df.columns:
            continue
        df[bg_col]    = estimate_background(df[col], window_min=bg_win, cadence_sec=60)
        df[noise_col] = estimate_noise(df[col], window_min=10, cadence_sec=60)
        df[f"excess_{col}"] = (df[col] - df[bg_col]).clip(lower=0)

    return df


# ---------------------------------------------------------------------------
# Save / load processed data
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def process_and_save_solexs(day_dir: str | Path, cfg: dict | None = None) -> Path:
    """day_dir: AL1_SLX_L1_YYYYMMDD_v1.0/ directory extracted from PRADAN ZIP."""
    from pipeline.readers import read_solexs
    df_raw = read_solexs(day_dir)
    df = preprocess_solexs(df_raw, cfg)
    date_str = df.index[0].strftime("%Y%m%d")
    out = Path("data/processed") / f"{date_str}_solexs.parquet"
    save_parquet(df, out)
    return out


def process_and_save_helios(fits_path: str | Path, cfg: dict | None = None) -> Path:
    from pipeline.readers import read_helios
    df_raw = read_helios(fits_path)
    df = preprocess_helios(df_raw, cfg)
    date_str = df.index[0].strftime("%Y%m%d")
    out = Path("data/processed") / f"{date_str}_helios.parquet"
    save_parquet(df, out)
    return out
