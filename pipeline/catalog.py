"""
Merge independent SoLEXS and HEL1OS flare catalogs into a master catalog.

Physics-based association:
  1. Flare interval overlap: [start, end] intervals between SoLEXS (SXR) and HEL1OS (HXR)
     must temporally overlap or fall within close proximity.
  2. Neupert effect rise-phase bias: Non-thermal HXR emission peaks during the impulsive
     acceleration phase of the flare, so HEL1OS peak is expected primarily in the rise phase:
         start_time_sxr - 2 min <= peak_time_hxr <= peak_time_sxr + 3 min
  3. Multi-burst aggregation: Multiple impulsive HXR bursts within a single thermal SXR
     flare envelope are consolidated into one canonical event, tracking the dominant
     HXR peak flux, total hxr_burst_count, and primary Neupert lag:
         neupert_lag_sec = (peak_time_sxr - peak_time_hxr).total_seconds()
     (Positive when HXR precedes SXR peak, consistent with the Neupert effect).

Outputs master catalog with columns:
  start_time, peak_time, end_time, duration_sec, rise_time_sec, decay_time_sec,
  solexs_peak_flux, helios_peak_flux, flare_class, hardness_ratio,
  confirmed_multi_instrument, neupert_lag_sec, hxr_burst_count
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pipeline.detector import FlareEvent, events_to_dataframe

MERGE_WINDOW = pd.Timedelta(minutes=3)
DEFAULT_LEAD_MARGIN = pd.Timedelta(minutes=2)
DEFAULT_LAG_MARGIN = pd.Timedelta(minutes=3)

CATALOG_COLUMNS = [
    "start_time",
    "peak_time",
    "end_time",
    "duration_sec",
    "rise_time_sec",
    "decay_time_sec",
    "solexs_peak_flux",
    "helios_peak_flux",
    "flare_class",
    "hardness_ratio",
    "confirmed_multi_instrument",
    "neupert_lag_sec",
    "hxr_burst_count",
]


def upgrade_catalog_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure catalog DataFrame conforms to the enhanced schema with physical timing
    and Neupert metrics. Works cleanly on both new catalogs and legacy parquet files.
    """
    if df.empty:
        return pd.DataFrame(columns=CATALOG_COLUMNS)

    df = df.copy()

    # Normalize timestamp columns
    for col in ["start_time", "peak_time", "end_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
        else:
            df[col] = pd.NaT

    # Duration metrics
    if "duration_sec" not in df.columns:
        dur = (df["end_time"] - df["start_time"]).dt.total_seconds()
        df["duration_sec"] = np.where(df["end_time"].notna() & df["start_time"].notna(), dur, np.nan)
    else:
        df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")

    if "rise_time_sec" not in df.columns:
        rise = (df["peak_time"] - df["start_time"]).dt.total_seconds()
        df["rise_time_sec"] = np.where(df["peak_time"].notna() & df["start_time"].notna(), rise, np.nan)
    else:
        df["rise_time_sec"] = pd.to_numeric(df["rise_time_sec"], errors="coerce")

    if "decay_time_sec" not in df.columns:
        decay = (df["end_time"] - df["peak_time"]).dt.total_seconds()
        df["decay_time_sec"] = np.where(df["end_time"].notna() & df["peak_time"].notna(), decay, np.nan)
    else:
        df["decay_time_sec"] = pd.to_numeric(df["decay_time_sec"], errors="coerce")

    # Multi-instrument & Neupert metrics
    if "confirmed_multi_instrument" not in df.columns:
        df["confirmed_multi_instrument"] = False
    else:
        df["confirmed_multi_instrument"] = df["confirmed_multi_instrument"].fillna(False).astype(bool)

    if "neupert_lag_sec" not in df.columns:
        df["neupert_lag_sec"] = np.nan
    else:
        df["neupert_lag_sec"] = pd.to_numeric(df["neupert_lag_sec"], errors="coerce")

    if "hxr_burst_count" not in df.columns:
        # Infer reasonable burst count from existing flags if missing
        is_hxr = df["flare_class"].eq("HXR") if "flare_class" in df.columns else pd.Series(False, index=df.index)
        df["hxr_burst_count"] = np.where(
            df["confirmed_multi_instrument"] | is_hxr,
            1,
            0,
        )
    else:
        df["hxr_burst_count"] = pd.to_numeric(df["hxr_burst_count"], errors="coerce").fillna(0).astype(int)

    # Standardize remaining numeric and class columns
    for num_col in ["solexs_peak_flux", "helios_peak_flux", "hardness_ratio"]:
        if num_col not in df.columns:
            df[num_col] = np.nan
        else:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    if "flare_class" not in df.columns:
        df["flare_class"] = "Unknown"
    else:
        df["flare_class"] = df["flare_class"].astype(str)

    # Reorder columns to standard canonical schema
    existing_cols = [c for c in CATALOG_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in CATALOG_COLUMNS]
    return df[existing_cols + extra_cols]


def _load_catalog(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, parse_dates=["start_time", "peak_time", "end_time"])


def merge_catalogs(
    solexs_events: list[FlareEvent] | pd.DataFrame,
    helios_events: list[FlareEvent] | pd.DataFrame,
    merge_window: pd.Timedelta = MERGE_WINDOW,
    lead_margin: pd.Timedelta = DEFAULT_LEAD_MARGIN,
    lag_margin: pd.Timedelta = DEFAULT_LAG_MARGIN,
) -> pd.DataFrame:
    """
    Merge SoLEXS and HEL1OS event lists into a single master catalog using interval
    overlap and Neupert rise-phase association.

    Association logic:
      - A HEL1OS burst is associated with a SoLEXS flare if:
        1. Temporal overlap: [start_sxr, end_sxr] overlaps with [start_hxr, end_hxr]
           (allowing open-ended end times), AND
        2. Neupert rise-phase bias: HEL1OS peak is within [start_sxr - lead_margin, peak_sxr + lag_margin]
        OR
        3. Peak proximity: abs(peak_time_sxr - peak_time_hxr) <= merge_window.
      - Many-to-one matches (multiple HXR bursts for one SXR flare) are aggregated:
        - Keeps dominant HEL1OS peak flux.
        - Records total hxr_burst_count.
        - Calculates neupert_lag_sec = (peak_sxr - peak_hxr_dominant).total_seconds().

    Args:
        solexs_events: list of FlareEvent or DataFrame from detect_solexs_flares()
        helios_events: list of FlareEvent or DataFrame from detect_helios_flares()
        merge_window:  fallback peak proximity window (default 3 min)
        lead_margin:   max lead time for HXR peak before SXR start (default 2 min)
        lag_margin:    max lag time for HXR peak after SXR peak (default 3 min)

    Returns:
        Master catalog DataFrame sorted by peak_time.
    """
    if isinstance(solexs_events, list):
        sdf = events_to_dataframe(solexs_events)
    else:
        sdf = solexs_events.copy()

    if isinstance(helios_events, list):
        hdf = events_to_dataframe(helios_events)
    else:
        hdf = helios_events.copy()

    if sdf.empty and hdf.empty:
        return pd.DataFrame(columns=CATALOG_COLUMNS)

    for df in (sdf, hdf):
        if not df.empty:
            for t_col in ["start_time", "peak_time", "end_time"]:
                if t_col in df.columns:
                    df[t_col] = pd.to_datetime(df[t_col])

    parts: list[pd.DataFrame] = []
    matched_helios_indices = set()

    if not sdf.empty:
        sdf = sdf.sort_values("peak_time").reset_index(drop=True)
        # Prefer GOES-calibrated W/m² when present; fall back to raw peak_flux
        if "peak_flux_wm2" in sdf.columns:
            sdf["solexs_peak_flux"] = sdf["peak_flux_wm2"].fillna(sdf["peak_flux"]).astype(float)
        else:
            sdf["solexs_peak_flux"] = sdf["peak_flux"].astype(float)

        solexs_rows: list[dict] = []

        for s_idx, s_row in sdf.iterrows():
            s_start = s_row["start_time"]
            s_peak  = s_row["peak_time"]
            s_end   = s_row["end_time"]
            s_flux  = s_row["solexs_peak_flux"]
            s_cls   = s_row.get("flare_class", "")

            # Effective bounds for interval checking
            s_end_eff = s_end if pd.notna(s_end) else (s_peak + pd.Timedelta(minutes=30))

            matched_h_bursts = []
            if not hdf.empty:
                for h_idx, h_row in hdf.iterrows():
                    h_start = h_row["start_time"]
                    h_peak  = h_row["peak_time"]
                    h_end   = h_row["end_time"]
                    h_end_eff = h_end if pd.notna(h_end) else (h_peak + pd.Timedelta(minutes=15))

                    # Condition A: Neupert rise-phase bias & interval overlap
                    in_rise_window = (h_peak >= s_start - lead_margin) and (h_peak <= s_peak + lag_margin)
                    intervals_overlap = (max(s_start, h_start) <= min(s_end_eff, h_end_eff) + pd.Timedelta(minutes=1))

                    # Condition B: Direct peak proximity fallback
                    close_peaks = abs(s_peak - h_peak) <= merge_window

                    if (in_rise_window and intervals_overlap) or close_peaks:
                        matched_h_bursts.append((h_idx, h_row))

            dur_sec = (s_end - s_start).total_seconds() if pd.notna(s_end) else np.nan
            rise_sec = (s_peak - s_start).total_seconds() if pd.notna(s_start) else np.nan
            decay_sec = (s_end - s_peak).total_seconds() if (pd.notna(s_end) and pd.notna(s_peak)) else np.nan

            if matched_h_bursts:
                # Track matched HEL1OS indices for standalone filtering
                for h_idx, _ in matched_h_bursts:
                    matched_helios_indices.add(h_idx)

                # Identify dominant HXR burst (highest peak flux)
                best_h_idx, best_h_row = max(matched_h_bursts, key=lambda x: x[1]["peak_flux"])
                h_peak_flux = float(best_h_row["peak_flux"])
                h_peak_time = best_h_row["peak_time"]

                # Neupert lag: SXR peak time minus dominant HXR peak time
                neupert_lag = (s_peak - h_peak_time).total_seconds()
                h_burst_count = len(matched_h_bursts)
                confirmed = True
                hardness = (h_peak_flux / s_flux) if (s_flux is not None and s_flux > 0) else np.nan
            else:
                h_peak_flux = np.nan
                neupert_lag = np.nan
                h_burst_count = 0
                confirmed = False
                hardness = np.nan

            solexs_rows.append({
                "start_time": s_start,
                "peak_time": s_peak,
                "end_time": s_end,
                "duration_sec": dur_sec,
                "rise_time_sec": rise_sec,
                "decay_time_sec": decay_sec,
                "solexs_peak_flux": s_flux,
                "helios_peak_flux": h_peak_flux,
                "flare_class": s_cls,
                "hardness_ratio": hardness,
                "confirmed_multi_instrument": confirmed,
                "neupert_lag_sec": neupert_lag,
                "hxr_burst_count": h_burst_count,
            })

        if solexs_rows:
            parts.append(pd.DataFrame(solexs_rows))

    # Add standalone HEL1OS events not matched to any SoLEXS flare
    if not hdf.empty:
        hdf_unmatched = hdf[~hdf.index.isin(matched_helios_indices)].copy()
        if not hdf_unmatched.empty:
            h_rows: list[dict] = []
            for _, h_row in hdf_unmatched.iterrows():
                h_start = h_row["start_time"]
                h_peak  = h_row["peak_time"]
                h_end   = h_row["end_time"]
                dur_sec = (h_end - h_start).total_seconds() if pd.notna(h_end) else np.nan
                rise_sec = (h_peak - h_start).total_seconds() if pd.notna(h_start) else np.nan
                decay_sec = (h_end - h_peak).total_seconds() if (pd.notna(h_end) and pd.notna(h_peak)) else np.nan

                h_rows.append({
                    "start_time": h_start,
                    "peak_time": h_peak,
                    "end_time": h_end,
                    "duration_sec": dur_sec,
                    "rise_time_sec": rise_sec,
                    "decay_time_sec": decay_sec,
                    "solexs_peak_flux": np.nan,
                    "helios_peak_flux": float(h_row["peak_flux"]),
                    "flare_class": "HXR",
                    "hardness_ratio": np.nan,
                    "confirmed_multi_instrument": False,
                    "neupert_lag_sec": np.nan,
                    "hxr_burst_count": 1,
                })
            parts.append(pd.DataFrame(h_rows))

    if not parts:
        return pd.DataFrame(columns=CATALOG_COLUMNS)

    combined = pd.concat(parts, ignore_index=True)
    combined = upgrade_catalog_schema(combined)
    return combined.sort_values("peak_time").reset_index(drop=True)


def append_to_master(
    new_events: pd.DataFrame,
    master_path: str | Path = "data/processed/master_catalog.parquet",
) -> pd.DataFrame:
    """Append new events to the persistent master catalog, deduplicating by peak_time."""
    existing = _load_catalog(master_path)
    if not existing.empty:
        existing = upgrade_catalog_schema(existing)
        combined = pd.concat([existing, new_events], ignore_index=True)
    else:
        combined = new_events.copy()

    if combined.empty:
        return pd.DataFrame(columns=CATALOG_COLUMNS)

    combined = upgrade_catalog_schema(combined)
    combined = combined.sort_values("peak_time").reset_index(drop=True)

    # Deduplicate: within each cluster of events whose peak_times are within 1 min,
    # keep the confirmed event (or the first if none/all confirmed).
    grp = (combined["peak_time"].diff().abs() >= pd.Timedelta(minutes=1)).cumsum()
    confirmed_int = combined["confirmed_multi_instrument"].fillna(False).astype(int)

    combined = (
        combined
        .assign(_grp=grp, _conf=confirmed_int)
        .sort_values(["_grp", "_conf"], ascending=[True, False])
        .drop_duplicates("_grp", keep="first")
        .drop(columns=["_grp", "_conf"])
        .sort_values("peak_time")
        .reset_index(drop=True)
    )

    p = Path(master_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(p, index=False)
    return combined


def load_master_catalog(path: str | Path = "data/processed/master_catalog.parquet") -> pd.DataFrame:
    """Load the master catalog and ensure all modern schema columns are present."""
    df = _load_catalog(path)
    return upgrade_catalog_schema(df)


def catalog_summary(catalog: pd.DataFrame) -> str:
    """Provide a concise summary string of the catalog composition."""
    if catalog.empty:
        return "Empty catalog."

    cat = upgrade_catalog_schema(catalog)
    counts = cat["flare_class"].str[0].value_counts().to_dict()
    multi = int(cat["confirmed_multi_instrument"].sum())

    lag_series = cat.loc[cat["confirmed_multi_instrument"] & cat["neupert_lag_sec"].notna(), "neupert_lag_sec"]
    median_lag = f"{lag_series.median():+.1f}s" if not lag_series.empty else "N/A"

    multi_bursts = int((cat["hxr_burst_count"] > 1).sum())

    summary_items = [
        f"Total: {len(cat)} events",
        " | ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
        f"Multi-instrument: {multi}",
        f"Median Neupert lag: {median_lag}",
        f"Multi-burst events: {multi_bursts}",
    ]
    return " | ".join(summary_items)
