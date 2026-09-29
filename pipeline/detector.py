"""
Algorithmic nowcasting: detect solar flares in preprocessed SoLEXS / HEL1OS / GOES light curves.

Algorithm per instrument:
  1. excess = flux - background
  2. Trigger when excess > N_sigma * noise for >= K consecutive samples
  3. Rise-rate guard: dF/dt must also exceed threshold
  4. Find start / peak / end
  5. Classify by GOES-equivalent peak flux (SoLEXS long channel)
  6. Suppress overlapping events within suppression_window

Returns a list of FlareEvent dataclass instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FlareEvent:
    start_time: pd.Timestamp
    peak_time: pd.Timestamp
    end_time: Optional[pd.Timestamp]
    peak_flux: float
    flare_class: str
    instrument: str           # "solexs" | "helios" | "goes"
    channel: str              # e.g. "flux_long", "hxr_total"
    peak_excess: float = 0.0
    rise_rate: float = 0.0
    peak_flux_wm2: float = 0.0  # GOES-equivalent peak flux (peak_flux * class_scale)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_goes(peak_flux_wm2: float) -> str:
    """GOES-equivalent flare classification by peak flux in W/m² (1–8 Å channel)."""
    if peak_flux_wm2 >= 1e-4:
        return f"X{peak_flux_wm2 / 1e-4:.1f}"
    elif peak_flux_wm2 >= 1e-5:
        return f"M{peak_flux_wm2 / 1e-5:.1f}"
    elif peak_flux_wm2 >= 1e-6:
        return f"C{peak_flux_wm2 / 1e-6:.1f}"
    elif peak_flux_wm2 >= 1e-7:
        return f"B{peak_flux_wm2 / 1e-7:.1f}"
    else:
        return f"A{max(0.1, peak_flux_wm2 / 1e-8):.1f}"


def _load_cfg(cfg_path: str = "config.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------

def detect_flares(
    df: pd.DataFrame,
    flux_col: str,
    bg_col: str,
    noise_col: str,
    instrument: str,
    cfg: dict | None = None,
    class_scale: float = 1.0,
) -> list[FlareEvent]:
    """
    Detect flares in a preprocessed time-series DataFrame.

    Args:
        df:          preprocessed DataFrame (DatetimeIndex, uniform cadence)
        flux_col:    column name for observed flux
        bg_col:      column name for background estimate
        noise_col:   column name for noise estimate
        instrument:  label for the FlareEvent record
        cfg:         config dict (loaded from config.yaml if None)
        class_scale: multiply peak flux by this before GOES classification, to
                     convert native units (e.g. SoLEXS counts/s) to W/m². 1.0 for
                     data already in W/m² (GOES). Detection SNR stays in native units.

    Returns:
        List of FlareEvent instances, sorted by start_time.
    """
    if cfg is None:
        cfg = _load_cfg()

    nc = cfg["nowcasting"]
    sigma_thr   = nc["sigma_threshold"]
    min_consec  = nc["min_consecutive_samples"]
    rise_thr    = nc["rise_rate_threshold"]
    suppress_m  = nc["suppression_window_min"]
    end_factor  = nc["end_factor"]
    # rise_rate_threshold is defined in W/m²/s; convert to native flux units so the
    # guard behaves consistently across instruments (fix 7).
    rise_thr_native = rise_thr / class_scale if class_scale else rise_thr

    flux    = df[flux_col].values
    bg      = df[bg_col].values
    noise   = df[noise_col].values
    times   = df.index

    # Cadence in seconds
    if len(times) > 1:
        cadence_sec = (times[1] - times[0]).total_seconds()
    else:
        return []

    # Step 1: excess signal
    excess = flux - bg

    # Step 2: trigger mask (excess > sigma * noise AND not NaN)
    trigger = (excess > sigma_thr * noise) & np.isfinite(flux) & np.isfinite(bg)

    # Step 3: rise-rate guard — backward difference only (causal: no future data).
    # np.gradient uses centered differences (dflux[i] depends on flux[i+1]), which
    # leaks one future sample into the nowcasting trigger and inflates lead times.
    dflux_dt = np.empty_like(flux, dtype=float)
    dflux_dt[0] = 0.0
    dflux_dt[1:] = (flux[1:] - flux[:-1]) / cadence_sec
    trigger &= (dflux_dt > rise_thr_native)

    # Step 4: find contiguous trigger windows >= min_consec
    events: list[FlareEvent] = []
    in_flare = False
    start_idx = 0

    for i in range(len(trigger)):
        if trigger[i] and not in_flare:
            # Check if next min_consec-1 points also trigger
            end_check = min(i + min_consec, len(trigger))
            if trigger[i:end_check].sum() >= min_consec:
                in_flare = True
                start_idx = i
        elif in_flare and not trigger[i]:
            # Check if flux has returned to background level
            if flux[i] <= end_factor * bg[i] or np.isnan(flux[i]):
                # Find peak within [start_idx, i]
                window_flux = np.where(np.isfinite(flux[start_idx:i]), flux[start_idx:i], -np.inf)
                if len(window_flux) == 0 or window_flux.max() <= 0:
                    in_flare = False
                    continue
                peak_idx = start_idx + int(np.argmax(window_flux))
                peak_flux = float(flux[peak_idx])
                peak_excess = float(excess[peak_idx]) if np.isfinite(excess[peak_idx]) else 0.0
                max_rise = float(np.nanmax(dflux_dt[start_idx:i+1]))

                peak_wm2 = peak_flux * class_scale
                flare_class = _classify_goes(peak_wm2) if instrument in ("solexs", "goes") else "HXR"

                events.append(FlareEvent(
                    start_time=times[start_idx],
                    peak_time=times[peak_idx],
                    end_time=times[i] if i < len(times) else times[-1],
                    peak_flux=peak_flux,
                    flare_class=flare_class,
                    instrument=instrument,
                    channel=flux_col,
                    peak_excess=peak_excess,
                    rise_rate=max_rise,
                    peak_flux_wm2=peak_wm2,
                ))
                in_flare = False

    # Handle flare still active at end of data
    if in_flare:
        window_flux = np.where(np.isfinite(flux[start_idx:]), flux[start_idx:], -np.inf)
        if len(window_flux) > 0 and window_flux.max() > 0:
            peak_idx = start_idx + int(np.argmax(window_flux))
            peak_flux = float(flux[peak_idx])
            peak_excess = float(excess[peak_idx]) if np.isfinite(excess[peak_idx]) else 0.0
            peak_wm2 = peak_flux * class_scale
            flare_class = _classify_goes(peak_wm2) if instrument in ("solexs", "goes") else "HXR"
            events.append(FlareEvent(
                start_time=times[start_idx],
                peak_time=times[peak_idx],
                end_time=None,
                peak_flux=peak_flux,
                flare_class=flare_class,
                instrument=instrument,
                channel=flux_col,
                peak_excess=peak_excess,
                rise_rate=0.0,
                peak_flux_wm2=peak_wm2,
            ))

    # Step 5: suppression — within a window, keep only the strongest event
    events = _suppress_overlapping(events, window_minutes=suppress_m)

    return sorted(events, key=lambda e: e.start_time)


def _suppress_overlapping(events: list[FlareEvent], window_minutes: int) -> list[FlareEvent]:
    """Remove weaker overlapping events within window_minutes of a stronger one."""
    if not events:
        return events

    events = sorted(events, key=lambda e: e.peak_flux, reverse=True)
    keep = []
    window = pd.Timedelta(minutes=window_minutes)

    for ev in events:
        # Check if any already-kept event is within window of this one
        overlap = any(
            abs(ev.peak_time - k.peak_time) < window
            for k in keep
        )
        if not overlap:
            keep.append(ev)

    return keep


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def detect_solexs_flares(df: pd.DataFrame, cfg: dict | None = None) -> list[FlareEvent]:
    """Detect flares in preprocessed SoLEXS data using the long channel (1–8 Å)."""
    if cfg is None:
        cfg = _load_cfg()
    scale = cfg.get("calibration", {}).get("solexs_counts_to_wm2", 1.0)
    return detect_flares(
        df,
        flux_col="flux_long",
        bg_col="bg_long",
        noise_col="noise_long",
        instrument="solexs",
        cfg=cfg,
        class_scale=scale,
    )


def detect_helios_flares(df: pd.DataFrame, cfg: dict | None = None) -> list[FlareEvent]:
    """Detect flares in preprocessed HEL1OS data using total HXR count rate."""
    return detect_flares(
        df,
        flux_col="hxr_total",
        bg_col="bg_hxr_total",
        noise_col="noise_hxr_total",
        instrument="helios",
        cfg=cfg,
    )


def detect_goes_flares(df: pd.DataFrame, cfg: dict | None = None) -> list[FlareEvent]:
    """Detect flares in preprocessed GOES XRS data using the long channel (1–8 Å)."""
    return detect_flares(
        df,
        flux_col="flux_long",
        bg_col="bg_long",
        noise_col="noise_long",
        instrument="goes",
        cfg=cfg,
    )


def events_to_dataframe(events: list[FlareEvent]) -> pd.DataFrame:
    """Convert list of FlareEvent to a DataFrame for catalog storage."""
    if not events:
        return pd.DataFrame(columns=[
            "start_time", "peak_time", "end_time", "peak_flux", "peak_flux_wm2",
            "flare_class", "instrument", "channel", "peak_excess", "rise_rate",
        ])
    return pd.DataFrame([
        {
            "start_time": e.start_time,
            "peak_time": e.peak_time,
            "end_time": e.end_time,
            "peak_flux": e.peak_flux,
            "peak_flux_wm2": e.peak_flux_wm2,
            "flare_class": e.flare_class,
            "instrument": e.instrument,
            "channel": e.channel,
            "peak_excess": e.peak_excess,
            "rise_rate": e.rise_rate,
        }
        for e in events
    ])


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _demo():
    """Quick sanity check: inject a synthetic flare into a flat background."""
    import numpy as np

    n = 3600  # 1 hour at 1-second cadence
    times = pd.date_range("2024-01-01", periods=n, freq="1s")
    bg_val = 1e-7
    noise_val = 5e-9
    flux = np.full(n, bg_val) + np.random.normal(0, noise_val, n)

    # Inject a C3 flare at t=1800s: 60-second rise to 3e-6, 120-second decay
    peak_idx = 1800
    flare_peak = 3e-6   # solidly C-class (3× above C boundary)
    for i in range(peak_idx - 60, peak_idx + 120):
        if 0 <= i < n:
            if i < peak_idx:
                flux[i] += (flare_peak - bg_val) * (i - (peak_idx - 60)) / 60
            else:
                flux[i] += (flare_peak - bg_val) * np.exp(-(i - peak_idx) / 40)

    df = pd.DataFrame({
        "flux_long": flux,
        "bg_long": np.full(n, bg_val),
        "noise_long": np.full(n, noise_val),
    }, index=times)

    cfg = {
        "nowcasting": {
            "sigma_threshold": 3.0,
            "min_consecutive_samples": 5,
            "rise_rate_threshold": 1e-11,
            "suppression_window_min": 15,
            "end_factor": 1.5,
        }
    }

    events = detect_flares(df, "flux_long", "bg_long", "noise_long", "goes", cfg)
    assert len(events) == 1, f"Expected 1 flare, got {len(events)}"
    assert events[0].flare_class.startswith("C"), f"Expected C-class, got {events[0].flare_class}"
    print(f"OK: detected {events[0].flare_class} flare at {events[0].peak_time}")


if __name__ == "__main__":
    _demo()
