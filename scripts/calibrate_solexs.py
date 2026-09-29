"""
Fit the SoLEXS counts -> GOES-equivalent flux (W/m^2) calibration constant.

Method: in the GOES/SoLEXS overlap window, detect flares in both instruments,
match them by peak time (+/- match_window), then fit

    goes_peak_flux_wm2  ~=  scale * solexs_peak_counts

`scale` is what belongs in config.yaml -> calibration.solexs_counts_to_wm2.
It only affects flare-CLASS labelling (A/B/C/M/X), never detection SNR.

Two estimates are reported:
  - median ratio (robust point estimate -> use this as `scale`)
  - log-log slope (should be ~1 if a single scalar is a good model; a slope far
    from 1 means SoLEXS<->GOES is non-linear and a scalar under/over-calls the
    extreme classes -- a flag to consider energy-resolved calibration later)

Usage:
    python scripts/calibrate_solexs.py                 # all overlap days
    python scripts/calibrate_solexs.py --max-days 60   # quick subset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
import pipeline._compat  # noqa: F401

from pipeline.readers import read_goes_nc
from pipeline.preprocessor import preprocess_goes
from pipeline.detector import detect_solexs_flares, detect_goes_flares


def load_config(path: str = "config.yaml") -> dict:
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def _goes_file_for(date_key: str, goes_dir: Path) -> Path | None:
    hits = sorted(goes_dir.glob(f"sci_xrsf-l2-avg1m_g16_d{date_key}_*.nc"))
    hits += sorted(goes_dir.glob(f"dn_xrsf-l2-avg1m_g16_d{date_key}_*.nc"))
    return hits[0] if hits else None


def collect_pairs(cfg: dict, max_days: int | None, match_window_min: int = 5) -> pd.DataFrame:
    """Return a DataFrame of matched (solexs_counts, goes_wm2) flare-peak pairs."""
    proc_dir = ROOT / cfg["paths"]["processed_data"]
    goes_dir = ROOT / cfg["paths"]["raw_data"] / "goes"
    window = pd.Timedelta(minutes=match_window_min)

    sxr_files = sorted(proc_dir.glob("*_solexs.parquet"))
    rows = []
    used_days = 0

    for sf in sxr_files:
        date_key = sf.name[:8]
        gf = _goes_file_for(date_key, goes_dir)
        if gf is None:
            continue  # no overlapping GOES day
        try:
            sxr = pd.read_parquet(sf)
            solexs_ev = detect_solexs_flares(sxr, cfg)          # peak_flux = counts/s
            goes = preprocess_goes(read_goes_nc(gf), cfg)
            goes_ev = detect_goes_flares(goes, cfg)              # peak_flux = W/m^2
            if not solexs_ev or not goes_ev:
                continue

            gpeaks = pd.DatetimeIndex([e.peak_time for e in goes_ev])
            gflux = np.array([e.peak_flux for e in goes_ev])

            for s in solexs_ev:
                dt = np.abs(gpeaks - s.peak_time)
                j = int(np.argmin(dt))
                if dt[j] <= window and s.peak_flux > 0 and gflux[j] > 0:
                    rows.append({
                        "date": date_key,
                        "peak_time": s.peak_time,
                        "solexs_counts": float(s.peak_flux),
                        "goes_wm2": float(gflux[j]),
                        "dt_sec": float(dt[j].total_seconds()),
                    })
            used_days += 1
            if max_days and used_days >= max_days:
                break
        except Exception as e:
            print(f"  Skip {date_key}: {e}")

    return pd.DataFrame(rows)


def fit_scale(pairs: pd.DataFrame) -> dict:
    """Fit scale from matched pairs. Returns median ratio + log-log slope."""
    ratio = pairs["goes_wm2"].to_numpy() / pairs["solexs_counts"].to_numpy()
    median_scale = float(np.median(ratio))

    # log-log linear fit: log(goes) = slope*log(counts) + b  -> slope ~1 wanted
    lx = np.log10(pairs["solexs_counts"].to_numpy())
    ly = np.log10(pairs["goes_wm2"].to_numpy())
    slope, intercept = np.polyfit(lx, ly, 1)

    return {
        "n_pairs": int(len(pairs)),
        "median_scale": median_scale,
        "loglog_slope": float(slope),
        "loglog_intercept_scale": float(10 ** intercept),  # scale at counts=1
    }


def _demo():
    """Self-check the fit on synthetic pairs with a known scale."""
    rng = np.random.default_rng(0)
    counts = rng.uniform(10, 1e4, 200)
    true_scale = 3.2e-9
    goes = true_scale * counts * rng.uniform(0.8, 1.2, counts.size)  # +/-20% scatter
    df = pd.DataFrame({"solexs_counts": counts, "goes_wm2": goes})
    fit = fit_scale(df)
    assert abs(fit["median_scale"] - true_scale) / true_scale < 0.15, fit
    assert abs(fit["loglog_slope"] - 1.0) < 0.1, fit
    print(f"OK: recovered scale {fit['median_scale']:.2e} (true {true_scale:.2e}), slope {fit['loglog_slope']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-days", type=int, default=None, help="Cap overlap days processed (quick run).")
    ap.add_argument("--match-window", type=int, default=5, help="Peak-match tolerance in minutes.")
    ap.add_argument("--selftest", action="store_true", help="Run the synthetic self-check and exit.")
    ap.add_argument("--out", default=None, help="Optional CSV path to save matched pairs.")
    args = ap.parse_args()

    if args.selftest:
        _demo()
        sys.exit(0)

    cfg = load_config()
    print("Matching SoLEXS <-> GOES flares in the overlap window...")
    pairs = collect_pairs(cfg, args.max_days, args.match_window)

    if pairs.empty:
        print("No matched flare pairs found. Check that GOES and SoLEXS dates overlap "
              "and that both detect flares on those days.")
        sys.exit(1)

    if args.out:
        pairs.to_csv(args.out, index=False)
        print(f"Saved {len(pairs)} pairs -> {args.out}")

    fit = fit_scale(pairs)
    current = cfg.get("calibration", {}).get("solexs_counts_to_wm2", "unset")
    print("\n=== Calibration result ===")
    print(f"  matched pairs        : {fit['n_pairs']}")
    print(f"  current config scale : {current}")
    print(f"  RECOMMENDED scale    : {fit['median_scale']:.3e}   (median goes_wm2 / solexs_counts)")
    print(f"  log-log slope        : {fit['loglog_slope']:.3f}   (want ~1.0; far from 1 => scalar is a rough model)")
    print(f"\nSet in config.yaml:\n  calibration:\n    solexs_counts_to_wm2: {fit['median_scale']:.3e}")
