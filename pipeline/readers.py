"""
FITS readers for Aditya-L1 SoLEXS and HEL1OS Level-1 data.

Each reader returns a pd.DataFrame with a DatetimeIndex (UTC) and named flux columns.

Confirmed FITS structures (from PRADAN Level-1 ZIPs):

SoLEXS:
  ZIP → AL1_SLX_L1_YYYYMMDD_v1.x/SDD2/AL1_SOLEXS_YYYYMMDD_SDD2_L1.lc.gz
  Extension "RATE": TIME (Unix s, float64), COUNTS (float64)

HEL1OS V100/V101 (pre-binned lightcurves):
  ZIP → YYYY/MM/DD/HLS_.../czt/lightcurve_czt{1,2}.fits
                           cdte/lightcurve_cdte{1,2}.fits
  Each file has 5 BINTABLE extensions (energy bands); last = broadband:
    CdTe broadband: 1.8–90 keV   (DETNAM CdTe1/CdTe2)
    CZT  broadband: 18–160 keV   (DETNAM CZT1/CZT2)
  Each extension columns: MJD (float64), ISOT (30-char string), CTR (cts/s), STAT_ERR (cts/s)

HEL1OS V111 (raw event list — no pre-binned lightcurves):
  ZIP → YYYY/MM/DD/HLS_.../events/evt.fits
  4 BINTABLE extensions: CDTE1-EVENTS, CDTE2-EVENTS, CZT1-EVENTS, CZT2-EVENTS
  Each row = one photon event; columns include MJD, ener (keV), utc-isot (23-char string)
  read_helios() detects V111 automatically and bins events into 1-s count rates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits


# ---------------------------------------------------------------------------
# SoLEXS reader
# ---------------------------------------------------------------------------

def _read_solexs_lc(lc_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read TIME (Unix seconds) and COUNTS from a SoLEXS .lc.gz file."""
    with fits.open(lc_path, memmap=False) as hdul:
        table = hdul["RATE"].data
        time_unix = table["TIME"].astype(float)
        counts = table["COUNTS"].astype(float)
    return time_unix, counts


def read_solexs(path: str | Path) -> pd.DataFrame:
    """
    Read SoLEXS Level-1 data.

    Accepts either:
      - A day directory: AL1_SLX_L1_YYYYMMDD_v1.0/   (contains SDD1/ and SDD2/)
      - A direct .lc.gz file path

    Real on-disk structure (confirmed from PRADAN v1.0 ZIPs):
      SDD1/<date>_SDD1_L1.lc.gz  — soft channel (absent in many days; skip if missing)
      SDD2/<date>_SDD2_L1.lc.gz  — primary broadband light curve (86400 rows, 1 Hz)

    Columns TIME = Unix epoch seconds, COUNTS = count rate.

    Returns DataFrame with columns:
        flux_long   — SDD2 count rate (primary, always present)
        flux_short  — SDD1 count rate (when available, else NaN)
    """
    path = Path(path)

    # Resolve to directory containing SDD1/SDD2 subfolders
    if path.suffix in (".gz", ".fits", ".fit", ".lc"):
        day_dir = path.parent.parent   # .../SDD2/file.lc.gz → day_dir
        lc_sdd2 = path if "SDD2" in str(path) else None
        lc_sdd1 = None
    else:
        day_dir = path
        sdd2_files = sorted(day_dir.glob("SDD2/*.lc.gz"))
        sdd1_files = sorted(day_dir.glob("SDD1/*.lc.gz"))
        lc_sdd2 = sdd2_files[0] if sdd2_files else None
        lc_sdd1 = sdd1_files[0] if sdd1_files else None

    if lc_sdd2 is None:
        raise FileNotFoundError(f"No SDD2 .lc.gz found in {day_dir}")

    t_unix, counts_long = _read_solexs_lc(lc_sdd2)
    index = pd.DatetimeIndex(
        pd.to_datetime(t_unix, unit="s", utc=True).tz_convert(None), name="time"
    )

    df = pd.DataFrame({"flux_long": counts_long}, index=index)

    if lc_sdd1 is not None:
        _, counts_short = _read_solexs_lc(lc_sdd1)
        df["flux_short"] = counts_short
    else:
        df["flux_short"] = np.nan

    return df.sort_index()


# ---------------------------------------------------------------------------
# HEL1OS reader
# ---------------------------------------------------------------------------

def _read_helios_lc(fits_path: Path) -> pd.Series:
    """
    Read the broadband (last) extension from one HEL1OS lightcurve FITS file.
    Returns a count-rate Series indexed by UTC datetime strings from ISOT column.
    """
    with fits.open(fits_path, memmap=False) as hdul:
        ext = hdul[-1]
        # ISOT may be bytes (older astropy) or str (newer); normalize to str list
        isot = [
            s.strip() if isinstance(s, str) else s.decode("ascii", errors="replace").strip()
            for s in ext.data["ISOT"]
        ]
        ctr  = ext.data["CTR"].astype(float)

    idx = pd.to_datetime(isot, utc=False)
    s = pd.Series(ctr, index=idx, name=fits_path.stem)
    return s[~s.index.duplicated(keep="first")].sort_index()


def _read_helios_events(evt_path: Path) -> pd.DataFrame:
    """
    Read a V111 HEL1OS evt.fits (raw photon event list) and bin into 1-s count rates.

    Extensions: CDTE1-EVENTS, CDTE2-EVENTS, CZT1-EVENTS, CZT2-EVENTS.
    Each row is one photon; utc-isot gives the timestamp.
    We bin events per second to produce the same hxr_total/hxr_soft/hxr_hard
    output shape as _read_helios_lc().
    """
    ext_groups = {
        "cdte": ["CDTE1-EVENTS", "CDTE2-EVENTS"],
        "czt":  ["CZT1-EVENTS",  "CZT2-EVENTS"],
    }
    cdte_series: list[pd.Series] = []
    czt_series:  list[pd.Series] = []

    with fits.open(evt_path, memmap=False) as hdul:
        ext_names = [hdu.name for hdu in hdul]
        for group_name, exts in ext_groups.items():
            target = cdte_series if group_name == "cdte" else czt_series
            for ext_name in exts:
                if ext_name not in ext_names:
                    continue
                ext = hdul[ext_name]
                if ext.data is None or len(ext.data) == 0:
                    continue
                try:
                    isot = [
                        s.strip() if isinstance(s, str) else s.decode("ascii", errors="replace").strip()
                        for s in ext.data["utc-isot"]
                    ]
                    idx = pd.to_datetime(isot, utc=False)
                    # Each row = 1 photon event; sum per second = count rate (cts/s)
                    s = pd.Series(1.0, index=idx, name=ext_name.lower())
                    s = s.resample("1s").sum()
                    target.append(s)
                except Exception:
                    pass

    all_series = cdte_series + czt_series
    if not all_series:
        raise ValueError(f"No readable event data in {evt_path}")

    df = pd.concat(all_series, axis=1).fillna(0.0)
    df.index.name = "time"

    cdte_cols = [s.name for s in cdte_series]
    czt_cols  = [s.name for s in czt_series]
    all_cols  = [s.name for s in all_series]

    df["hxr_soft"]  = df[cdte_cols].mean(axis=1) if cdte_cols else np.nan
    df["hxr_hard"]  = df[czt_cols].mean(axis=1)  if czt_cols  else np.nan
    df["hxr_total"] = df[all_cols].mean(axis=1)

    return df[["hxr_total", "hxr_soft", "hxr_hard"]].sort_index()


def read_helios(day_dir: str | Path) -> pd.DataFrame:
    """
    Read HEL1OS Level-1 data from an extracted ZIP directory.

    Supports two formats detected automatically:
      V100/V101: pre-binned lightcurve FITS files under cdt/ and czt/
      V111:      raw photon event list at events/evt.fits (binned into 1-s rates)

    Returns DataFrame with DatetimeIndex and columns:
        hxr_total — mean count rate across all broadband detectors
        hxr_soft  — CdTe broadband (1.8–90 keV)
        hxr_hard  — CZT  broadband (18–160 keV)
    """
    day_dir = Path(day_dir)

    # --- V100/V101: pre-binned lightcurve FITS ---
    lc_paths = {
        "cdte1": next(day_dir.rglob("lightcurve_cdte1.fits"), None),
        "cdte2": next(day_dir.rglob("lightcurve_cdte2.fits"), None),
        "czt1":  next(day_dir.rglob("lightcurve_czt1.fits"),  None),
        "czt2":  next(day_dir.rglob("lightcurve_czt2.fits"),  None),
    }

    if any(p is not None for p in lc_paths.values()):
        cdte_series: list[pd.Series] = []
        czt_series:  list[pd.Series] = []

        for name in ("cdte1", "cdte2"):
            if lc_paths[name] is not None:
                try:
                    cdte_series.append(_read_helios_lc(lc_paths[name]))
                except Exception:
                    pass

        for name in ("czt1", "czt2"):
            if lc_paths[name] is not None:
                try:
                    czt_series.append(_read_helios_lc(lc_paths[name]))
                except Exception:
                    pass

        all_series = cdte_series + czt_series
        if all_series:
            resampled = [s.resample("1s").mean() for s in all_series]
            df = pd.concat(resampled, axis=1)
            df.index.name = "time"
            cdte_cols = [s.name for s in cdte_series]
            czt_cols  = [s.name for s in czt_series]
            all_cols  = [s.name for s in all_series]
            df["hxr_soft"]  = df[cdte_cols].mean(axis=1) if cdte_cols else np.nan
            df["hxr_hard"]  = df[czt_cols].mean(axis=1)  if czt_cols  else np.nan
            df["hxr_total"] = df[all_cols].mean(axis=1)
            return df[["hxr_total", "hxr_soft", "hxr_hard"]].sort_index()

    # --- V111: raw event list ---
    evt_path = next(day_dir.rglob("evt.fits"), None)
    if evt_path is not None:
        return _read_helios_events(evt_path)

    raise ValueError(f"No readable HEL1OS data under {day_dir}")


# ---------------------------------------------------------------------------
# GOES XRS reader (for pre-training and validation)
# ---------------------------------------------------------------------------

def read_goes_txt(path: str | Path) -> pd.DataFrame:
    """
    Parse NOAA GOES XRS 1-min average text file (downloaded by scripts/download_goes.py).
    Returns columns: flux_short (0.5–4 Å), flux_long (1–8 Å).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", ":")):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                yr, mo, dy = int(parts[0]), int(parts[1]), int(parts[2])
                hhmm = parts[3].zfill(4)
                hh, mm = int(hhmm[:2]), int(hhmm[2:])
                ts = pd.Timestamp(yr, mo, dy, hh, mm)
                short = float(parts[5])
                long_ = float(parts[6])
                rows.append({"time": ts, "flux_short": short, "flux_long": long_})
            except Exception:
                continue
    df = pd.DataFrame(rows).set_index("time").sort_index()
    # Mask fill values (NOAA uses -1.0e+05 or 9.9e+05)
    df[df < -1e4] = np.nan
    df[df > 1e4] = np.nan
    return df


def read_goes_nc(path: str | Path) -> pd.DataFrame:
    """
    Read a GOES-16 XRS Level-2 1-minute NetCDF4 file from NOAA NCEI.

    Variables used:
        time      — seconds since 2000-01-01 12:00:00 UTC (1440 samples/day)
        xrsa_flux — 0.5–4 Å channel (W/m²)  → flux_short
        xrsb_flux — 1–8 Å channel (W/m²)    → flux_long
        xrsa_flag / xrsb_flag — quality flags (0 = good)
    """
    import netCDF4 as nc4

    TIME_EPOCH = pd.Timestamp("2000-01-01 12:00:00", tz="UTC")

    with nc4.Dataset(path) as ds:
        t_sec = ds.variables["time"][:].data.astype(float)
        xa    = ds.variables["xrsa_flux"][:].data.astype(float)
        xb    = ds.variables["xrsb_flux"][:].data.astype(float)
        flag_a = ds.variables.get("xrsa_flag")
        flag_b = ds.variables.get("xrsb_flag")
        fa = flag_a[:].data.astype(int) if flag_a is not None else np.zeros(len(t_sec), int)
        fb = flag_b[:].data.astype(int) if flag_b is not None else np.zeros(len(t_sec), int)

    index = pd.DatetimeIndex([TIME_EPOCH + pd.Timedelta(seconds=s) for s in t_sec], name="time").tz_localize(None)

    df = pd.DataFrame({"flux_short": xa, "flux_long": xb}, index=index)

    # Mask fill values and bad-quality samples (flag != 0 means quality issue)
    df.loc[xa < 0, "flux_short"] = np.nan
    df.loc[xb < 0, "flux_long"]  = np.nan
    df.loc[fa != 0, "flux_short"] = np.nan
    df.loc[fb != 0, "flux_long"]  = np.nan

    return df.sort_index()


def read_goes_csv(path: str | Path) -> pd.DataFrame:
    """
    Parse a GOES XRS CSV file (e.g. from SWPC JSON-to-CSV exports).
    Expected columns: time_tag, xrsa (short), xrsb (long).
    """
    df = pd.read_csv(path, parse_dates=["time_tag"], index_col="time_tag")
    df.index.name = "time"
    rename = {}
    for c in df.columns:
        cu = c.lower()
        if "xrsa" in cu or "short" in cu or "0.5" in cu:
            rename[c] = "flux_short"
        elif "xrsb" in cu or "long" in cu or "1-8" in cu:
            rename[c] = "flux_long"
    df = df.rename(columns=rename)
    # Add any missing channel as NaN rather than raising KeyError — SWPC exports
    # sometimes contain only one band (e.g. xrsb-only files).
    for col in ("flux_short", "flux_long"):
        if col not in df.columns:
            df[col] = np.nan
    df = df[["flux_short", "flux_long"]]
    df[df < -1e4] = np.nan
    return df.sort_index()
