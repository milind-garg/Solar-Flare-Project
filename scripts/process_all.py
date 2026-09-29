"""
Batch-process all raw Aditya-L1 days into processed Parquet + a master flare catalog.

Closes the pipeline loop: readers → preprocess → detect → merge → append_to_master.
The dashboard's "Master Catalog" table and the fine-tune labels both read the
catalog this produces.

Data layout on disk (from PRADAN):
  SoLEXS:  data/raw/aditya_l1/solexs/level1/YYYY/MM/<SUBDIR>/AL1_SLX_L1_YYYYMMDD_vX.Y.zip
  HEL1OS:  data/raw/aditya_l1/hel1os/level1/YYYY/MM/DD/<SUBDIR>/HLS_YYYYMMDD_HHmmss_<dur>sec_lev1_VXXX.zip

<SUBDIR> varies (N00_0000, UNP_9999, C24_xxxx, …) — all are processed identically.
Multiple ZIPs per HEL1OS day are concatenated into one day DataFrame.
When multiple SoLEXS versions exist for the same date, the highest version wins.
ZIPs are extracted to a temp directory, processed, then deleted.

Usage:
    python scripts/process_all.py                 # process every SoLEXS day found
    python scripts/process_all.py --rebuild       # wipe the master catalog first
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

# Project root on path so `pipeline` imports work when run as a script.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it  # no-op fallback

import pipeline._compat  # noqa: F401 — installs parquet fallback if pyarrow missing
from pipeline.readers import read_solexs, read_helios
from pipeline.preprocessor import preprocess_solexs, preprocess_helios, save_parquet
from pipeline.detector import detect_solexs_flares, detect_helios_flares, events_to_dataframe
from pipeline.catalog import merge_catalogs, append_to_master, catalog_summary, load_master_catalog


def load_config(path: str = "config.yaml") -> dict:
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# ZIP discovery helpers
# ---------------------------------------------------------------------------

def _solexs_version(zip_path: Path) -> tuple[int, int]:
    """Parse vX.Y from AL1_SLX_L1_YYYYMMDD_vX.Y.zip → (X, Y) for sorting."""
    m = re.search(r"_v(\d+)\.(\d+)\.zip$", zip_path.name, re.IGNORECASE)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def collect_solexs_zips(solexs_root: Path) -> dict[str, Path]:
    """
    Walk solexs/level1/**/*.zip and return {YYYYMMDD: best_zip_path}.
    When multiple versions exist for a date, the highest vX.Y wins.
    """
    by_date: dict[str, Path] = {}
    for z in solexs_root.rglob("AL1_SLX_L1_*.zip"):
        m = re.search(r"AL1_SLX_L1_(\d{8})_v", z.name)
        if not m:
            continue
        date_key = m.group(1)
        if date_key not in by_date or _solexs_version(z) > _solexs_version(by_date[date_key]):
            by_date[date_key] = z
    return dict(sorted(by_date.items()))


def collect_helios_zips(hel1os_root: Path) -> dict[str, list[Path]]:
    """
    Walk hel1os/level1/**/*.zip and return {YYYYMMDD: [zip1, zip2, …]}.
    Multiple ZIPs per day are all returned (they cover different time segments).
    """
    by_date: dict[str, list[Path]] = defaultdict(list)
    for z in hel1os_root.rglob("HLS_*.zip"):
        m = re.search(r"HLS_(\d{8})_", z.name)
        if not m:
            continue
        by_date[m.group(1)].append(z)
    return {k: sorted(v) for k, v in sorted(by_date.items())}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_zip(zip_path: Path, dest: Path) -> Path:
    """Extract zip_path into dest/ and return dest."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


def _read_solexs_from_zip(zip_path: Path) -> pd.DataFrame:
    """Extract SoLEXS ZIP to a temp dir, read it, delete the temp dir."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _extract_zip(zip_path, tmp_path)
        # The ZIP contains AL1_SLX_L1_YYYYMMDD_v1.x/ directory at its root.
        day_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and "AL1_SLX" in d.name]
        if not day_dirs:
            # Flat extraction (files directly in root) — pass root as day dir
            day_dir = tmp_path
        else:
            day_dir = day_dirs[0]
        return read_solexs(day_dir)


def _read_helios_from_zips(zip_paths: list[Path]) -> pd.DataFrame:
    """
    Extract and concatenate all HEL1OS ZIPs for a single day.
    Each ZIP covers a ~8-hour segment; we concat and sort by time.
    """
    frames = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i, z in enumerate(zip_paths):
            seg_dir = tmp_path / str(i)
            seg_dir.mkdir()
            _extract_zip(z, seg_dir)
            try:
                frames.append(read_helios(seg_dir))
            except Exception as e:
                print(f"    HEL1OS segment {z.name}: {e}")

    if not frames:
        raise ValueError("No readable HEL1OS segments found")
    df = pd.concat(frames).sort_index()
    # Drop duplicate timestamps from overlapping segments
    df = df[~df.index.duplicated(keep="first")]
    return df


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

def process_all(rebuild: bool = False) -> None:
    cfg = load_config()
    raw_root = ROOT / cfg["paths"]["raw_data"] / "aditya_l1"
    proc_dir = ROOT / cfg["paths"]["processed_data"]
    master_path = ROOT / cfg["paths"]["master_catalog"]
    proc_dir.mkdir(parents=True, exist_ok=True)

    if rebuild and master_path.exists():
        master_path.unlink()
        print(f"Removed existing catalog {master_path}")

    # Discover all ZIPs
    solexs_root = raw_root / "solexs"
    hel1os_root = raw_root / "hel1os"

    solexs_zips = collect_solexs_zips(solexs_root)
    hel1os_zips = collect_helios_zips(hel1os_root) if hel1os_root.exists() else {}

    if not solexs_zips:
        print(f"No SoLEXS ZIPs found under {solexs_root}")
        return

    print(f"Found {len(solexs_zips)} SoLEXS days, {len(hel1os_zips)} HEL1OS days")

    # Collect per-day catalogs in memory; write to disk once at the end
    # instead of calling append_to_master (read→deduplicate→write) for every day.
    day_catalogs: list[pd.DataFrame] = []

    for date_key, sxr_zip in tqdm(solexs_zips.items(), desc="Processing days"):
        out_sxr = proc_dir / f"{date_key}_solexs.parquet"
        if out_sxr.exists() and not rebuild:
            # Already processed — skip unless rebuilding
            solexs_events = events_to_dataframe(
                detect_solexs_flares(pd.read_parquet(out_sxr), cfg)
            )
        else:
            try:
                raw_sxr = _read_solexs_from_zip(sxr_zip)
                sxr = preprocess_solexs(raw_sxr, cfg)
                save_parquet(sxr, out_sxr)
                solexs_events = events_to_dataframe(detect_solexs_flares(sxr, cfg))
            except Exception as e:
                print(f"  Skip SoLEXS {date_key}: {e}")
                continue

        helios_events = None
        if date_key in hel1os_zips:
            out_hxr = proc_dir / f"{date_key}_helios.parquet"
            if out_hxr.exists() and not rebuild:
                hxr = pd.read_parquet(out_hxr)
                helios_events = events_to_dataframe(detect_helios_flares(hxr, cfg))
            else:
                try:
                    raw_hxr = _read_helios_from_zips(hel1os_zips[date_key])
                    hxr = preprocess_helios(raw_hxr, cfg)
                    save_parquet(hxr, out_hxr)
                    helios_events = events_to_dataframe(detect_helios_flares(hxr, cfg))
                except Exception as e:
                    print(f"  Skip HEL1OS {date_key}: {e}")

        master = merge_catalogs(
            solexs_events,
            helios_events if helios_events is not None else events_to_dataframe([]),
        )
        if not master.empty:
            day_catalogs.append(master)

    if day_catalogs:
        append_to_master(pd.concat(day_catalogs, ignore_index=True), master_path)

    catalog = load_master_catalog(master_path)
    print(f"\nMaster catalog -> {master_path}")
    print(catalog_summary(catalog))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-build the master flare catalog from raw Aditya-L1 data.")
    parser.add_argument("--rebuild", action="store_true", help="Delete the existing master catalog before processing.")
    args = parser.parse_args()
    process_all(rebuild=args.rebuild)
