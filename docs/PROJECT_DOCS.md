# Aditya-L1 Solar Flare AI — Complete Project Reference

> **5th Semester Cornerstone Project**
> Forecasting and/or Nowcasting of Solar Flares using combined Soft and Hard X-ray data from ISRO's Aditya-L1 spacecraft.

---

## Table of Contents

1. [What Is a Solar Flare?](#1-what-is-a-solar-flare)
2. [What We Are Building](#2-what-we-are-building)
3. [Instruments & Data Sources](#3-instruments--data-sources)
4. [Dataset](#4-dataset)
5. [Feature Engineering — The 8 Input Features](#5-feature-engineering--the-8-input-features)
6. [Model Architecture](#6-model-architecture)
7. [Pipeline Walkthrough (Stages 1–8)](#7-pipeline-walkthrough-stages-18)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Setup & Run — Step-by-Step](#9-setup--run--step-by-step)
10. [Glossary](#10-glossary)

---

## 1. What Is a Solar Flare?

A solar flare is a sudden, intense release of magnetic energy from the Sun's atmosphere. In minutes to hours it converts stored magnetic energy into radiation, heat, and accelerated particles — all of which can affect Earth.

### Physical Mechanism

Flares occur in **active regions (ARs)** — areas of complex, twisted magnetic fields near sunspots. When oppositely-directed field lines are forced together (**magnetic reconnection**), the topology snaps, releasing energy comparable to millions of hydrogen bombs simultaneously.

The release proceeds in phases:

| Phase | Dominant emission | Relevant instrument |
|---|---|---|
| **Pre-flare** | Subtle SXR increase; HXR precursors | HEL1OS (hard X-ray onset) |
| **Impulsive** | Non-thermal particle acceleration; QPPs | HEL1OS (fires first) |
| **Gradual** | Thermal plasma cooling | SoLEXS (soft X-ray peak & decay) |

### Why It Matters

- **Geomagnetic storms** → power-grid failures
- **Radio blackouts** → GPS and aviation disruption
- **Energetic particle radiation** → satellite damage
- **30-minute advance warning** allows protective measures (satellite safe-mode, aviation rerouting)

### GOES Flare Classification

The universal standard classifies flares by their peak **1–8 Å soft X-ray flux** measured at Earth (W/m²):

| Class | Peak Flux (W/m²) | Relative Energy | Notes |
|---|---|---|---|
| **A** | 10⁻⁸ – 10⁻⁷ | Quiet Sun baseline | Continuous background |
| **B** | 10⁻⁷ – 10⁻⁶ | 10× A | Daily during solar minimum |
| **C** | 10⁻⁶ – 10⁻⁵ | 100× A | Daily during solar maximum |
| **M** | 10⁻⁵ – 10⁻⁴ | 1,000× A | Weekly during solar maximum |
| **X** | ≥ 10⁻⁴ | 10,000× A | Rare; extreme events |

Each class has decimal subdivisions (C3.5, M1.2, X9.0, …). The SoLEXS calibration constant used in this project is `solexs_counts_to_wm2 = 1.806e-8` counts/(W/m²).

### Quasi-Periodic Pulsations (QPPs)

QPPs are **repetitive intensity variations** in flare emission, typically in hard X-rays, with periods of 2–30 minutes. They encode information about the magnetic reconnection process and are a key project requirement. Our `qpp_power` feature (feature #7) quantifies QPP strength as the fraction of spectral power in the 120–1800 second period band.

---

## 2. What We Are Building

A complete end-to-end ML system with three components:

### Nowcasting — Real-time Flare Detection
- `pipeline/detector.py`
- σ-threshold detector that identifies flares as they happen in SoLEXS soft X-ray flux
- Runs on each new 1-minute data window; produces `FlareEvent` dataclasses

### Forecasting — 30-minute Advance Prediction
- `model/cnn_lstm.py`, `model/train.py`
- CNN-LSTM model predicts the probability of a flare in the **next 30 minutes**
- Uses a 60-minute history window of 8 engineered features: shape `(batch, 60, 8)`
- Current results (Fine-tune #3): **AUC = 0.818, TSS = 0.361, HSS = 0.310, P90 = 27 min**

### Dashboard — Live Monitoring
- `dashboard/app.py`
- Plotly/Dash web app showing: flux light curves, flare probability timeline, class forecasts (A/B/C/M/X), master flare catalog

### The Transfer Learning Strategy

Aditya-L1 launched September 2023 and has limited labeled data. GOES-16 has been continuously monitoring since 2016 with the same X-ray bands.

```
Phase 1 — Pretrain: CNN-LSTM on 20+ years of GOES-16 XRS data (2020–2023)
             ↓  (save best checkpoint)
Phase 2 — Fine-tune: Transfer to SoLEXS data; freeze CNN; train LSTM + heads at lr=1e-4
             ↓
Final model: Generalizes from GOES physics → Aditya-L1 cadence/calibration
```

---

## 3. Instruments & Data Sources

### SoLEXS — Solar Low Energy X-ray Spectrometer
- **Spacecraft:** Aditya-L1 (ISRO)
- **Band:** Soft X-rays (0.5–10 keV)
- **Dynamic range:** A-class through X-class
- **Native cadence:** ~1 second
- **Units:** counts/s (Silicon Drift Detectors)
- **File format:** FITS `.lc.gz` inside ZIP archives
- **Data access:** ISSDC PRADAN portal (`pradan1.issdc.gov.in`)
- **Physics:** Gradual thermal plasma heating; primary detection channel

### HEL1OS — High Energy L1 Orbiting X-ray Spectrometer
- **Spacecraft:** Aditya-L1 (ISRO)
- **Band:** Hard X-rays (10–150 keV)
- **Native cadence:** ~1 second
- **Units:** counts/s
- **File format:** FITS ZIP archives (multiple ZIPs per day covering ~8-hour segments)
- **Data access:** ISSDC PRADAN portal
- **Physics:** Non-thermal particle acceleration; fires **first** during the impulsive phase; QPPs live here

### GOES-16 XRS — X-Ray Sensor, 1-minute averages
- **Spacecraft:** GOES-16 (NOAA)
- **Band A:** 0.5–4 Å (short)
- **Band B:** 1–8 Å (long — the global standard)
- **Cadence:** 1 minute (Level-2 science product)
- **Units:** W/m²
- **File format:** NetCDF4 (`.nc`)
- **Archive:** NOAA NCEI (`data.ngdc.noaa.gov`) — 2017–present; ~12–16 month lag to science-grade
- **Coverage used:** 2020–2023 (~1461 days) for pretraining

> **Why L1?** The Sun-Earth L1 Lagrange point is 1.5 million km from Earth toward the Sun. No Earth occultations — uninterrupted 24/7 solar viewing.

---

## 4. Dataset

### Directory Layout

```
data/
├── raw/
│   ├── goes/                           # Downloaded by scripts/download_goes.py
│   │   └── sci_xrsf-l2-avg1m_g16_dYYYYMMDD_v*.nc
│   └── aditya_l1/
│       ├── solexs/level1/YYYY/MM/<SUBDIR>/
│       │   └── AL1_SLX_L1_YYYYMMDD_vX.Y.zip
│       └── hel1os/level1/YYYY/MM/DD/<SUBDIR>/
│           └── HLS_YYYYMMDD_HHmmss_<dur>sec_lev1_VXXX.zip
└── processed/                          # Built by scripts/process_all.py
    ├── YYYYMMDD_solexs.parquet
    ├── YYYYMMDD_helios.parquet
    └── master_catalog.parquet          # Unified flare catalog
```

### Dataset Summary

| Dataset | Coverage | Size/Format | Key Columns |
|---|---|---|---|
| GOES-16 XRS 1min | 2020–2023 (~1461 days) | ~2 GB NetCDF4 | `flux_short`, `flux_long` (W/m²) |
| SoLEXS L1 (processed) | Feb 2024–present | ~50 MB/day Parquet | `flux_short`, `flux_long`, `bg_long` (counts/s) |
| HEL1OS L1 (processed) | Feb 2024–present (patchy) | ~30 MB/day Parquet | `hxr_total` (counts/s) |
| Master Flare Catalog | All processed days | <1 MB Parquet | `start_time`, `peak_time`, `end_time`, `peak_flux`, `class`, `source` |

### Calibration

SoLEXS counts/s are converted to W/m² equivalents using:
```python
solexs_counts_to_wm2 = 1.806e-8
```
This value affects flare classification labels (A/B/C/M/X) but **never** detection sensitivity, which operates on normalized flux ratios.

### Class Imbalance

Flare datasets are severely class-imbalanced — positive events are rare. The training pipeline applies a **negative-to-positive ratio cap of 5:1** in `build_windows()`. Evaluation metrics (TSS, HSS, AUC) are chosen to be insensitive to this imbalance.

---

## 5. Feature Engineering — The 8 Input Features

Every 1-minute timestep is described by 8 engineered features. A 60-minute window forms one model input: shape `(batch, 60, 8)`.

Source file: `model/features.py`

| # | Feature Name | Description |
|---|---|---|
| 0 | `log_flux_short` | log₁₀ of SXR 0.5–4 Å channel. Floored at `FLUX_FLOOR = 1e-9`. Tracks hotter coronal plasma. |
| 1 | `log_flux_long` | log₁₀ of SXR 1–8 Å channel — the primary GOES-equivalent. Defines the A/B/C/M/X class. |
| 2 | `log_hxr_total` | log₁₀ of HEL1OS broadband hard X-ray counts/s. Floored at `HXR_FLOOR = 0.1`. Zero-padded when HEL1OS unavailable. |
| 3 | `dflux_1min` | 1-minute finite difference of `flux_long`. Captures rapid impulsive rise — strongest single precursor. |
| 4 | `dflux_5min` | 5-minute flux derivative (per-minute rate). Captures slower pre-flare build-up; addresses ISRO "pre-flare sudden increases" requirement. |
| 5 | `hardness_ratio` | `hxr_total / flux_long`. Ratio of hard to soft emission — hardens before the soft peak, signaling non-thermal acceleration onset. |
| 6 | `excess_norm` | `(flux_long − bg_long) / bg_long`. Normalized excess above rolling background; SNR-like quantity. |
| 7 | `qpp_power` | QPP band-power fraction. Rolling 30-min FFT on native 1-second HEL1OS data, linearly detrended, band [120–1800 s]. Output ∈ [0,1]: ~0 = noise, ~1 = strong oscillation. Key project feature. |

### QPP Cadence Note

`qpp_power` is intentionally computed on the **native 1-second** HEL1OS cadence *before* resampling to 1-minute. At 1-minute cadence, the Nyquist frequency is 0.5 cycles/min (period = 2 min), which aliases all sub-2-minute QPPs. Computing on native data, then downsampling, preserves the full [120s, 1800s] band. See `build_feature_series()` in `model/features.py`.

---

## 6. Model Architecture

### CNN-LSTM Forecaster (`model/cnn_lstm.py`)

Total parameters: ~180K. Small enough for embedded / L1 deployment. Research benchmarks show this beats 366M-parameter foundation models on this task.

```
Input:          (B, 60, 8)            # Batch × 60-min window × 8 features
                ↓ permute → (B, 8, 60)

Conv1d Block 1: 8 → 32 channels, kernel=3, pad=1
                BatchNorm + ReLU + MaxPool(2)
                → (B, 32, 30)

Conv1d Block 2: 32 → 64 channels, kernel=3, pad=1
                BatchNorm + ReLU + MaxPool(2)
                → (B, 64, 15)
                ↓ permute → (B, 15, 64)

LSTM:           input=64, hidden=128, num_layers=2, dropout=0.3
                → take last timestep → (B, 128)

Binary Head:    Linear(128 → 1)       # BCEWithLogitsLoss
                → P(flare in next 30 min)

Class Head:     Linear(128 → 5)       # CrossEntropyLoss(ignore_index=-1)
  (optional)    → P(A), P(B), P(C), P(M), P(X)
```

### Training Details

#### Pretrain (GOES-16)
| Parameter | Value |
|---|---|
| Dataset | ~1461 GOES-16 days (2020–2023) |
| Window | 60-min input, 30-min lead time |
| Negative ratio cap | 5:1 |
| Optimizer | Adam, lr=1e-3, weight decay=1e-4 |
| Scheduler | ReduceLROnPlateau (patience=5) |
| Batch size | 256 |
| Max epochs | 100 (early stopping) |
| Split | 80/20 train/val (time-based) |

#### Fine-tune (Aditya-L1)
- Freeze CNN layers; train LSTM + heads only
- Or unfreeze all with lr=1e-4 (10× lower than pretrain)
- Labels from per-day σ-threshold detector (no catalog look-ahead → no leakage)

#### Normalization
Z-score per feature using training-set mean/std. Stats saved alongside the checkpoint at `model/checkpoints/best.pt` and loaded at every inference call.

#### Inference (Dashboard)
The dashboard batches **all 1440 windows** of the day into one forward pass — not a per-minute loop. Thread-safe model cache with double-checked locking; `_MODEL` is assigned last after all other globals are populated.

---

## 7. Pipeline Walkthrough (Stages 1–8)

### Stage 1 — Raw Data Acquisition
**File:** `scripts/download_goes.py` / PRADAN portal (manual for Aditya)

GOES-16 XRS NetCDF4 files are downloaded from NOAA NCEI via the science-grade archive with automatic fallback to NRT. HTML directory listings are parsed; the highest-version file per day is kept. Aditya-L1 ZIPs require manual download from PRADAN after registration.

```bash
python scripts/download_goes.py --start 2020-01-01 --end 2023-12-31
```

### Stage 2 — Reading Raw Files
**File:** `pipeline/readers.py`

| Function | Input | Output |
|---|---|---|
| `read_goes_nc(path)` | `.nc` file | DataFrame: `flux_short`, `flux_long` (W/m²) |
| `read_solexs(day_dir)` | extracted SoLEXS ZIP dir | DataFrame: counts/s, DatetimeIndex |
| `read_helios(seg_dir)` | extracted HEL1OS dir | DataFrame: `hxr_total` counts/s; segments concatenated, duplicates dropped |

### Stage 3 — Preprocessing
**File:** `pipeline/preprocessor.py`

- **GOES:** clip negatives; resample to exactly 1-min; compute rolling 5th-percentile background (`bg_long`) over 30-min window.
- **SoLEXS:** scale by `solexs_counts_to_wm2`; clip below 0; set values < 1e-12 to NaN.
- **HEL1OS:** clip below 0; same 1e-12 floor.
- Output saved via `save_parquet()` to `data/processed/`.

### Stage 4 — Flare Detection (Nowcasting)
**File:** `pipeline/detector.py`

Per-day σ-threshold detector:
1. Compute per-minute SNR: `(flux − bg) / bg`
2. Flag minutes exceeding `detection_sigma = 5.0` standard deviations above rolling background
3. Apply rise guard: require 3 consecutive rising minutes before confirming onset (reduces noise spikes)
4. `_classify_goes(peak_flux)` assigns A/B/C/M/X class; floored at A0.1 (never "A0.0")

Functions: `detect_solexs_flares()`, `detect_helios_flares()`, `events_to_dataframe()`

### Stage 5 — Catalog Merging
**File:** `pipeline/catalog.py`

Fuses per-day SoLEXS and HEL1OS event lists using `pd.merge_asof` — O((n+m) log(n+m)) nearest-timestamp join:

- SoLEXS event matched to nearest HEL1OS event within ±3 minutes → **"confirmed" multi-instrument event**
- Unmatched HEL1OS events → "HEL1OS-only"
- Unmatched SoLEXS events → "SoLEXS-only"
- `append_to_master()` deduplicates by `peak_time` before writing

```bash
python scripts/process_all.py          # builds master_catalog.parquet
python scripts/process_all.py --rebuild  # wipes catalog and reprocesses
```

### Stage 6 — Feature Engineering
**File:** `model/features.py` — `build_feature_series(df_sxr, df_hxr)`

1. SXR and HXR resampled to 1-min via mean
2. Log transforms with noise floors
3. Derivatives: 1-min and 5-min finite differences of `flux_long`
4. Hardness ratio and `excess_norm` computed per timestep
5. `qpp_power()` runs on native 1-second HEL1OS data with linear detrending, then downsampled to 1-min

Output: DataFrame with shape `(N_minutes, 8)`, columns = `FEATURE_NAMES`

### Stage 7 — Window Building & Training
**File:** `model/features.py` — `build_windows()` / `model/train.py`

`build_windows()` creates labeled sliding windows:
- Slide a 60-min window across the feature series
- **Label = 1** if any flare peak falls within the next 30 minutes
- Skip windows with >30% NaN values
- Cap negatives at 5:1 ratio (random subsample, seed=42)
- Optional `flare_classes` arg returns `y_cls` array for multi-class head training

`train.py` orchestrates the two-phase loop:
```
load data → build windows → Z-score normalize → Adam optimizer →
validate per epoch → save checkpoint on best validation AUC
```

```bash
python model/train.py --phase pretrain
python model/train.py --phase finetune
```

### Stage 8 — Evaluation & Dashboard
**Files:** `scripts/evaluate_model.py` / `dashboard/app.py`

**Evaluation:**
- Uses the last 20% of SoLEXS days (held out; time-based split)
- Labels generated per-day by the **detector** (not catalog) — avoids label leakage
- Reports threshold sweep (TPR / FAR / TSS / HSS / F1 / AUC) and P10/P50/P90 lead times

```bash
python scripts/evaluate_model.py --holdout-frac 0.2
```

**Dashboard tabs:**
1. **Live Flux** — SoLEXS soft X-ray light curve with detected flare markers
2. **Probability** — 30-min ahead flare probability timeline with alert threshold bands
3. **Class Distribution** — A/B/C/M/X probability breakdown from multi-class head
4. **Master Catalog** — All detected flares with class, peak flux, source, confirmation status

```bash
python dashboard/app.py
# Open: http://localhost:8050
```

---

## 8. Evaluation Metrics

### Why Not Accuracy?

A model that always predicts "no flare" achieves >99% accuracy because flares are rare. The following metrics are **robust to class imbalance**.

### Confusion Matrix Notation

```
TP — Flare predicted AND occurred
FP — Flare predicted but did NOT occur (false alarm)
FN — Flare occurred but NOT predicted (miss)
TN — No flare predicted AND none occurred
TPR — TP / (TP + FN)   true positive rate / recall
FAR — FP / (FP + TN)   false alarm rate
```

### Key Metrics

| Metric | Formula | Range | Current | Meaning |
|---|---|---|---|---|
| **TSS** | TPR − FAR | −1 to +1 | **0.361** | Community standard; 0=random, 1=perfect |
| **HSS** | (TP·TN − FP·FN) / denom | −∞ to +1 | **0.310** | Skill over a random forecast |
| **AUC** | Area under ROC | 0.5 to 1.0 | **0.818** | Threshold-independent; 0.5=random |
| **P90 Lead** | 90th percentile of lead times | — | **27 min** | 90% of true positives arrive ≥27 min early |

ISRO target for P90 lead time: 30 min. Current gap: −3 min.

### ISRO Requirements vs. Status

| Requirement | Status | Implementation |
|---|---|---|
| Nowcast / detect flares | ✅ Done | `detector.py` σ-threshold + rise guard |
| Unified SoLEXS+HEL1OS catalog | ✅ Done | `catalog.py` ±3 min merge |
| Forecast P(flare, next N min) | ✅ Done | CNN-LSTM, AUC=0.818 |
| Multi-class intensity (A/B/C/M/X) | ✅ Done | `class_head` Linear(128→5) |
| High TPR / low FAR | ✅ Done | TSS=0.361, HSS=0.310 |
| 15–30 min lead time | 🟡 Close | P90=27 min (−3 min short) |
| Pre-flare sudden increases | 🟡 Partial | `dflux_1min`, `dflux_5min` features |
| QPPs in hard X-rays | 🟡 Partial | `qpp_power` feature (needs HEL1OS data) |

---

## 9. Setup & Run — Step-by-Step

### Prerequisites

- Python 3.10+ (3.11 recommended)
- 8 GB RAM minimum; 16 GB recommended
- ~30 GB free disk space
- CUDA GPU optional (CPU works; ~3× slower training)
- Internet access for GOES download and pip
- **Aditya-L1 data:** ISSDC PRADAN account (free at `pradan1.issdc.gov.in`)

---

### Step 1 — Clone and create virtual environment

```bash
git clone https://github.com/<your-repo>/solar-flare-project.git
cd solar-flare-project

# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

---

### Step 2 — Install dependencies

```bash
pip install --upgrade pip

# CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU (check pytorch.org for correct CUDA version)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install pandas numpy scipy scikit-learn pyarrow
pip install astropy netCDF4 requests tqdm pyyaml
pip install plotly dash dash-bootstrap-components
```

> If `pyarrow` is unavailable, `pipeline/_compat.py` installs a Pickle fallback automatically — Parquet read/write still works.

---

### Step 3 — Review config.yaml

Open `config.yaml` and verify paths match your system. All paths are relative to the project root.

```yaml
paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  master_catalog: "data/processed/master_catalog.parquet"
  model_checkpoint: "model/checkpoints/best.pt"

goes:
  base_url: "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/data/xrsf-l2-avg1m_science/"

forecasting:
  input_window_min: 60
  lead_time_min: 30
  alert_threshold: 0.5
  detection_sigma: 5.0
```

---

### Step 4 — Download GOES-16 training data

Downloads ~2 GB of NetCDF4 files from NOAA. Expect 30–90 minutes.

```bash
# Full 4-year pretrain dataset (recommended)
python scripts/download_goes.py --start 2020-01-01 --end 2023-12-31

# Shorter test run (1 month, ~50 MB, fast verify)
python scripts/download_goes.py --start 2023-01-01 --end 2023-06-30 --verify
```

Files land in `data/raw/goes/`. Each file is ~1.5 MB.

---

### Step 5 — Download Aditya-L1 data (manual)

1. Register at `pradan1.issdc.gov.in` (free ISRO institutional account)
2. Navigate to **Aditya-L1 → SoLEXS → Level 1** → select your date range
3. Download ZIP archives: `AL1_SLX_L1_YYYYMMDD_vX.Y.zip`
4. Place in: `data/raw/aditya_l1/solexs/level1/YYYY/MM/<any_subdir>/`
5. Repeat for HEL1OS: `data/raw/aditya_l1/hel1os/level1/YYYY/MM/DD/<any_subdir>/`

> **No HEL1OS?** The project works without it — HEL1OS-dependent features (`log_hxr_total`, `hardness_ratio`, `qpp_power`) default to 0.0. You only need SoLEXS data to proceed.

---

### Step 6 — Process Aditya-L1 raw data

Extracts ZIPs, preprocesses, detects flares, and builds the master catalog.

```bash
# Process all available days (idempotent — skips already-processed days)
python scripts/process_all.py

# Force full rebuild (deletes existing catalog)
python scripts/process_all.py --rebuild
```

Output:
- `data/processed/YYYYMMDD_solexs.parquet`
- `data/processed/YYYYMMDD_helios.parquet`
- `data/processed/master_catalog.parquet`

---

### Step 7 — Pretrain on GOES data

```bash
python model/train.py --phase pretrain

# Override config without editing the file
python model/train.py --phase pretrain --batch-size 512 --max-epochs 50
```

Training prints per-epoch AUC on validation set. Best checkpoint saved to `model/checkpoints/best.pt`.

- CPU (1461 GOES days): ~2–4 hours
- GPU: ~20–40 minutes

> If you have no GOES data, skip to fine-tuning — the model initializes with random weights and still converges, with higher variance.

---

### Step 8 — Fine-tune on Aditya-L1 data

```bash
# Fine-tune using the pretrained checkpoint
python model/train.py --phase finetune

# Fine-tune from a specific checkpoint
python model/train.py --phase finetune --ckpt model/checkpoints/pretrain_best.pt
```

Fine-tuning freezes the CNN layers and trains LSTM + heads at lr=1e-4. Expect 10–30 minutes per run.

---

### Step 9 — Evaluate the model

```bash
# Default: last 20% of processed days as held-out test set
python scripts/evaluate_model.py

# Custom split fraction
python scripts/evaluate_model.py --holdout-frac 0.3

# Specific checkpoint
python scripts/evaluate_model.py --ckpt model/checkpoints/finetune_best.pt
```

Output:
```
=== Threshold Sweep (TPR / FAR / TSS / HSS / F1 / AUC) ===
  threshold   TPR     FAR     TSS     HSS     F1      AUC
  ...

=== Lead-Time Analysis (threshold=0.5) ===
  n_true_positives: ...
  p10_min: ...
  p50_min: ...
  p90_min: 27.0
```

---

### Step 10 — Launch the live dashboard

```bash
python dashboard/app.py
# Open: http://localhost:8050
```
```bash
C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe dashboard/app.py
# Open: http://localhost:8050
```

The dashboard automatically loads the latest checkpoint and processed data. Runs batched inference over all 1440 daily windows.

Dashboard tabs:
1. **Live Flux** — SoLEXS light curve with detected flare markers
2. **Probability** — 30-min ahead forecast with alert threshold
3. **Class Distribution** — A/B/C/M/X probability breakdown
4. **Master Catalog** — All detected flares with class, peak flux, source, confirmation

---

### Step 11 — Sanity checks

```bash
# Verify QPP feature at 1-second cadence
python model/features.py
# Expected: OK: qpp_power osc=X.XXX > noise=Y.YYY

# Check catalog
python -c "
from pipeline.catalog import load_master_catalog, catalog_summary
from pathlib import Path
print(catalog_summary(load_master_catalog(Path('data/processed/master_catalog.parquet'))))
"

# Verify a GOES file is readable
python scripts/download_goes.py --start 2023-01-01 --end 2023-01-07 --verify
```

---

### Troubleshooting

| Problem | Fix |
|---|---|
| `UnicodeDecodeError` on Windows | Use `open(path, encoding='utf-8')` not `Path.read_text()` (Windows defaults to cp1252) |
| `pyarrow` not installed | `_compat.py` uses Pickle fallback — works, but install `pyarrow` for production |
| No flares in catalog | Check data covers solar cycle 25 peak (2024–2025). Lower `detection_sigma` for quiet periods. |
| Low AUC after fine-tuning | Ensure pretrain completed. Need ≥30 days of SoLEXS data; reduce `holdout_frac` to 0.1. |
| Dashboard shows no model | Checkpoint missing — run `train.py` first. Check `model_checkpoint` path in config.yaml. |

---

## 10. Glossary

| Term | Full Name / Meaning |
|---|---|
| **Aditya-L1** | ISRO's solar observatory spacecraft at the Sun-Earth L1 Lagrange point. Launched Sep 2023, reached L1 Jan 2024. |
| **AR** | Active Region. A magnetically complex sunspot group from which flares originate. |
| **AUC** | Area Under the ROC Curve. Threshold-independent model performance; 0.5 = random, 1.0 = perfect. |
| **bg_long** | Rolling 5th-percentile background of the 1–8 Å channel. Used to compute `excess_norm`. |
| **CNN-LSTM** | Convolutional Neural Network + Long Short-Term Memory hybrid. Conv layers extract local temporal patterns; LSTM captures long-range dependencies. |
| **dflux_1min** | 1-minute finite difference of `flux_long`. Key impulsive-rise feature (#3). |
| **dflux_5min** | 5-minute flux derivative normalized to per-minute rate. Slow-precursor feature (#4). |
| **FAR** | False Alarm Rate = FP / (FP + TN). Fraction of quiet periods where an alert fires. |
| **GOES-16** | NOAA Geostationary Operational Environmental Satellite. GOES-16 XRS is the global reference for solar X-ray monitoring. |
| **HEL1OS** | High Energy L1 Orbiting X-ray Spectrometer. Hard X-ray (10–150 keV) instrument on Aditya-L1. |
| **HSS** | Heidke Skill Score. Skill relative to a random forecast matching the class distribution. |
| **HXR** | Hard X-ray emission. Non-thermal signature from accelerated electrons during flare impulsive phase. |
| **ISSDC** | Indian Space Science Data Centre. ISRO's data archive; hosts Aditya-L1 data via PRADAN. |
| **L1** | Sun-Earth Lagrange Point 1. Gravitational balance point 1.5 million km from Earth, toward the Sun. |
| **LSTM** | Long Short-Term Memory network. Recurrent architecture with gated memory cells for sequential data. |
| **MAD** | Median Absolute Deviation. Robust scale estimator used in background estimation. |
| **NCEI** | National Centers for Environmental Information (NOAA). Archives GOES XRS data. |
| **NOAA** | National Oceanic and Atmospheric Administration (USA). Operates GOES satellites. |
| **NRT** | Near-Real-Time. The GOES NRT archive covers recent months before science-grade processing completes. |
| **P90** | 90th percentile of lead times. "90% of true-positive alerts arrive this early or earlier." Current: 27 min. |
| **PRADAN** | ISRO's data portal at `pradan1.issdc.gov.in`. Registration required for Aditya-L1 Level-1 data. |
| **QPP** | Quasi-Periodic Pulsations. Repetitive intensity variations in flare emission (periods 2–30 min). |
| **qpp_power** | Feature #7. Band-power fraction of the 120–1800 s period band in a rolling FFT of native HEL1OS data. |
| **ROC** | Receiver Operating Characteristic curve. Plots TPR vs FAR at all thresholds; AUC summarizes it. |
| **SDD** | Silicon Drift Detector. The detector technology used in SoLEXS for X-ray photon counting. |
| **SNR** | Signal-to-Noise Ratio. In detection context: `(flux − background) / background`. |
| **SoLEXS** | Solar Low Energy X-ray Spectrometer. Soft X-ray instrument on Aditya-L1. Primary detection channel. |
| **SXR** | Soft X-ray emission. Thermal signature from heated flare plasma; peaks minutes after HXR. |
| **TPR** | True Positive Rate (recall/sensitivity) = TP / (TP + FN). |
| **TSS** | True Skill Statistic = TPR − FAR. The community standard metric for solar flare forecasting. |
| **URSC** | U R Rao Satellite Centre (ISRO, Bangalore). Operates Aditya-L1 and publishes the official SoLEXS flare catalog. |
| **XRS** | X-Ray Sensor. Instrument on GOES measuring solar flux in 0.5–4 Å and 1–8 Å bands. |
| **Z-score** | Normalization: (x − mean) / std. Applied per feature using training-set statistics saved in the checkpoint. |

---

*Fine-tune #3 results: AUC = 0.818 · TSS = 0.361 · HSS = 0.310 · P90 = 27 min*
