# Milestone 4 — Working Prototype & System Demonstration
### Cornerstone Project | 5th Semester
**Aditya-L1 Solar Flare Nowcasting & Forecasting System**
*Aditya-L1 Solar Flare Observatory*

---

> **GitHub Repository:** `https://github.com/milind-garg/Solar-Flare-Project`
>
> **Dashboard Demo:** Live at `http://127.0.0.1:8050` · Dashboard at `http://127.0.0.1:8050/dashboard/`

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Overview](#2-system-overview)
3. [Architecture](#3-architecture)
4. [Data & Instruments](#4-data--instruments)
5. [Feature Engineering](#5-feature-engineering)
6. [ML Model](#6-ml-model--cnn-lstm)
7. [Training Pipeline](#7-training-pipeline)
8. [Evaluation Results](#8-evaluation-results)
9. [Dashboard Demonstration](#9-dashboard-demonstration)
10. [Repository Structure](#10-repository-structure)
11. [How to Run](#11-how-to-run)
12. [Challenges & Limitations](#12-challenges--limitations)

---

## 1. Problem Statement

**ISRO PS-15 (verbatim):**
> *"Forecasting and/or Nowcasting of Solar Flares using combined Soft and Hard X-ray data from Aditya-L1."*

Solar flares are sudden releases of magnetic energy from the Sun's atmosphere capable of causing geomagnetic storms, power-grid failures, GPS disruptions, satellite damage, and aviation radiation hazards. A **30-minute advance warning** allows satellites to enter safe-mode and aircraft to reroute over polar regions.

**ISRO's explicit requirements:**

| Requirement | Status |
|-------------|--------|
| Real-time flare detection / nowcasting | ✅ Done |
| Unified SoLEXS + HEL1OS flare catalog | ✅ Done |
| Forecast P(flare in next N minutes) | ✅ Done — AUC = 0.818 |
| Multi-class intensity probabilities (A/B/C/M/X) | ✅ Done |
| High TPR, low FPR | ✅ TSS = 0.361, HSS = 0.310 |
| 15–30 min lead time | 🟡 P90 = 27 min (3 min short of target) |
| Pre-flare sudden increases | ✅ `dflux_1min`, `dflux_5min` features |
| Quasi-Periodic Pulsations (QPPs) in HXR | ✅ `qpp_power` — rolling FFT on HEL1OS |

---

## 2. System Overview

The system has **three tightly integrated components:**

```
┌──────────────────────────────────────────────────────────┐
│              ADITYA-L1 SOLAR FLARE SYSTEM                │
│                                                          │
│  ┌───────────────┐  ┌──────────────────┐  ┌──────────┐  │
│  │  NOWCASTING   │  │   FORECASTING    │  │DASHBOARD │  │
│  │  (Track A)    │  │   (Track B)      │  │          │  │
│  │               │  │                  │  │ Curves   │  │
│  │ σ-threshold   │  │  CNN-LSTM model  │  │ Alerts   │  │
│  │ detector on   │  │  P(flare next    │  │ Forecast │  │
│  │ SoLEXS/HEL1OS │  │  30 min)         │  │ Catalog  │  │
│  │ light curves  │  │  + A/B/C/M/X     │  │ CSV DL   │  │
│  └───────┬───────┘  └────────┬─────────┘  └────┬─────┘  │
│          │                   │                  │         │
│          └───────────────────┴──────────────────┘         │
│                    master_catalog.parquet                  │
└──────────────────────────────────────────────────────────┘
```

| Component | Method | ML? |
|-----------|--------|-----|
| **Nowcasting** | Statistical thresholding (σ-excess + rise-rate guard) | ❌ Pure signal processing |
| **Forecasting** | 1D-CNN + 2-layer LSTM neural network | ✅ Supervised ML (PyTorch) |
| **Dashboard** | Dash by Plotly, Flask backend | — |

---

## 3. Architecture

### End-to-End Pipeline

```
Raw FITS / NetCDF  (PRADAN + NOAA NCEI)
  │
  │  pipeline/readers.py
  │  read_solexs() / read_helios() / read_goes_nc()
  ▼
Uniform 1-min Time Series (DatetimeIndex UTC)
  │
  │  pipeline/preprocessor.py
  │  rolling 60th-percentile bg + 1.4826·MAD noise
  │
  ├──────────────────────────────────────────────┐
  ▼  [Track A: Nowcasting]                       ▼  [Track B: Forecasting]
pipeline/detector.py                          model/features.py
  │  excess > N·σ + rise-rate guard             │  8 engineered features
  ▼                                             ▼
pipeline/catalog.py                          model/train.py
  │  Neupert-aware SXR + HXR merge             │  CNN-LSTM two-phase training
  ▼                                             ▼
master_catalog.parquet ──────────────► models/cnn_lstm_best.pt
  │                                             │
  └──────────────────────┬──────────────────────┘
                          ▼
                  dashboard/app.py (Dash, port 8050)
```

### Transfer Learning Strategy

```
Phase 1 — Pretrain on GOES-16 (2020–2023)
  • 1,922 NetCDF files · ~1.8 M sliding windows
  • LR = 1e-3 · 30 epochs · all layers trainable
  • val_loss = 0.1145
            ↓  save checkpoint
Phase 2 — Fine-tune on Aditya-L1 SoLEXS
  • 824 processed Parquet files · ~660 K windows
  • CNN layers frozen → LSTM + heads retrain at LR = 5e-6
  • Combined loss: BCE (binary) + CrossEntropy (A/B/C/M/X)
            ↓
Final model: AUC = 0.818 · TSS = 0.361 · HSS = 0.310
```

---

## 4. Data & Instruments

### Instruments Used

| Instrument | Spacecraft | Band | Cadence | Units | Role |
|------------|------------|------|---------|-------|------|
| **SoLEXS** | Aditya-L1 (ISRO) | Soft X-ray · 0.5–10 keV | ~1 second | counts/s | Primary detection & training |
| **HEL1OS** | Aditya-L1 (ISRO) | Hard X-ray · 10–150 keV | ~1 second | counts/s | Non-thermal precursors, QPPs |
| **GOES-16 XRS** | GOES-16 (NOAA) | 0.5–4 Å and 1–8 Å | 1 minute | W/m² | Pre-training teacher dataset |

> **Why L1?** The Sun-Earth L1 Lagrange point is 1.5 million km sunward. No Earth occultations — uninterrupted 24/7 solar monitoring.

### Dataset Scale

| Dataset | Coverage | Volume |
|---------|----------|--------|
| GOES-16 XRS NetCDF4 | 2020–2023 (~1,461 days) | 1,922 files · ~2 GB |
| SoLEXS L1 (processed) | Feb 2024–present | 824 Parquet files |
| HEL1OS L1 (processed) | Feb 2024–present (patchy) | ~1,008 Parquet files |
| **Master Flare Catalog** | All processed days | **3,435 events** |

### Master Catalog Breakdown

| Class | Count | Description |
|-------|-------|-------------|
| X | 45 | Extreme — R3+ strong radio blackout |
| M | 589 | Major — R1/R2 moderate blackout |
| C | 2,427 | Common — noticeable coronal activity |
| B | 257 | Minor — baseline flare activity |
| HXR-only | 117 | Detected in HEL1OS only |
| Multi-instrument confirmed | 171 | Matched SoLEXS + HEL1OS (Neupert-verified) |

### SoLEXS Calibration

SoLEXS count rates are converted to GOES-equivalent W/m² using a scalar fitted from **435 matched flare peak pairs:**

```
solexs_counts_to_wm2 = 1.806 × 10⁻⁸  [W/m² per count/s]
```

This only affects flare class labelling — detection sensitivity is SNR-based and instrument-agnostic.

---

## 5. Feature Engineering

Every 1-minute timestep is encoded as **8 engineered features.** A 60-minute sliding window forms one model input: shape `(batch, 60, 8)`.

| # | Feature | Formula | Physics Meaning |
|---|---------|---------|-----------------|
| 0 | `log_flux_short` | log₁₀(SXR 0.5–4 Å) floor 1e-9 | Hot coronal plasma (~10 MK) |
| 1 | `log_flux_long` | log₁₀(SXR 1–8 Å) floor 1e-9 | Primary GOES-equivalent channel |
| 2 | `log_hxr_total` | log₁₀(HEL1OS broadband) floor 0.1 | Non-thermal particle acceleration |
| 3 | `dflux_1min` | Δflux_long / 1 min | Impulsive rise — strongest precursor |
| 4 | `dflux_5min` | Δflux_long / 5 min (per-min rate) | Slower pre-flare build-up |
| 5 | `hardness_ratio` | hxr_total / flux_long | Spectral hardening before SXR peak |
| 6 | `excess_norm` | (flux_long − bg_long) / bg_long | Normalized SNR above background |
| 7 | `qpp_power` | Rolling 30-min FFT in [120s–1800s] band | **ISRO PS-15 QPP requirement** |

**QPP Feature Note:**
`qpp_power` is computed on the **native 1-second HEL1OS cadence** before resampling to 1-minute. This preserves the full QPP period band — at 1-min cadence, periods < 2 min would be aliased. Each window is linearly detrended to remove flare-rise ramp leakage before FFT.

---

## 6. ML Model — CNN-LSTM

**File:** `model/cnn_lstm.py` | **Class:** `SolarFlareForecaster` | **Parameters:** ~180K

### Architecture

```
Input: (B, 60, 8)   batch × 60-min window × 8 features
       │
       │ permute → (B, 8, 60)
       ▼
Conv1d(8 → 32, kernel=3, pad=1) + ReLU + MaxPool(2) + Dropout(0.3)
Conv1d(32 → 64, kernel=3, pad=1) + ReLU + MaxPool(2) + Dropout(0.3)
       → (B, 64, 15)
       │ permute → (B, 15, 64)
       ▼
LSTM(input=64, hidden=128, num_layers=2, dropout=0.3)
       → last hidden state → (B, 128)
       │
       ├─────────────────────────────────────┐
       ▼                                     ▼
Linear(128 → 1)                     Linear(128 → 5)
BCEWithLogitsLoss                   CrossEntropyLoss(ignore_index=-1)
P(flare in next 30 min)             P(A) · P(B) · P(C) · P(M) · P(X)
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **1D-CNN** before LSTM | Extracts local patterns (rapid rise, impulsive spikes) before temporal modeling |
| **Transfer learning** from GOES | Overcomes Aditya-L1 data scarcity — bootstraps from 20+ years of GOES labels |
| **`pos_weight = 6.1`** in BCE | Compensates for class imbalance (flares are rare events) |
| **`strict=False`** in checkpoint load | Binary pretrain checkpoints can initialize multi-class models |
| **Norm stats inside checkpoint** | `norm_mean` / `norm_std` saved at train time → inference always consistent |
| **Batch inference** in dashboard | All 1440 windows of a day batched in one forward pass — 1000× faster than per-minute loop |

---

## 7. Training Pipeline

### Phase 1 — Pretrain (GOES-16)

```bash
python model/train.py --phase pretrain --goes-dir data/raw/goes --epochs 30
```

| Hyperparameter | Value |
|---------------|-------|
| Training data | 1,922 GOES-16 NC files (2020–2023) |
| Sliding window | 60-min input · 30-min lead time |
| Negative:Positive ratio cap | 5:1 |
| Optimizer | Adam, LR = 1e-3 |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Gradient clipping | 1.0 |
| Batch size | 256 |
| Best val_loss | **0.1145** |

### Phase 2 — Fine-tune (Aditya-L1)

```bash
# Run 1: Frozen CNN
python model/train.py --phase finetune --epochs 10 --lr 1e-4

# Run 2: Continued frozen, lower LR
python model/train.py --phase finetune --epochs 25 --lr 3e-5

# Run 3: Full unfreeze (best checkpoint)
python model/train.py --phase finetune --epochs 15 --lr 5e-6 --unfreeze-conv
```

| Run | Epochs | LR | Conv Frozen | val_loss | AUC | TSS | HSS |
|-----|--------|----|-------------|----------|-----|-----|-----|
| Fine-tune #1 | 10 | 1e-4 | ✓ | 1.6162 | 0.794 | 0.325 | — |
| Fine-tune #2 | 25 | 3e-5 | ✓ | 1.5844 | 0.813 | 0.349 | — |
| **Fine-tune #3** | **15** | **5e-6** | **✗** | **1.5759** | **0.818** | **0.361** | **0.310** |

---

## 8. Evaluation Results

### Full Threshold Sweep

```
threshold   TPR     FAR     TSS     HSS     F1      AUC
  0.30     0.877   0.432   0.445   0.161   0.295   0.818
  0.50     0.457   0.096   0.361   0.310   0.387   0.818  ← operational
  0.70     0.222   0.017   0.205   0.283   0.321   0.818
```

### Performance Summary

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **AUC-ROC** | **0.818** | NOAA ops: 0.70–0.85 ✅ |
| **TSS @ 0.5** | **0.361** | > 0 = has skill ✅ |
| **HSS @ 0.5** | **0.310** | > 0 = beats climatology ✅ |
| **TPR @ 0.3** | **87.7%** | High recall mode |
| **FAR @ 0.5** | **9.6%** | Operational false alarm rate |
| **Lead Time P90** | **27 min** | ISRO target 30 min 🟡 |
| **Flare Classes** | A/B/C/M/X | 5-class head ✅ |
| Catalog size | 3,435 events | — |

### Metric Definitions

- **AUC** — Area Under ROC Curve. Threshold-independent; 0.5 = random, 1.0 = perfect.
- **TSS** (True Skill Statistic) = TPR − FPR. Standard space-weather metric. 0 = no skill.
- **HSS** (Heidke Skill Score) — skill relative to a random/climatology baseline.
- **P90 Lead Time** — 90th-percentile of (first alert → flare peak). Worst 10% of predictions still give 27 min warning.

---

## 9. Dashboard Demonstration

> Launch: `python dashboard/app.py` → visit `http://127.0.0.1:8050/dashboard/`

### Landing Page

<!-- INSERT IMAGE: Landing page screenshot (index.html) -->

### Tab 1 — Light Curves & Telemetry

<!-- INSERT IMAGE: Light Curves tab — SXR flux light curve with flare markers, background line, threshold lines -->

**Features shown:**
- SXR 1–8 Å light curve with rolling background estimate
- GOES flare class threshold lines (B / C / M / X)
- Shaded flare duration intervals + peak markers with Neupert lag annotation
- HEL1OS hard X-ray impulsive count rate (when data available)
- Spectral hardness ratio (HXR/SXR) precursor index
- Detected event table: start / peak / end, class, SNR, rise rate

### Tab 2 — Predictive ML Forecast

<!-- INSERT IMAGE: Forecast tab — P(flare) probability trace, alert threshold, A/B/C/M/X class bars, KPI tiles -->

**Features shown:**
- `P(flare in next 30 min)` probability trace from CNN-LSTM
- Adjustable alert threshold slider (default = 0.50)
- Red fill above threshold for instant alert visibility
- A/B/C/M/X intensity class probability bars
- KPI tiles: current P(flare), AUC, TSS, Lead P90, most-likely class
- Active alert banner when P > threshold

### Tab 3 — Master Event Catalog

<!-- INSERT IMAGE: Catalog tab — sortable/filterable table with 3,435 events, KPI tiles, Download CSV button -->

**Features shown:**
- Full 3,435-event catalog: sortable, filterable, paginated (25/page)
- Columns: Peak Time, Class, Start/End, Duration, Rise, Decay, Neupert Lag, HXR Bursts, Confirmed
- Color-coded rows by class (X = red, M = amber, C = yellow, B = green)
- KPI tiles: total events, per-class counts, multi-instrument count, median Neupert lag
- Download as CSV button

---

## 10. Repository Structure

```
Solar Flare Project/
│
├── pipeline/
│   ├── readers.py          # FITS/NetCDF readers (SoLEXS, HEL1OS, GOES)
│   ├── preprocessor.py     # Resampling, background estimation, MAD noise
│   ├── detector.py         # σ-threshold flare detector + GOES classifier
│   ├── catalog.py          # Neupert-aware SXR+HXR multi-instrument merger
│   └── _compat.py          # Parquet fallback shim (pickle if pyarrow absent)
│
├── model/
│   ├── cnn_lstm.py         # SolarFlareForecaster (CNN-LSTM, PyTorch)
│   ├── features.py         # 8-feature engineering + qpp_power rolling FFT
│   ├── train.py            # Two-phase pretrain + fine-tune training pipeline
│   └── evaluate.py         # TPR, FAR, TSS, HSS, AUC, lead-time analysis
│
├── models/
│   └── cnn_lstm_best.pt    # Best checkpoint (AUC=0.818, TSS=0.361, HSS=0.310)
│
├── dashboard/
│   ├── app.py              # Dash by Plotly operations dashboard (1,500+ lines)
│   └── assets/theme.css    # Aerospace slate dark design system
│
├── scripts/
│   ├── download_goes.py    # NOAA NCEI GOES-16 batch downloader
│   ├── process_all.py      # Full pipeline batch processor → master catalog
│   ├── calibrate_solexs.py # SoLEXS → W/m² calibration constant fitter
│   └── evaluate_model.py   # Held-out evaluation with full metrics report
│
├── tests/
│   └── test_catalog.py     # 6 unit tests (association, Neupert, schema, dedup)
│
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_nowcasting.ipynb # Detector algorithm analysis
│   └── 03_forecasting.ipynb# Model architecture experiments
│
├── assets/                 # Logo, hero images, theme
├── index.html              # Cinematic space observatory landing page
├── config.yaml             # All pipeline & model hyperparameters
├── requirements.txt        # 14 Python dependencies with minimum versions
└── README.md               # Quick-start guide
```

---

## 11. How to Run

### Prerequisites

```bash
# Python 3.12 required (torch compatibility)
pip install -r requirements.txt
```

### Step 1 — Download GOES-16 Data

```bash
python scripts/download_goes.py --start 2020-01-01 --end 2023-12-31
# Downloads ~1,922 NetCDF4 files to data/raw/goes/
```

### Step 2 — Process Aditya-L1 Data

Place PRADAN ZIPs into:
- SoLEXS: `data/raw/aditya_l1/solexs/level1/YYYY/MM/`
- HEL1OS: `data/raw/aditya_l1/hel1os/level1/YYYY/MM/DD/`

```bash
python scripts/process_all.py --rebuild
# Output: data/processed/*_solexs.parquet, *_helios.parquet, master_catalog.parquet
```

### Step 3 — Calibrate SoLEXS (Optional)

```bash
python scripts/calibrate_solexs.py
# Prints recommended solexs_counts_to_wm2 value for config.yaml
```

### Step 4 — Train the Model

```bash
# Phase 1: Pre-train on GOES-16
python model/train.py --phase pretrain --goes-dir data/raw/goes --epochs 30

# Phase 2a: Fine-tune (CNN frozen)
python model/train.py --phase finetune --epochs 10 --lr 1e-4

# Phase 2b: Full unfreeze (final best checkpoint)
python model/train.py --phase finetune --epochs 15 --lr 5e-6 --unfreeze-conv
```

### Step 5 — Evaluate

```bash
python scripts/evaluate_model.py
# Prints: TSS / HSS / AUC threshold sweep + lead-time statistics
```

### Step 6 — Launch Dashboard

```bash
python dashboard/app.py
# Landing page:  http://127.0.0.1:8050
# Dashboard:     http://127.0.0.1:8050/dashboard/
```

### Step 7 — Run Unit Tests

```bash
python -m pytest tests/ -v
# 6 tests: catalog association, Neupert lag, multi-burst, schema, deduplication
```

---

## 12. Challenges & Limitations

### 1. Limited Aditya-L1 Labeled Data
Aditya-L1 launched September 2023 — only ~2 years of science data exists versus GOES-16's 20+ years. Transfer learning from GOES was the central design decision to overcome this constraint.

### 2. Lead Time Gap (27 vs 30 min Target)
The P90 forecast lead of **27 min** falls 3 min short of ISRO's 30-min target. The 60-minute history window is conservative. Possible improvements: use pre-flare SXR brightening patterns more aggressively, or inject Neupert-effect features directly.

### 3. HEL1OS Data Availability
HEL1OS data is patchy — not every day has hard X-ray coverage. On HEL1OS-absent days `qpp_power` and `hardness_ratio` fall back to 0, degrading the non-thermal precursor signal.

### 4. False Alarm Rate at High Recall
At `threshold=0.3` (high-recall mode), FAR = 43%. This is the classical precision-recall tradeoff for rare-event prediction. Space weather operations typically prioritize high recall over low FAR — missing a flare is costlier than a false alarm.

### 5. Scalar SoLEXS Calibration
The `1.806e-8` calibration constant is a linear approximation of a response that is energy-resolved. A full effective area curve would improve class accuracy, particularly for X-class flares where the SDD detectors approach saturation.

---

## Appendix — Configuration Reference

```yaml
# config.yaml — complete reference

nowcasting:
  sigma_threshold: 3.0           # Excess must exceed N × background noise
  min_consecutive_samples: 5     # Sustained trigger for K samples before declaring flare
  rise_rate_threshold: 1.0e-9    # W/m²/s — minimum dF/dt to reject slow drifts
  suppression_window_min: 15     # Merge detections within this window (keep peak)
  background_window_min: 60      # Rolling window for 60th-percentile background
  end_factor: 1.5                # Flare ends when flux ≤ end_factor × background

calibration:
  # Fitted via median-ratio regression from 435 matched GOES/SoLEXS peak-flux pairs
  solexs_counts_to_wm2: 1.806e-08

forecasting:
  input_window_min: 60           # Minutes of history fed to model
  lead_time_min: 30              # Predict flare occurrence within next N minutes
  alert_threshold: 0.5           # P > threshold → issue alert
  model_features: 8              # Input features per timestep
  n_classes: 5                   # A / B / C / M / X intensity head

paths:
  raw_data: data/raw/
  processed_data: data/processed/
  model_checkpoint: models/cnn_lstm_best.pt
  master_catalog: data/processed/master_catalog.parquet
```

---

*Milestone 4 — Working Prototype Submission*  
*Project: Aditya-L1 Solar Flare Observatory | 5th Semester Cornerstone*
