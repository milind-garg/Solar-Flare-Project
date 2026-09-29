# Solar Flare Nowcasting & Forecasting — Aditya-L1

> **ISRO Problem Statement 15 — Aditya-L1**  
> Real-time solar flare detection (*nowcasting*) and ~30-minute advance prediction (*forecasting*) from India's Aditya-L1 X-ray instruments (SoLEXS & HEL1OS), enhanced with transfer learning from NOAA GOES-16.

---

## 🌟 Overview

This repository provides an end-to-end space-weather pipeline and operations dashboard for detecting and predicting solar flares from Aditya-L1:

| System | Purpose | Method | Machine Learning? |
| :--- | :--- | :--- | :--- |
| **Nowcasting** | Detects flare onset, peak, end, and class (A/B/C/M/X) in real-time light curves | Statistical thresholding (`excess > N·MAD_noise` + rise-rate guard) | ❌ Pure Signal Processing |
| **Forecasting** | Predicts probability of flare peaking in the next 30 min from 60 min history | 1D-CNN + 2-layer LSTM deep neural net | ✅ Supervised ML (PyTorch) |
| **Dashboard** | Operations UI with live curves, alerts, catalogs, and probability traces | Dash by Plotly | — |

---

## 🛰️ Multi-Instrument Data Strategy

```
  TEACHER (NOAA GOES-16 XRS)                TARGET (ISRO Aditya-L1)
┌───────────────────────────────┐          ┌───────────────────────────────────┐
│ • 20+ years of solar flares   │ pretrain │ • SoLEXS: Soft X-ray (SDD1/SDD2)  │
│ • Well-calibrated flux (W/m²) │─────────►│ • HEL1OS: Hard X-ray (10–150 keV) │
│ • Large-scale pre-training    │ finetune │ • Transfer-learned with small data│
└───────────────────────────────┘          └───────────────────────────────────┘
```

1. **GOES-16 XRS**: Serves as the "teacher" dataset for large-scale pre-training of the CNN-LSTM.
2. **Aditya-L1 SoLEXS**: Target soft X-ray spectrometer for fine-tuning and primary real-time nowcasting.
3. **Aditya-L1 HEL1OS**: Target hard X-ray spectrometer capturing non-thermal precursors (Neupert effect) to boost forecast lead times.

---

## 🏗️ Architecture

```
Raw FITS / NetCDF
  │  pipeline/readers.py (read_solexs / read_helios / read_goes_nc)
  ▼
Uniform 1-min Time Series (UTC DatetimeIndex, normalized flux)
  │  pipeline/preprocessor.py (rolling 60th percentile bg + 1.4826·MAD noise)
  ├──────────────────────────────────────────────┐
  ▼ [Track A: Nowcasting]                        ▼ [Track B: Forecasting]
pipeline/detector.py                           model/features.py (8-feature windows)
  │ (excess SNR + rise rate)                     │
  ▼                                              ▼
pipeline/catalog.py                            model/train.py (CNN-LSTM)
  │                                              │
  ▼                                              ▼
master_catalog.parquet (Flare event list) ───► models/cnn_lstm_best.pt
  │                                              │
  └──────────────────────┬───────────────────────┘
                         ▼
             dashboard/app.py (Dash UI)
```

### 8 Forecasting Features
1. `log_flux_short`: 0.5–4 Å soft X-ray flux
2. `log_flux_long`: 1–8 Å primary soft X-ray flux
3. `log_hxr_total`: 10–150 keV hard X-ray flux (HEL1OS)
4. `dflux_1min`: 1-minute flux derivative (rise rate)
5. `dflux_5min`: 5-minute flux derivative (precursor slope)
6. `hardness_ratio`: Hard / soft ratio (non-thermal precursor trigger)
7. `excess_norm`: `(flux - background) / background`
8. `qpp_power`: Quasi-periodic pulsation power (rolling FFT peak, HEL1OS hard X-ray precursor)

---

## ⚡ Quickstart

### 1. Environment Setup
```bash
# Clone the repository
git clone <YOUR_REPO_URL>
cd "Solar Flare Project"

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Detector (Sanity Check)
```bash
python pipeline/detector.py
```
*Expected: `OK: detected C... flare` on a synthetic test curve.*

### 3. Build SoLEXS Flare Catalog (Nowcasting)
```bash
python scripts/process_all.py --rebuild
```
Processes raw FITS files from `data/raw/aditya_l1/solexs/` and generates `data/processed/master_catalog.parquet`.

### 4. Train the Forecasting Model
```bash
# Phase 1: Pre-train on GOES-16 data
python model/train.py --phase pretrain --goes-dir data/raw/goes --epochs 30

# Phase 2: Fine-tune on Aditya-L1 SoLEXS
python model/train.py --phase finetune --sxr-dir data/processed --epochs 10
```

### 5. Launch Operations Dashboard
```bash
C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe dashboard/app.py
```
Visit `http://127.0.0.1:8050` for the landing page and `http://127.0.0.1:8050/dashboard/` for the interactive operations dashboard (requires Python 3.12 with torch).

---

## 📁 Repository Structure

```
├── dashboard/               # Dash by Plotly web application & visualization
│   └── app.py
├── pipeline/                # Core data ingestion & nowcasting signal processing
│   ├── readers.py           # NetCDF4 and FITS readers for GOES, SoLEXS, HEL1OS
│   ├── preprocessor.py      # Resampling, background estimation & MAD noise
│   ├── detector.py          # Real-time threshold flare detector & classifier
│   └── catalog.py           # Event catalog builder and cross-instrument matching
├── model/                   # ML forecasting engine
│   ├── cnn_lstm.py          # PyTorch 1D-CNN + LSTM neural network
│   ├── features.py          # 8-feature engineering & sliding window builder
│   ├── train.py             # Pre-training & fine-tuning pipelines
│   └── evaluate.py          # TPR, FAR, and lead-time evaluation metrics
├── models/                  # Saved PyTorch model checkpoints
├── scripts/                 # Batch processing & data download utilities
│   ├── download_goes.py
│   └── process_all.py
├── notebooks/               # Exploratory data analysis & experiments
├── config.yaml              # Global pipeline & model configurations
├── requirements.txt         # Project dependencies
├── GUIDE.md                 # Complete walkthrough guide
├── WORKFLOW.md              # Technical data flow & tensor shape specs
└── RESEARCH.md              # Physics foundations & mission literature
```

---

## 📜 Documentation

- [GUIDE.md](GUIDE.md) — Comprehensive guide on algorithms, dataset roles, and tuning parameters.
- [WORKFLOW.md](WORKFLOW.md) — End-to-end technical breakdown of data shapes, units, and training mechanics.
- [RESEARCH.md](RESEARCH.md) — Solar physics background (Neupert effect, X-ray classifications, instrument details).
- [RUN_CHECKLIST.md](RUN_CHECKLIST.md) — Step-by-step execution and evaluation checklist.
