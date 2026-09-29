Project Audit Report

2026-08-27 · v1.0

## Solar Flare Nowcasting & Forecasting

Complete audit of architecture, code, data, and execution status for the ISRO PS-15 competition submission built on Aditya-L1 Level-1 instrument data.

PROJECT COMPLETION

## 01 Project Overview

ISRO Problem Statement 15 (PS-15)

COMPETITION

60 / 100

Aditya-L1 · India's first dedicated solar observatory at L1

MISSION

SoLEXS (0.5–8 Å SXR) + HEL1OS (10–200 keV HXR)

INSTRUMENTS

Real-time flare nowcasting + 30-min probabilistic forecasting from Level-1 FITS

OBJECTIVE

C:\Projects\Claude Code\Solar Flare Project

PROJECT ROOT

Pre-train CNN-LSTM on GOES-16 (4 yr public data) → fine-tune on Aditya-L1

STRATEGY

NOAA NCEI GOES-16 XRS Level-2 NetCDF4 · ISSDC PRADAN (Aditya-L1)

REFERENCE DATA

FLARE CLASSES (GOES-equivalent):

≥ 10⁻⁴ W/m²

≥ 10⁻⁵

## 02 Code & File Inventory

≥ 10⁻⁶

B

≥ 10⁻⁷

< 10⁻⁷

```
Solar Flare Project/ ├── config.yaml — all thresholds, paths, model hypers ├── requirements.txt — 12
Python dependencies │ ├── pipeline/ │ ├── readers.py — FITS (SoLEXS, HEL1OS) + GOES NC/txt/csv │ ├──
preprocessor.py — resample · background · noise · Parquet │ ├── detector.py — algorithmic nowcasting,
FlareEvent model │ └── catalog.py — multi-instrument merge → master Parquet │ ├── model/ │ ├──
cnn_lstm.py — SolarFlareForecaster architecture │ ├── features.py — 7-feature engineering + sliding
windows │ ├── train.py — 2-phase CLI trainer (pretrain / finetune) │ └── evaluate.py —
TPR/FAR/AUC/lead-time metrics │ ├── models/ │ └── cnn_lstm_best.pt — checkpoint (exists; not yet
trained) │ ├── dashboard/ │ └── app.py — Streamlit · Historical + Live modes │ ├── scripts/ │ └──
download_goes.py — NCEI NetCDF4 downloader + verifier │ ├── notebooks/ │ ├── 01_eda.ipynb — exploratory
data analysis │ ├── 02_nowcasting.ipynb — algorithm walkthrough │ └── 03_forecasting.ipynb — model
demo │ └── data/raw/goes/ └── sci_xrsf-l2-avg1m_g16_d*.nc × 1,462 files downloaded
```


## 03 System Architecture

## 04 Model & Feature Specification

CNN-LSTM ARCHITECTURE

TRAINING CONFIG

(batch, 60, 7)

INPUT SHAPE

60 min (1-min cadence)

WINDOW

7→32→64, kernel=3, pad=1

CONV LAYERS

30 min

LEAD TIME

MaxPool1d(2) → seq 60→30

POOL

P ≥ 0.5 (configurable)

ALERT THRESHOLD

hidden=128, layers=2,

LSTM

5× positives (class balance)

NEG. RATIO CAP

drop=0.3

Linear(128→1) +

HEAD

1e-3 · epochs 30

PRETRAIN LR

BCEWithLogits

1e-4 · epochs 10 · conv

FINETUNE LR

pos_weight = neg/pos

CLASS WEIGHT

frozen

(auto)

10 % (random seed 0)

VAL SPLIT

Adam +

OPTIMIZER

256

ReduceLROnPlateau(p=3)

BATCH SIZE

norm = 1.0

GRAD CLIP

## INPUT FEATURES (7 PER TIMESTEP)

## 05 Nowcasting Algorithm & Parameters

| PARAMETER | VALUE | DESCRIPTION |
| --- | --- | --- |
| sigma_threshold | 3.0 σ | Excess must exceed N × noise before trigger fires |
| min_consecutive_samples | 5 samples | Must stay above threshold for K samples (rejects spikes) |


- 7-feature engineering + sliding-window builder

- CNN-LSTM model architecture (SolarFlareForecaster)

- 2-phase training CLI (pretrain + finetune)

- Evaluation: TPR, FAR, AUC, lead-time analysis

- Streamlit dashboard: Historical + Live modes

- GOES-16 downloader script (1,462 files retrieved)

- 3 analysis notebooks scaffolded

- config.yaml (all params externalised)

- Self-labeling (detector labels own training data — no external catalog needed)

## 09 Immediate Next Steps

All code is in place. The remaining work is execution — running the training pipeline and sourcing Aditya-L1 data. No code changes are required to proceed with steps 1–4.

- ! Fine-tune on Aditya-L1 processed data (~10 epochs)

- ! Run full evaluation report on held-out test set

- ! Validate nowcasting on real Aditya-L1 event

- ! Complete notebook analyses (EDA, nowcasting walkthrough, model demo)

- ! Live dashboard smoke-test against processed Aditya-L1 day

- ! Calibrate SoLEXS count-rate → GOES-equivalent W/m² (CALIB_CH1/CH2 header values)

## 1 Pre-train the CNN-LSTM on GOES data

python model/train.py --phase pretrain --goes-dir data/raw/goes --epochs 30

Reads all 1,462 NetCDF4 files, self-labels flares via detector, builds sliding windows, trains. Saves best checkpoint to models/cnn_lstm_best.pt. Expect 1–4 hours on CPU depending on hardware.

2

→ ISSDC PRADAN portal (manual browser download required)

Place SoLEXS FITS in data/raw/solexs/ and HEL1OS FITS in data/raw/helios/. Run FITS.info() in notebooks/01_eda.ipynb first to confirm actual HDU column names match the reader expectations.

## Download Aditya-L1 Level-1 FITS files

## Process FITS → Parquet

from pipeline.preprocessor import process_and_save_solexs, process_and_save_helios

Call both functions for each FITS file. Output lands in data/processed/YYYYMMDD_solexs.parquet and YYYYMMDD_helios.parquet. The dashboard and fine-tune script expect exactly these names.

## Fine-tune on Aditya-L1

python model/train.py --phase finetune --sxr-dir data/processed --epochs 10

Loads pretrained checkpoint, freezes Conv layers, re-trains LSTM + head on Aditya-L1 windows at lr=1e-4. Requires both SoLEXS Parquets and a master catalog (generated during nowcasting runs).

## Launch the dashboard

## streamlit run dashboard/app.py

Select "GOES XRS" instrument and any 2020+ date to immediately test against downloaded GOES data. Switch to "SoLEXS" once Aditya-L1 Parquets are ready. Live mode auto-refreshes every 60 s.


Generated 2026-08-27 · ISRO PS-15 · Aditya-L1 Solar Flare Nowcasting & Forecasting

C:\Projects\Claude Code\Solar Flare Project
