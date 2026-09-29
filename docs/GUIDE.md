# Solar Flare Nowcasting & Forecasting — Complete Guide

**ISRO Problem Statement 15 — Aditya-L1**
Detect solar flares *as they happen* (nowcast) and *predict them ~30 min ahead* (forecast), from India's Aditya-L1 X-ray instruments, using public NOAA/GOES data as a training crutch.

---

## 1. What this project actually is

Two systems in one repo. **It is NOT just "train an ML model."**

| System | What it does | How | ML? |
|--------|--------------|-----|-----|
| **Nowcasting** | Detects a flare happening *right now* in the live light curve, finds its start/peak/end, classifies it (A/B/C/M/X) | Threshold algorithm: `excess > N·noise` sustained + rise-rate guard | ❌ No ML — pure signal processing |
| **Forecasting** | Predicts "will a flare peak in the next 30 min?" from the last 60 min of data | CNN-LSTM neural net, outputs a probability | ✅ Yes |
| **Dashboard** | Dash UI: live light curves, detected flares, forecast probability, master catalog | Dash by Plotly | — |

The ML model is one of three deliverables. The detector (nowcasting) needs **no training at all** — it works today on any light curve. The forecaster is the ML piece. The dashboard ties them together for judges to see.

**Why two approaches?** Nowcasting is instant and explainable (good for "we detected an X2 flare at 14:32"). Forecasting is the hard, impressive part (predicting the future). ISRO PS-15 wants both.

**Dashboard note:** The dashboard runs on **Dash by Plotly** (not Streamlit). Launch with Python 3.12: `C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe dashboard/app.py` → `http://127.0.0.1:8050`.

---

## 2. The 3 datasets — why three, and what each is for

This is the part that confuses everyone. You have three X-ray datasets because **two are your real target and one is a teacher.**

```
   TEACHER (huge, public, labeled)          TARGET (the actual ISRO instruments)
   ┌─────────────────────────┐              ┌──────────────────────────────────┐
   │        GOES XRS          │   pretrain   │   SoLEXS (soft X-ray, 1–8 Å-ish) │
   │  NOAA, US satellite      │  ─────────►  │   HEL1OS (hard X-ray, 10–150 keV)│
   │  20+ yrs, millions of    │  fine-tune   │   Aditya-L1, India, ~1.5 yrs data │
   │  flares, well-calibrated │              │                                    │
   └─────────────────────────┘              └──────────────────────────────────┘
```

### Dataset A — GOES XRS (the teacher) ✅ you have it
- **What:** NOAA's GOES-16 satellite X-ray Sensor. Measures soft X-ray flux in two bands (0.5–4 Å "short", 1–8 Å "long") in **W/m²**. The 1–8 Å band is *the* official definition of flare class worldwide (a C1 flare = 1×10⁻⁶ W/m²).
- **Why you need it:** A CNN-LSTM has ~200k parameters. Aditya-L1 has only ~1.5 years of data — not enough to train a deep net from scratch without overfitting. GOES has **decades** of flares. So you **pre-train** the model on GOES to teach it "what a flare looks like," then **fine-tune** on Aditya. This is transfer learning.
- **Format:** NetCDF4, one file per day: `sci_xrsf-l2-avg1m_g16_d20200101_v2-2-1.nc`
- **On your disk:** `data/raw/goes/` — **1,924 files, 544 MB, 2020-01-01 onward.** ✔️ Done.

### Dataset B — SoLEXS (target instrument #1, SOFT X-ray) ✅ you have it
- **What:** **So**lar **L**ow **E**nergy **X**-ray **S**pectrometer on Aditya-L1. The direct analog of GOES XRS — soft X-rays. This is your **primary** flare-detection channel. Two detectors: SDD1 (soft) and SDD2 (primary broadband).
- **Why:** It's the ISRO instrument you're actually judged on. It measures **counts/s**, not W/m² — so it needs calibration to convert to GOES-equivalent flux (that's the `solexs_counts_to_wm2` knob in `config.yaml`, currently a placeholder you must tune).
- **Format:** PRADAN ZIPs → folders `AL1_SLX_L1_YYYYMMDD_v1.0/` each with `SDD1/` and `SDD2/*.lc.gz` FITS light curves (1 Hz, 86400 rows/day).
- **On your disk:** `data/raw/aditya_l1/solexs/` — **~823 days extracted, 11 GB.** ✔️ Done. **And already processed** → 797 `*_solexs.parquet` files in `data/processed/`.

### Dataset C — HEL1OS (target instrument #2, HARD X-ray) ⏳ you're still sourcing
- **What:** **H**igh **E**nergy **L1** **O**rbiting **S**pectrometer. Measures **hard** X-rays (10–150 keV) — much higher energy than SoLEXS/GOES. Units: counts/s per energy band.
- **Why it matters:** Hard X-rays are a **flare precursor**. In many flares the hard X-ray burst *leads* the soft X-ray peak (the Neupert effect). So HEL1OS gives the forecaster an early-warning feature the soft channel alone can't. The model uses a **hardness ratio** = `hxr_total / flux_long` as feature #6 — a rising hardness ratio is a strong "flare imminent" signal.
- **It is optional, not blocking.** The whole pipeline runs on SoLEXS alone; HEL1OS is a bonus feature. When HEL1OS files are absent, that feature is just set to 0 and everything still works.
- **Format:** Level-1 FITS with multiple energy-band `COUNTS_*` columns.
- **On your disk:** `data/raw/aditya_l1/hel1os/` — **empty (0 files).** ⏳ You said you'll manage this.

**Summary of the split:**
- GOES = train the model cheaply (public, huge, labeled).
- SoLEXS = the real soft-X-ray channel, primary detection + fine-tuning target.
- HEL1OS = hard-X-ray precursor channel, optional accuracy boost.

---

## 3. What models are we using

**One model:** a **CNN-LSTM** (`model/cnn_lstm.py`). No LLMs, no transformers, no pretrained foundation models.

```
Input:  last 60 minutes × 8 features  (batch, 60, 8)
  │
  ├─ Conv1d(8→32) → ReLU → Conv1d(32→64) → ReLU → MaxPool → Dropout
  │     (CNN learns short-term shapes: rise slopes, spikes)
  │
  ├─ LSTM(64→128, 2 layers)
  │     (LSTM learns the temporal sequence — how the shape evolves)
  │
  ├─ Linear(128→1) → probability a flare peaks in next 30 min
  └─ Linear(128→5) → per-class probabilities (A/B/C/M/X)
```

The **8 input features** per minute (`model/features.py`):
1. `log_flux_short` — soft X-ray 0.5–4 Å
2. `log_flux_long` — soft X-ray 1–8 Å (the main channel)
3. `log_hxr_total` — hard X-ray total (HEL1OS; 0 if absent)
4. `dflux_1min` — 1-min rise rate
5. `dflux_5min` — 5-min rise rate (slower precursor)
6. `hardness_ratio` — hard/soft ratio (the key precursor)
7. `excess_norm` — how far above background the flux is
8. `qpp_power` — quasi-periodic pulsation power (rolling FFT peak, HEL1OS QPP precursor)

**Why CNN-LSTM and not something fancier?** With ~1.5 yr of Aditya data and 60-step windows, a Transformer would overfit. The code comment says it plainly: swap to a Transformer only if you ever exceed ~500k training samples. Don't add complexity you can't feed.

**Loss/training tricks already handled:** class imbalance (flares are rare) via `pos_weight` in `BCEWithLogitsLoss`, and negative-sample capping (5:1) in `build_windows`. You don't need to touch these.

---

## 4. The pipeline end-to-end

```
RAW FITS/NetCDF
   │  pipeline/readers.py      read_solexs / read_helios / read_goes_nc
   ▼
   │  pipeline/preprocessor.py resample, background & noise estimate
   ▼
   ├──► pipeline/detector.py   NOWCAST: find flares → FlareEvent list
   │        │
   │        ▼  pipeline/catalog.py   merge → data/processed/master_catalog.parquet
   │
   └──► model/features.py      build 8-feature windows
            │
            ▼  model/train.py   FORECAST: pretrain (GOES) → fine-tune (Aditya)
            │
            ▼  models/cnn_lstm_best.pt
                     │
                     ▼  dashboard/app.py   Dash UI
```

- `scripts/download_goes.py` — fetches GOES NetCDF (you already ran this).
- `scripts/process_all.py` — runs the whole SoLEXS→parquet→catalog loop (you already ran this too).

---

## 5. What you've actually already done (verified on disk today)

You said "no work done," but your disk disagrees — an earlier session got you further:

| Step | State |
|------|-------|
| GOES downloaded | ✅ 1,924 files, 544 MB |
| SoLEXS downloaded + extracted | ✅ ~823 days, 11 GB |
| SoLEXS processed to parquet | ✅ 797 files in `data/processed/` |
| Master flare catalog built | ✅ `master_catalog.parquet` (116 KB) — `process_all.py` has run |
| Model checkpoint | ✅ `models/cnn_lstm_best.pt` (959 KB) — a training run happened |
| HEL1OS | ⏳ empty folder |

**Caveat:** because you don't remember running these, treat the existing checkpoint and catalog as *unverified*. I'd re-run `process_all.py` and the pretrain cleanly (steps below) so you know exactly what's in them. The checkpoint being <1 MB is normal for a model this small — it's not a placeholder.

---

## 6. Commands — the exact order to run

Run everything from the project root: `C:\Projects\Claude Code\Solar Flare Project`

### Step 0 — Environment (once)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1 — Sanity-check the code (30 seconds, no data needed)
```bash
python pipeline/detector.py
```
Should print `OK: detected C… flare …`. This proves the detector works on a synthetic flare.

### Step 2 — (Re)build the SoLEXS catalog from raw data
```bash
python scripts/process_all.py --rebuild
```
Reads all `data/raw/aditya_l1/solexs/` days → writes `data/processed/*_solexs.parquet` + `master_catalog.parquet`. Prints how many flares it found. **This is your nowcasting result** — a real flare catalog from Aditya SoLEXS data.

### Step 3 — Pre-train the forecaster on GOES
```bash
python model/train.py --phase pretrain --goes-dir data/raw/goes --epochs 30
```
Trains on all 1,924 GOES days. Watch `vl_loss` fall. Writes `models/cnn_lstm_best.pt`. On CPU this is slow (hours); on your GPU, much faster.

### Step 4 — Fine-tune on Aditya SoLEXS
```bash
python model/train.py --phase finetune --sxr-dir data/processed --epochs 10
```
Loads the GOES checkpoint, freezes the CNN layers, adapts the LSTM to SoLEXS. Uses `master_catalog.parquet` for labels.

### Step 5 — Launch the dashboard
```bash
C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe dashboard/app.py
```
Opens at `http://127.0.0.1:8050`. This is what you demo to judges. (Requires Python 3.12 — that's where torch is installed.)

### Later — when you get HEL1OS
Drop the FITS files into `data/raw/aditya_l1/hel1os/`, then re-run Step 2 (it auto-detects them by date and adds `*_helios.parquet` + the hardness-ratio feature), then re-run Step 4.

---

## 7. The two things you MUST tune (don't skip)

These are placeholders in `config.yaml` that affect result *correctness*, not whether the code runs:

1. **`calibration.solexs_counts_to_wm2`** (calibrated to `1.806e-8` from 435 matched GOES/SoLEXS flare pairs) — converts SoLEXS counts/s to GOES W/m² so flare classes (C/M/X) come out right. This is already set in `config.yaml`. Wrong value = flares detected fine but mislabeled (a real M-flare called a C).

2. **`nowcasting.sigma_threshold` / `rise_rate_threshold`** — detection sensitivity. Too low = false alarms from noise; too high = misses small flares. Tune by eye against known flares in the dashboard.

Everything else in `config.yaml` has sane defaults.

---

## 8. Mental model for the ISRO submission

You have three defensible claims for judges:

1. **"We detect flares in real time from Aditya SoLEXS"** — the detector + catalog (Step 2). Works now, explainable.
2. **"We forecast flares 30 min ahead"** — the CNN-LSTM, transfer-learned from GOES so it works despite Aditya's short mission (Steps 3–4).
3. **"Here's a live operations dashboard"** — Dash by Plotly (Step 5).

HEL1OS is your "and we fuse hard X-ray precursors for earlier warning" upgrade — nice-to-have, not the foundation.

---

## Quick reference

| Thing | Location |
|-------|----------|
| Config (all tunables) | `config.yaml` |
| FITS/NetCDF readers | `pipeline/readers.py` |
| Nowcast detector | `pipeline/detector.py` |
| Feature engineering | `model/features.py` |
| Model architecture | `model/cnn_lstm.py` |
| Training (both phases) | `model/train.py` |
| Batch process raw → catalog | `scripts/process_all.py` |
| Dashboard | `dashboard/app.py` |
| GOES download | `scripts/download_goes.py` |
| Trained model output | `models/cnn_lstm_best.pt` |
| Flare catalog output | `data/processed/master_catalog.parquet` |
