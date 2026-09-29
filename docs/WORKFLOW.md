# Technical Workflow — Solar Flare Nowcasting & Forecasting

End-to-end walkthrough traced module by module, in the order data flows. Units and array shapes are called out explicitly because they're where the design decisions live.

---

## Stage 0 — Two tracks, one core

The repo is two pipelines that meet at the feature layer:

- **Nowcasting track:** `readers` → `preprocessor` → `detector.py` → `catalog.py`. No ML. Produces the flare catalog.
- **Forecasting track:** `readers` → `preprocessor` → `features.py` → `cnn_lstm.py` / `train.py`. ML. Produces the `.pt` model.

The dashboard runs both live. The catalog from track 1 also supplies the **labels** for track 2's fine-tuning — the one coupling between them.

```
RAW FITS/NetCDF
   │ readers.py
   ▼ uniform DataFrame (UTC index, named flux cols)
   │ preprocessor.py  → resample + background + noise
   ├──► detector.py ──► catalog.py ──► master_catalog.parquet   (Track A: nowcast)
   │                                        │ (labels)
   └──► features.py ──► cnn_lstm.py/train.py ◄┘ ──► cnn_lstm_best.pt  (Track B: forecast)
                                                       │
                                                       ▼ dashboard/app.py (runs A + B live)
```

---

## Stage 1 — Ingestion (`pipeline/readers.py`)

Every raw format is normalized to **one shape: a pandas DataFrame with a UTC `DatetimeIndex` and named flux columns.** Nothing downstream touches FITS/NetCDF again.

| Reader | Input | Output columns | Cadence | Units |
|--------|-------|----------------|---------|-------|
| `read_solexs` | `AL1_SLX_L1_*/SDD2/*.lc.gz` (+SDD1) | `flux_long` (SDD2), `flux_short` (SDD1 or NaN) | 1 Hz | **counts/s** |
| `read_helios` | HEL1OS L1 FITS | `hxr_total`, `hxr_soft`, `hxr_hard`, `hxr_mid` | 1 Hz | **counts/s** (÷ livetime) |
| `read_goes_nc` | GOES-16 NetCDF | `flux_short` (0.5–4 Å), `flux_long` (1–8 Å) | 1 min | **W/m²** |

SoLEXS `flux_long` comes from **SDD2** (always present); timestamps are Unix epoch seconds. GOES time is seconds since 2000-01-01 12:00 UTC. Both readers mask fill/negative values and quality-flag failures to NaN. **The unit mismatch (counts/s vs W/m²) is the central problem the rest of the code bridges.**

---

## Stage 2 — Preprocessing (`pipeline/preprocessor.py`)

Same three steps per instrument; only cadence differs (SoLEXS/HEL1OS → `1s`, GOES → `1min`):

1. **`resample_uniform`** — regular grid; forward-fill gaps ≤30 samples, longer gaps stay NaN; clip negatives to 0.
2. **`estimate_background`** — rolling **60th-percentile**, 60-min centered window. The 60th percentile sits just above quiet-Sun level and ignores flare spikes, so it tracks solar-cycle drift without being dragged up by flares.
3. **`estimate_noise`** — rolling **MAD × 1.4826** (robust σ), 10-min window. MAD instead of std so the flare itself doesn't inflate the noise estimate.

Adds `bg_long`, `noise_long`, `excess_flux_long`, etc. → `data/processed/YYYYMMDD_solexs.parquet`.

---

## Stage 3 — Nowcasting detection (`pipeline/detector.py`)

Core is `detect_flares()`:

```
excess  = flux − background
trigger = (excess > sigma_threshold · noise)   # 3σ (config)
        AND (dflux/dt > rise_rate_threshold)    # rejects slow drifts
```

- **Start:** `trigger` holds for `min_consecutive_samples` (5) in a row — kills single-sample spikes.
- **End:** flux falls back to `end_factor × background` (1.5×).
- **Peak:** argmax of flux in the start→end span.
- `_suppress_overlapping` keeps only the strongest event within 15 min.

**Units bridge & classification:** peak flux × `class_scale` before classifying. GOES scale = 1.0 (already W/m²); SoLEXS scale = `solexs_counts_to_wm2` (the `1.0e-8` placeholder). `_classify_goes` maps W/m² → A/B/C/M/X. **Key subtlety the code gets right:** detection SNR is computed in *native counts/s*, so a wrong calibration constant **mislabels class but never misses a flare**. HEL1OS events are labeled `"HXR"` (no soft-X-ray class equivalent).

Output: list of `FlareEvent` dataclasses → `events_to_dataframe`.

---

## Stage 4 — Multi-instrument fusion (`pipeline/catalog.py`)

`merge_catalogs` is SoLEXS-anchored with interval overlap and solar flare physics (the **Neupert effect**):

- **Temporal Association:** Evaluates interval overlap between SoLEXS `[start_sxr, end_sxr]` and HEL1OS `[start_hxr, end_hxr]`.
- **Neupert Rise-Phase Constraint:** HEL1OS non-thermal impulsive peaks are physically expected during the SXR rise phase:
  `start_time_sxr - 2 min <= peak_time_hxr <= peak_time_sxr + 3 min` (or direct peak proximity `±3 min` fallback).
- **Multi-Burst Aggregation:** If multiple HEL1OS bursts occur during an SXR flare, they are aggregated into 1 canonical row with dominant HXR peak flux, `hxr_burst_count`, and `neupert_lag_sec = (peak_sxr - peak_hxr_dominant)`.
- **Physical Metrics:** Computes `duration_sec`, `rise_time_sec`, `decay_time_sec`, `neupert_lag_sec`, and `hxr_burst_count`.
- **Deduplication:** `append_to_master` concatenates into `master_catalog.parquet`, dropping duplicate events within 1 min while prioritizing multi-instrument confirmed entries. `upgrade_catalog_schema` ensures 100% backward compatibility for existing catalogs.


---

## Stage 5 — Feature engineering (`model/features.py`)

`build_feature_series` merges SXR + optional HXR, resamples both to **1-min**, produces **7 features/minute**:

| # | Feature | Meaning |
|---|---------|---------|
| 0 | `log10(flux_short)` | soft 0.5–4 Å |
| 1 | `log10(flux_long)` | soft 1–8 Å (main) |
| 2 | `log10(hxr_total)` | hard X-ray, **0 if no HEL1OS** |
| 3 | `dflux_1min` | 1-min rise rate |
| 4 | `dflux_5min` | 5-min rise rate |
| 5 | `hardness_ratio` | hxr/sxr — precursor |
| 6 | `excess_norm` | (flux−bg)/bg |
| 7 | `qpp_power` | rolling FFT peak power (QPP precursor, HEL1OS) |

`log10` uses a floor (`1e-9` W/m², `1e-1` counts/s); `log1p` was tried first and gave no dynamic range for W/m² fluxes.

`build_windows` → supervised samples:
- Slide a 60-min window → each sample **(60, 8)**.
- **Label = 1 if any catalog flare peaks within the next 30 min** — the label lives in the future relative to the window; that's what makes it forecasting.
- Skip windows >30% NaN.
- **Class balancing:** negatives capped at `5 × positives` (seeded RNG). Output `X:(N,60,8)`, `y:(N,)`.

---

## Stage 6 — Model + training (`model/cnn_lstm.py`, `model/train.py`)

**Architecture** (`SolarFlareForecaster`), forward shapes:

```
x            (B, 60, 8)
→ permute    (B, 8, 60)                              Conv1d wants (N, C, L)
→ Conv1d(8→32)→ReLU→Conv1d(32→64)→ReLU→MaxPool(2)→Dropout   (B, 64, 30)
→ permute    (B, 30, 64)                             LSTM wants (N, L, H)
→ LSTM(64→128, 2 layers, dropout 0.3), last hidden h[-1]    (B, 128)
→ Linear(128→1) → logit                              (B,)
→ Linear(128→5) → class logits  [optional, when n_classes=5]  (B, 5)
```

CNN learns local shapes (rise slopes, spikes); LSTM learns how the 30-step sequence evolves. Output is a raw logit → `BCEWithLogitsLoss`.

**Two-phase training:**

- **Pretrain** (`--phase pretrain`): loads GOES NetCDF in **90-day chunks (RAM ~400 MB)**, **self-labels** each day by running `detect_goes_flares` (no external label file — the detector generates labels), trains all layers, `lr=1e-3`.
- **Finetune** (`--phase finetune`): loads GOES checkpoint, **freezes Conv layers** (low-level detectors transfer; only LSTM+head adapt to SoLEXS), labels from `master_catalog.parquet`, `lr=1e-4`.

**Two correctness-critical details:**
1. **Class imbalance** — `pos_weight = n_neg/n_pos` in the loss.
2. **Normalization persistence** — `make_loaders` z-scores on train stats; `save_checkpoint` stores `norm_mean`/`norm_std` **inside the `.pt`**. Inference must reproduce the exact z-scoring or forecasts are garbage. `load_norm` reads them back; legacy checkpoints without them warn.

Best-val-loss → `models/cnn_lstm_best.pt`. `ReduceLROnPlateau` + grad clip 1.0.

---

## Stage 7 — Evaluation (`model/evaluate.py`)

Not accuracy (useless on imbalanced data):
- **Threshold sweep** at 0.3/0.5/0.7: TPR (recall), **FAR** (false-alarm rate), precision, F1, AUC-ROC.
- **`compute_lead_times`** — the operational metric: minutes between each alert and the actual flare peak (median + p90), plus false-alarm count. Tells you if a 30-min-ahead forecast is actually useful.

---

## Stage 8 — Serving (`dashboard/app.py`)

Dash by Plotly, two modes (Historical / Live-poll-60s via `dcc.Interval`). Per selected date:

1. Load processed parquet (SoLEXS or GOES).
2. **Nowcast:** run detector live → alert banners colored by class (X/M red, C orange).
3. **Forecast:** `run_forecaster` slides the model minute-by-minute (applying stored `norm_mean/std`) → `P(flare)` per minute; banner fires when `P ≥ threshold`.
4. Plots: SXR light curve + background + flare markers, HXR curve, hardness ratio, probability trace with threshold line.
5. Show/export `master_catalog.parquet`.

Inference uses the **same** `build_feature_series` + normalization as training — no separate feature path, which is why the persisted norm stats are load-bearing.

---

## One-line mental model

> Raw FITS/NetCDF → uniform 1-min series with background/noise → **(A)** threshold detector → flare catalog, and **(B)** 60-min×7-feature windows labeled "flare in next 30 min" → CNN-LSTM (pretrained on GOES, fine-tuned on SoLEXS) → probability. Dashboard runs A and B together.

Two decisions everything hinges on: **GOES self-labels via the detector** (no manual labels for pretraining), and **counts→W/m² calibration affects only class labels, never detection** (untuned calibration degrades gracefully).
