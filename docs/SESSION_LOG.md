# Solar Flare Project — Session Log & Technical Reference

> Covers sessions spanning approximately 2026-09-15 to 2026-09-18.  
> This file documents what was built, what was found, what was fixed, and what is still running.

---

## 1. Project Overview

**Goal:** Solar flare nowcasting (real-time detection) and 30-minute-ahead forecasting from ISRO Aditya-L1 satellite instruments. Originally built for ISRO Bharatiya Antariksh Hackathon 2026 PS-15; now a portfolio-grade space weather ML pipeline.

**Satellite instruments used:**
| Instrument | Channel | Units | Role |
|---|---|---|---|
| SoLEXS (Aditya-L1) | 1–8 Å soft X-ray | counts/s | Primary flux source |
| HEL1OS (Aditya-L1) | 10–160 keV hard X-ray | counts/s | Spectral context |
| GOES-16 XRS | 1–8 Å | W/m² | Pre-training teacher signal |

**Two-track architecture:**
- **Track A — Nowcasting:** Pure signal processing. Threshold detector (σ×noise + rise-rate guard) → `FlareEvent` dataclass → master catalog parquet. No ML.
- **Track B — Forecasting:** CNN-LSTM model pretrained on GOES-16, fine-tuned on Aditya-L1 SoLEXS. 60-minute windows of 8 features → P(flare in next 30 min) + A/B/C/M/X intensity class.

---

## 2. Model Architecture

**Class:** `SolarFlareForecaster` in `model/cnn_lstm.py`

```
Input: (batch, 60 timesteps, 8 features)
  → Conv1d(8 → 32, kernel=3, ReLU)
  → Conv1d(32 → 64, kernel=3, ReLU)
  → MaxPool1d(2)
  → LSTM(64 → 128, num_layers=2, batch_first=True)
  → Linear(128 → 1)          ← binary head: P(flare), BCEWithLogitsLoss
  → Linear(128 → 5)  [optional] ← class head: A/B/C/M/X, CrossEntropyLoss(ignore_index=-1)
```

**8 features** (`model/features.py` → `FEATURE_NAMES`):
1. `log_flux_short` — log10 of SXR short-channel flux
2. `log_flux_long` — log10 of SXR long-channel flux
3. `log_hxr_total` — log10 of HXR total counts (0 if no HEL1OS)
4. `dflux_1min` — 1-minute delta in long-channel flux
5. `dflux_5min` — 5-minute delta in long-channel flux
6. `hardness_ratio` — HXR / SXR (spectral hardening proxy)
7. `excess_norm` — (flux − background) / background
8. `qpp_power` — quasi-periodic pulsation power (HEL1OS PS-15 feature)

**Z-score normalisation:** applied per-feature on the training set; `norm_mean` and `norm_std` are saved inside the checkpoint so inference is always consistent.

**Config keys** (`config.yaml`):
```yaml
forecasting:
  input_window_min: 60
  lead_time_min: 30
  alert_threshold: 0.5
  model_features: 8
  n_classes: 5
calibration:
  solexs_counts_to_wm2: 1.806e-08   # fitted from 435 matched flare pairs
```

---

## 3. Data

### 3.1 Datasets

| Dataset | Location | Count | Format |
|---|---|---|---|
| GOES-16 XRS | `data/raw/goes/` | 1,922 NC files | NetCDF4 |
| SoLEXS processed | `data/processed/YYYYMMDD_solexs.parquet` | 824 files | Parquet |
| HEL1OS processed | `data/processed/YYYYMMDD_helios.parquet` | ~1,008 files | Parquet |
| Master catalog | `data/processed/master_catalog.parquet` | 3,435 events | Parquet |

### 3.2 Master Catalog Breakdown

| Class | Count |
|---|---|
| X | 45 |
| M | 589 |
| C | 2,427 |
| B | 257 |
| HXR-only | 117 |
| Multi-instrument confirmed | 171 |

### 3.3 SoLEXS Calibration

`scripts/calibrate_solexs.py` fitted a scalar conversion from SoLEXS long-channel counts/s to GOES-equivalent W/m²:

```
solexs_counts_to_wm2 = 1.806e-08
```

Fitted from **435 matched GOES/SoLEXS flare pairs** (peak-flux linear regression). Updated in `config.yaml` line 16.

---

## 4. Training Pipeline

### 4.1 Two-Phase Strategy

**Phase 1 — Pretrain on GOES-16** (large corpus, calibrated W/m² labels):
- 1,922 GOES NetCDF files → 1.8M sliding windows (60 min, step 1 min)
- 30 epochs, LR=1e-3, all layers trainable
- Self-labels flares via `detect_goes_flares()`
- Result: `val_loss = 0.1145`, checkpoint saved to `models/cnn_lstm_best.pt`

**Phase 2 — Fine-tune on Aditya-L1** (domain adaptation):
- 824 SoLEXS parquets → ~660K windows
- Conv layers frozen (GOES-learned spatial patterns preserved)
- LSTM + linear heads retrained at lower LR
- Combined loss: `BCEWithLogitsLoss(pos_weight=6.1) + CrossEntropyLoss(ignore_index=-1)`

### 4.2 Commands (sequential order)

```powershell
# 1. Pretrain (delete stale checkpoint first)
Remove-Item models/cnn_lstm_best.pt -ErrorAction SilentlyContinue
python model/train.py --phase pretrain --goes-dir data/raw/goes --epochs 30

# 2. Rebuild master catalog (can run in parallel with pretrain)
python scripts/process_all.py --rebuild

# 3. Calibrate SoLEXS scale (can run after catalog exists)
python scripts/calibrate_solexs.py

# 4. Fine-tune (frozen conv, 10 epochs)
python model/train.py --phase finetune --epochs 10

# 5. Evaluate
python scripts/evaluate_model.py

# 6. Extended fine-tune (frozen conv, 25 more epochs at lower LR)
python model/train.py --phase finetune --epochs 25 --lr 3e-5

# 7. Evaluate again
python scripts/evaluate_model.py

# 8. Unfrozen conv fine-tune (current — see Section 5)
python model/train.py --phase finetune --epochs 15 --lr 5e-6 --unfreeze-conv
```

### 4.3 Training Results Summary

| Run | Epochs | LR | Frozen Conv | val_loss | AUC | TSS@0.5 | TPR@0.3 | Median Lead | P90 Lead | FAR |
|---|---|---|---|---|---|---|---|---|---|---|
| Fine-tune #1 | 10 | 1e-4 | ✓ | 1.6162 | 0.7936 | 0.325 | 85.8% | 13.95 min | 26.85 min | 82.94% |
| Fine-tune #2 | 25 | 3e-5 | ✓ | 1.5844 | **0.813** | 0.349 | 88.1% | 14.0 min | 27.0 min | 83.86% |
| Fine-tune #3 | 15 | 5e-6 | ✗ | **1.5759** | **0.818** | **0.361** | 87.7% | 13.98 min | 26.98 min | 83.75% |

**Notes on metrics:**
- AUC 0.813 is solid for space weather (NOAA operational models: 0.70–0.85)
- TSS 0.349 = decent skill, room to improve
- P90 lead of 27 min is 3 min short of ISRO's 30-min target
- False Alarm Rate (FAR) 83% is the main weakness — model is recall-biased due to `pos_weight=6.1`
- Combined val_loss ~1.58 looks high but it's `BCE + CrossEntropy`; BCE alone would show lower value

**Why val_acc looks low:** `pos_weight=6.1` biases the binary head toward predicting positive (flare). Raw accuracy drops below the 85.93% null baseline (predict-all-negative) but AUC is the right metric here.

---

## 5. Unfrozen Conv Fine-Tune (Complete)

**Motivation:** After 25 frozen-conv epochs, loss was still slowly decreasing (1.6162 → 1.5844). The conv layers output GOES-tuned feature maps that the LSTM must reinterpret for SoLEXS data. Unfreezing at very low LR lets conv layers adapt without catastrophic forgetting.

**Command:**
```powershell
python model/train.py --phase finetune --epochs 15 --lr 5e-6 --unfreeze-conv
```

**What changed in `train.py`:**
- Added `--unfreeze-conv` flag (argparse, `store_true`)
- Fine-tune branch now: `freeze = not args.unfreeze_conv`

**Results:** All 15 epochs ran on CUDA. Best val_loss=**1.5759** at epoch 14 (vs fine-tune #2's 1.5844). AUC improved to **0.818** (from 0.813), TSS to **0.361** (from 0.349). Model saved to `models/cnn_lstm_best.pt`. HSS@0.5 = 0.310 (new metric now tracked).

**Full eval output:**
```
threshold  TPR     FAR     TSS     HSS     F1      AUC
  0.3    0.877  0.432  0.445  0.161  0.295  0.818
  0.5    0.457  0.096  0.361  0.310  0.387  0.818  ← operational threshold
  0.7    0.222  0.017  0.205  0.283  0.321  0.818
Lead P90: 26.98 min  |  FAR: 83.75%
```

**Epoch progression:**
```
Ep  9 | vl_loss=1.5807  Ep 10 | vl_loss=1.5806  Ep 11 | vl_loss=1.5784
Ep 12 | vl_loss=1.5787  Ep 13 | vl_loss=1.5773  Ep 14 | vl_loss=1.5759 ← best
Ep 15 | vl_loss=1.5760
```

**Evaluate with:**
```powershell
& "C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe" scripts/evaluate_model.py
```

---

## 6. Dashboard

### 6.1 Stack

- **Framework:** Dash by Plotly (explicitly NOT Streamlit — previous session used Streamlit, this session replaced it entirely)
- **File:** `dashboard/app.py`
- **Launch:**
  ```powershell
  C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe dashboard/app.py
  # Open http://127.0.0.1:8050
  ```
- **Note:** Must use Python 3.12 (torch is installed there, not in Python 3.11)

### 6.2 Architecture

```
Layout:
  Header bar ("☀️ Solar Flare Monitor — Aditya-L1 | ISRO PS-15")
  Controls: DatePickerSingle | Instrument dropdown | Threshold slider | Options checklist | Load button
  Alert banners (dcc div, id="alert-banners")
  Status bar (id="status-bar")
  dcc.Tabs (id="main-tabs"):
    Tab "curves"   → Light Curves & Nowcast
    Tab "forecast" → Forecast
    Tab "catalog"  → Catalog
  dcc.Loading → html.Div(id="tab-content")
  dcc.Store(id="store-main")          ← JSON cache of loaded day's data
  dcc.Interval(id="live-interval")    ← 60s live refresh (disabled by default)

Callbacks:
  toggle_live:    options-check → live-interval.disabled
  load_day:       load-btn + live-interval → store-main + status-bar
  update_alerts:  store-main + threshold-slider → alert-banners
  render_tab:     main-tabs + store-main + threshold-slider → tab-content
```

### 6.3 dcc.Store Schema

```json
{
  "sxr":        "<JSON split-orient string of SXR DataFrame>",
  "hxr":        "<JSON split-orient string of HXR DataFrame or null>",
  "sxr_evs":    [{"peak_time": "...", "flare_class": "B5.1", ...}],
  "hxr_evs":    [...],
  "master_evs": [...],
  "prob":        {"times": ["2026-09-12 00:00:00", ...], "probs": [0.023, ...]},
  "cls":         {"_times": [...], "A": [...], "B": [...], "C": [...], "M": [...], "X": [...]},
  "date":        "2026-09-12",
  "instrument":  "solexs"
}
```

**Serialisation helpers:**
- `_df_to_json(df)` → converts DataFrame to JSON with `orient="split"`, stringified index
- `_json_to_df(s)` → `pd.read_json(io.StringIO(s), orient="split")` + datetime index conversion
  - ⚠️ Must use `io.StringIO(s)` — pandas ≥ 2.0 interprets raw strings as file paths (see Bug #1)

### 6.4 Tab Content

**Light Curves & Nowcast (`tab="curves"`):**
- SXR Flux (1–8 Å) — log-scale, blue line, dashed background, annotated flare vlines
- HXR Count Rate (HEL1OS 10–160 keV) — purple line
- Hardness Ratio (HXR/SXR) — orange fill
- Detected Flares Today — sortable DataTable

**Forecast (`tab="forecast"`):**
- KPI row: Current P(flare) | AUC | TSS | Lead P90
- Most-likely class tile (from class head softmax)
- P(flare in next 30 min) probability chart with threshold line
- Intensity-Class Probabilities (A/B/C/M/X) stacked traces

**Catalog (`tab="catalog"`):**
- KPI tiles: Total | X | M | C | B | Multi-instrument
- Full master catalog DataTable (sortable, filterable, 25 rows/page)
- "⬇ Download CSV" → Flask route `/download-catalog`

### 6.5 Flask Route

```python
@app.server.route("/download-catalog")
def _download_catalog():
    # Returns master_catalog.parquet as CSV download
```

### 6.6 Dark Theme Tokens

```python
BG     = "#08081a"   # page background
CARD   = "#10102a"   # card background
CARD2  = "#14143a"   # secondary card / selected tab
BORDER = "#22224a"   # borders
FG     = "#d0d8ff"   # foreground text
DIM    = "#5060a0"   # muted text / labels
BLUE   = "#4488ff"   # accent / SXR line
ORANGE = "#ff8844"   # P(flare) fill
```

---

## 7. Bugs Found and Fixed

### Bug 1 — `pd.read_json` treats large strings as file paths (pandas ≥ 2.0)

**File:** `dashboard/app.py`, function `_json_to_df`  
**Symptom:** 500 error on `tab-content` callback; `FileNotFoundError: No such file or directory: '{"columns": [...], "index": [...]...}'`  
**Cause:** Pandas ≥ 2.0 changed `read_json` to check if a string argument is a file path first. A very long JSON string (the 86,400-row SXR DataFrame) triggers the file-path check and fails.  
**Fix:**
```python
# Before (broken on pandas ≥ 2.0):
df = pd.read_json(s, orient="split")

# After (correct):
df = pd.read_json(io.StringIO(s), orient="split")
```

### Bug 2 — Duplicate `yaxis` kwarg in `update_layout()`

**File:** `dashboard/app.py`, functions `_fig_probability` and `_fig_class_prob`  
**Symptom:** `TypeError: Figure.update_layout() got multiple values for keyword argument 'yaxis'`  
**Cause:** `_GRAPH_LAYOUT` dict already contains `yaxis=dict(showgrid=True, ...)`, and the call also passed `yaxis=dict(range=[0, 1])` explicitly.  
**Fix:**
```python
# Before (conflict):
fig.update_layout(..., yaxis=dict(range=[0, 1]), **_GRAPH_LAYOUT)

# After (use magic underscore notation):
fig.update_layout(..., yaxis_range=[0, 1], **_GRAPH_LAYOUT)
```

---

## 8. Environment Notes

### Python Versions

| Version | Path | Has torch? | Has plotly/dash? |
|---|---|---|---|
| 3.11 | `C:\Users\HP\AppData\Local\Programs\Python\Python311\` | ✗ | ? |
| 3.12 | `C:\Users\HP\AppData\Local\Programs\Python\Python312\` | ✓ | ✓ |

Always use Python 3.12 for this project:
```powershell
& "C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe" <script>
```

### Key Packages (Python 3.12)
- `torch` (CPU build)
- `plotly`, `dash`
- `pandas` ≥ 2.0
- `pyarrow` 25.0.1 (parquet support)
- `numpy`, `scipy`, `yaml`, `tqdm`

### Project Root
```
C:\Projects\Claude Code\Solar Flare Project\
```

### Key File Paths

| Purpose | Path |
|---|---|
| Config | `config.yaml` |
| Model checkpoint | `models/cnn_lstm_best.pt` |
| Master catalog | `data/processed/master_catalog.parquet` |
| SoLEXS parquets | `data/processed/YYYYMMDD_solexs.parquet` |
| HEL1OS parquets | `data/processed/YYYYMMDD_helios.parquet` |
| GOES NetCDF | `data/raw/goes/sci_xrsf-l2-avg1m_g16_d*.nc` |
| Dashboard | `dashboard/app.py` |
| Model | `model/cnn_lstm.py` |
| Features | `model/features.py` |
| Training | `model/train.py` |
| Evaluate | `scripts/evaluate_model.py` |
| Catalog build | `pipeline/catalog.py` |
| Detector | `pipeline/detector.py` |
| Preprocessor | `pipeline/preprocessor.py` |

---

## 9. Pipeline Verification Outputs

### Pretrain output (Phase 1)
```
GOES NetCDF files: 1922/1922
Dataset: 1,802,xxx windows | pos=1.17%
Ep  1 | tr_loss=... | vl_loss=...
...
Best val_loss=0.1145 -> saved to models/cnn_lstm_best.pt
```

### Calibration output
```
Matched 435 GOES/SoLEXS flare pairs
Fitted scale: 1.806e-08 counts/s → W/m²
→ Updated config.yaml: solexs_counts_to_wm2: 1.806e-08
```

### Catalog rebuild output
```
3,435 events total
B: 257 | C: 2,427 | M: 589 | X: 45 | Multi-instrument confirmed: 171
```

### Evaluate output (after fine-tune #2, best so far)
```
AUC:              0.813318
TSS @ 0.5:        0.349
TPR @ threshold 0.3: 88.1%
Median lead time: 14.0 min
P90 lead time:    27.0 min
False alarm rate: 83.86%
```

---

## 10. Known Issues & Future Work

### Remaining gap to ISRO targets
- **P90 lead time:** 27 min achieved, target is 30 min (3 min short)
- **False alarm rate:** 83% — reduce by tuning threshold or adding FAR-penalty term to loss
- **Unfrozen conv run** (in progress): may improve AUC toward 0.83+

### HEL1OS V111 format
Files from Oct–Nov 2025 use a different internal ZIP structure (`V111` vs `V100`/`V101`). The reader `pipeline/readers.py` cannot parse them. Those days have no HXR data currently.

### Live data ingestion
The dashboard has a 60-second live refresh (`dcc.Interval`) but no automated pipeline to download and process today's Aditya-L1 data. Would need a cron job calling `process_all.py` for the current date.

### SoLEXS date picker interaction
The Dash `dcc.DatePickerSingle` does not respond to raw DOM value manipulation or `form_input`. Reliable workaround:
1. Use `form_input` to set the text field value
2. Press Tab to blur
3. Click Load

The date picker's React internal state then passes the ISO date string to the `load_day` callback via `State("date-picker", "date")`.

### Next training step (after unfrozen conv run)
```powershell
python scripts/evaluate_model.py
```
If AUC > 0.83 and P90 lead > 27 min, the model is ready for a final eval report.  
If not, consider: longer unfrozen run, label smoothing on the class head, or threshold tuning.

---

## 11. Quick-Start Checklist

To restart from scratch on a new machine or after a long break:

```powershell
# 1. Ensure Python 3.12 with torch, plotly, dash, pandas, pyarrow
& "...\Python312\python.exe" -c "import torch, plotly, dash, pandas, pyarrow; print('ok')"

# 2. Check data exists
Get-ChildItem data/processed -Filter "*_solexs.parquet" | Measure-Object  # should be 824
Get-Item models/cnn_lstm_best.pt                                           # checkpoint

# 3. Verify calibration in config.yaml
Select-String "solexs_counts_to_wm2" config.yaml                          # should be 1.806e-08

# 4. Launch dashboard
& "...\Python312\python.exe" dashboard/app.py
# → http://127.0.0.1:8050

# 5. Evaluate model
& "...\Python312\python.exe" scripts/evaluate_model.py
```

---

## 12. Session Log — 2026-09-29: Catalog Association & Neupert Effect Overhaul

### Accomplishments:
1. **Physical Flare Association (`pipeline/catalog.py`)**:
   - Replaced naive `±3 min` peak-only matching with a hybrid interval overlap algorithm incorporating the **Neupert effect**.
   - Enforced physical rise-phase window: `start_time_sxr - 2m <= peak_time_hxr <= peak_time_sxr + 3m`, with fallback peak proximity.
   - Implemented multi-burst aggregation: multiple HXR impulsive sub-bursts during a single SXR event are grouped into 1 canonical flare, saving dominant `helios_peak_flux`, `hxr_burst_count`, and `neupert_lag_sec = (peak_sxr - peak_hxr_dominant)`.
2. **Schema Extension & Migration**:
   - Added `duration_sec`, `rise_time_sec`, `decay_time_sec`, `neupert_lag_sec`, and `hxr_burst_count` to canonical catalog schema.
   - Built `upgrade_catalog_schema(df)` ensuring seamless backward compatibility.
   - Migrated existing `data/processed/master_catalog.parquet` (3,435 events) in-place with all 13 columns cleanly populated.
3. **Dashboard Catalog Tab Enhancements (`dashboard/app.py`)**:
   - Added **Median Neupert Lag** KPI card in the Catalog tab header.
   - Added human-formatted columns (`Duration`, `Rise`, `Decay`, `Neupert Lag`, `HXR Bursts`) to the interactive `dash_table.DataTable`.
4. **Unit Test Suite (`tests/test_catalog.py`)**:
   - Created 8 unit tests covering exact matches, Neupert rise-phase lags, multi-burst aggregation, open-ended flares, standalone HXR events, schema migration, cluster deduplication, and catalog summary string.
   - 100% test pass rate.

