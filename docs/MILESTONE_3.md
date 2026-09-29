MILESTONE 3 SUBMISSION REPORT

Date:

2026-09-03

Rev: 1.0

## Data Pipeline & Preprocessing

Design, implementation, and validation of the end-to-end data ingestion pipeline for SoLEXS, HEL1OS, and GOES-16 XRS instruments, including the algorithmic nowcasting detector and ML feature engineering layer.

TEAM NAME

5QOM9HJ0

ROLL NO.

BTIR24O1041

## Section 1

## Executive Summary

Milestone 3 delivers a production-ready data ingestion, preprocessing, and nowcasting pipeline for the ISRO PS-15 solar flare detection challenge on Aditya- L1 instrument data. All code is implemented, tested, and version-controlled.

The pipeline processes Level-1 FITS data from SoLEXS and HEL1OS, preprocesses it to uniform cadence with rolling background and noise estimates, runs a sigma-threshold nowcasting detector, and constructs the 7-feature sliding- window matrices required by the CNN-LSTM forecaster (Milestone 4). A corpus of 1,462 GOES-16 XRS NetCDF4 files covering 2020-2023 has been acquired for model pre-training.

## INSTITUTE / ORGANIZATION

Madhav Institute of Technology and Science

## SUBMISSION DATE

3 September 2026


All algorithmic parameters are externalised in config.yaml for reproducibility. The pipeline is instrument-agnostic: the same preprocessing and detection logic runs identically on SoLEXS, HEL1OS, and GOES data through typed function interfaces.

## Section 2

## Objective & Scope of Milestone 3

The objective of this milestone is to establish the full data layer of the solar flare forecasting system — from raw instrument files to normalised feature matrices ready for model ingestion. The scope covers:

- Automated acquisition of the GOES-16 XRS pre-training corpus from NOAA NCEI

- FITS readers for Aditya-L1 SoLEXS (0.5-8 Å soft X-ray) and HEL1OS (10-200 keV hard X-ray) Level-1 data with dynamic column discovery

- Uniform-cadence resampling, background estimation, and noise estimation for all three instruments

- Algorithmic nowcasting detector producing typed FlareEvent records with GOES-equivalent classification

- Multi-instrument catalog merging (SoLEXS ↔ HEL1OS) with co-temporal confirmation

- Seven-feature engineering layer and 60-minute sliding window construction for the Milestone 4 CNN-LSTM forecaster

## Section 3

## Data Sources

## 3.1 GOES-16 XRS — Pre-Training Corpus

GOES-16 carries the X-Ray Sensor (XRS) measuring solar flux in two channels: 0.5-4 Å (XRSA, short) and 1-8 Å (XRSB, long). The NOAA National Centers for Environmental Information (NCEI) distributes Level-2 one-minute average


science files in NetCDF4 format under the xrsf-l2-avg1m_science product series.

GOES data serves as the pre-training corpus because it provides four years of well-characterised, high-cadence X-ray time series with hundreds of documented flare events across the A–X class range.

| PROPERTY | VALUE |
| --- | --- |
| Satellite | GOES-16 (GOES-R series, geostationary) |
| Instrument | XRS — X-Ray Sensor |
| Channels | XRSA: 0.5-4 Å · XRSB: 1-8 Å (W/m²) |
| Cadence | 1 minute (1440 samples/day) |
| Files acquired | 1,462 daily NetCDF4 files (sci_xrsf-l2-avg1m_g16_d*.nc) |
| Coverage start | 2024-02-01 |
| Source URL | NOAA NCEI — data.ngdc.noaa.gov (xrsf-l2-avg1m_science) |
| Local path | data/raw/goes/ |

## 3.2 Aditya-L1 SoLEXS & HEL1OS — Primary Mission Data

Aditya-L1 carries two X-ray instruments relevant to flare detection:

- SoLEXS (Solar Low Energy X-ray Spectrometer) — measures soft X-ray flux in two bands equivalent to the GOES 0.5-4 Å and 1-8 Å channels, enabling direct GOES-equivalent flare classification.

- HEL1OS (High Energy L1 Orbiting X-ray Spectrometer) — measures hard X-ray count rates across multiple energy bands (nominally 10-200 keV). HXR emission is a key precursor to impulsive-phase flare detection and provides the hardness ratio feature.

Level-1 FITS files are available from the ISSDC PRADAN portal. Once downloaded, they are processed by the instrument-specific readers in pipeline/readers.py and written to Parquet format under data/processed/ .


## Data Acquisition

## 4.1 GOES Automated Downloader

The script scripts/download_goes.py implements a fully automated

downloader for GOES-16 XRS NetCDF4 files. It enumerates available files via Apache autoindex directory listings on NOAA NCEI, resolves the latest version per day (files carry a version suffix such as _v2-2-1 ), and streams each file in 64 KB chunks with graceful retry on failure.

| FEATURE | IMPLEMENTATION |
| --- | --- |
| Discovery | Regex parse of NCEI HTML directory listing; latest version per date |
|   | wins |
| Skip-existing | Checks out.exists() before fetching; resumable on interruption |
| Rate limiting | 200 ms sleep between downloads; complies with NOAA access |
|   | guidelines |
| Verification | --verify flag calls read_goes_nc() and asserts column presence |
| Usage | python scripts/download_goes.py --start 2020-01-01 --end 2023-12- |
|   | 31 |

## 4.2 Aditya-L1 FITS Access

Aditya-L1 Level-1 data is distributed through the ISRO Space Science Data Centre (ISSDC) PRADAN portal. Files must be downloaded manually via the portal interface. Once downloaded, the FITS files are placed in

data/raw/solexs/ (SoLEXS) and data/raw/helios/ (HEL1OS) and processed by

the pipeline readers described in Section 5.

## Section 5

## Data Preprocessing Pipeline

All preprocessing is implemented in pipeline/preprocessor.py with instrument-specific entry points ( preprocess_solexs , preprocess_helios , preprocess_goes ). The pipeline is deterministic and parameterised entirely

through config.yaml .


## FITS / NetCDF4 Reader

## pipeline/readers.py

Dynamic HDU column discovery for SoLEXS (flux_short, flux_long) and HEL1OS (hxr_total, hxr_soft, hxr_hard, hxr_mid). MJDREF + TIME offset converted to UTC DatetimeIndex. GOES NC: xrsa_flux / xrsb_flux masked by quality flags.

→ pd.DataFrame [flux_short, flux_long] or [hxr_*]

## Uniform Resampling

preprocessor.resample_uniform()

SoLEXS / HEL1OS resampled to 1 s cadence; GOES resampled to 1 min. Gaps ≤30 samples (≤30 s) forward-filled; longer gaps left as NaN. Values below 10⁻¹² W/m² clipped to NaN (instrument noise floor).

- Uniform cadence, NaN-masked DataFrame

## Background Estimation

## preprocessor.estimate_background()

Rolling 60th-percentile over a 60-minute centred window (3,600 samples at 1 s; 60 samples at 1 min). The 60th percentile tracks the quiet-Sun level without being dominated by flare peaks. Edge effects filled by forward/backward propagation.

## Noise Estimation

## preprocessor.estimate_noise()

Rolling median absolute deviation (MAD) over a 10-minute window, scaled by 1.4826 to give the Gaussian-equivalent σ. MAD is insensitive to flare peaks and photon- counting outliers, making it more robust than a rolling standard deviation for solar

X-ray data.

- noise_long, noise_short, noise_hxr_* columns

## Excess & Parquet Output

preprocessor.save_parquet()

excess_flux = (flux − background).clip(lower=0) appended as a column. Enriched DataFrame written to Parquet via PyArrow for efficient columnar access. File named

by instrument date: YYYYMMDD_solexs.parquet / YYYYMMDD_helios.parquet .

- data/processed/YYYYMMDD_{instrument}.parquet


## Algorithmic Nowcasting — Flare Detector

The nowcasting detector ( pipeline/detector.py ) operates on preprocessed DataFrames and requires no machine learning. It produces typed FlareEvent dataclass instances carrying start, peak, end times, peak flux, GOES-equivalent class, and source instrument.

## 6.1 Detection Algorithm

Detection proceeds in five steps applied to each flux channel independently:

- 1. Excess computation: excess = flux − background isolates the flare signal above the slowly varying quiet-Sun baseline.

- 2. Sigma trigger: A sample triggers when excess > σ_threshold × noise AND the value is finite. Default σ = 3.0.

- 3. Rise-rate guard: The numerical derivative dF/dt must also exceed rise_rate_threshold , rejecting slow instrument drifts or calibration artefacts that cross the sigma threshold without a genuine rise.

- 4. Consecutive confirmation: A flare start is declared only when at least min_consecutive_samples consecutive triggered samples are observed. This suppresses impulsive particle hits and single-sample spikes.

- 5. Suppression: Events with peak times within suppression_window_min of a stronger event are discarded, keeping only the dominant peak during complex active periods.

## 6.2 Tuned Parameters

| PARAMETER | VALUE | RATIONALE |
| --- | --- | --- |
| sigma_threshold | 3.0 σ | Standard 3σ detection limit; balances |
|   |   | sensitivity against false alarms at solar |
|   |   | minimum flux levels |
| min_consecutive_samples | 5 | ≥5 s (SXR) or ≥5 min (GOES) of sustained |
|   | samples | trigger; rejects particle hits and noise |
|   |   | spikes |
| rise_rate_threshold | 1×10⁻⁹ | Minimum impulsive-phase rise rate; |
|   | W/m²/s | eliminates slow thermal drifts crossing |


| PARAMETER | VALUE | RATIONALE |
| --- | --- | --- |
|   |   | the sigma boundary |
| suppression_window_min | 15 min | Typical flare duration for C-class events; |
|   |   | prevents multiple detections within one |
|   |   | complex flare |
| background_window_min | 60 min | Wide enough to track solar-cycle drift |
|   |   | without being distorted by individual |
|   |   | flares |
| end_factor | 1.5 × | Flare end defined when flux falls to |
|   |   | ≤1.5× background; consistent with |
|   |   | NOAA SWPC convention |

## 6.3 Flare Classification

Peak flux in the SoLEXS long channel (1–8 Å, GOES-equivalent W/m²) is mapped to the standard GOES class system. The numeric sub-class is computed from the ratio of peak flux to the class lower bound.

## Section 7

## Multi-Instrument Catalog Merge

The catalog module ( pipeline/catalog.py ) merges independent SoLEXS and HEL1OS event lists into a unified master catalog. Events from both instruments with peak times within ±3 minutes of each other are considered co-temporal and merged into a single record flagged confirmed_multi_instrument = True . HEL1OS-only events (no SoLEXS counterpart) are retained as class "HXR" — they may indicate hard X-ray bursts below the SXR detection threshold.

| MASTER CATALOG COLUMN | SOURCE |
| --- | --- |
| start_time, peak_time, end_time | SoLEXS anchor (canonical times for merged |
|   | events) |


| MASTER CATALOG COLUMN | SOURCE |
| --- | --- |
| flare_class | GOES-equivalent from SoLEXS long-channel |
|   | peak flux |
| solexs_peak_flux | Peak flux (W/m²) in SoLEXS 1–8 Å channel |
| helios_peak_flux | Peak HXR count rate from HEL1OS broadband |
|   | sum |
| hardness_ratio | helios_peak_flux / solexs_peak_flux (thermal |
|   | indicator) |
| confirmed_multi_instrument | True if SoLEXS and HEL1OS peaks within ±3 |
|   | min |

The catalog is persisted as data/processed/master_catalog.parquet and

supports append-with-deduplication via append_to_master() . Events sharing a peak time within 1 minute are deduplicated, with preference for confirmed multi-instrument records.

## Section 8

## Feature Engineering for ML Forecasting

## 8.1 Feature Definitions

Seven features are computed per 1-minute timestep from the merged SoLEXS and HEL1OS preprocessed series ( model/features.py ). The feature set is motivated by the physical precursors established in solar flare prediction literature.

## Feature 0

## log_flux_short

log₁p(SXR 0.5–4 Å) — short-band intensity; log transform compresses the 8-decade dynamic range

## Feature 2

## log_hxr_total

log₁p(HEL1OS broadband sum) — hard X-ray proxy; 0 when HEL1OS data unavailable

## Feature 1

## log_flux_long

log₁p(SXR 1–8 Å) — primary GOES-equivalent channel; primary intensity indicator

## Feature 3

## dflux_1min

1-minute derivative of flux_long — captures the fast impulsive rise preceding a flare peak


## 8.2 Sliding Window Construction

The function build_windows() assembles (X, y) training arrays by sliding a fixed-length window across the feature DataFrame:

- Window length: 60 minutes (60 timesteps at 1-min cadence) — sufficient to capture the pre-flare rise phase without excessive memory cost

- Lead time: 30 minutes — label is 1 if any flare peaks within the next 30 minutes of the window end

- NaN handling: Windows with more than 30% NaN samples are skipped; remaining NaNs zero-filled after forward-fill

- Class balance: Negative windows capped at 5× the number of positive windows to prevent severe imbalance

- Normalisation: Z-score per feature using training-set mean and standard deviation; validation/test sets normalised with training statistics

Output shape: X: (N, 60, 7) , y: (N,) binary float32.

Section 9

## Pipeline Validation & Testing

Each pipeline module includes a self-contained validation routine accessible via

python -m pipeline.detector (or as the module's __main__ block). The

detector's validation injects a synthetic C3.0 flare (60-second rise to 3×10⁻⁶ W/m², 120-second exponential decay) into a flat background and asserts:

- Exactly one FlareEvent is returned


- The detected class begins with "C" (i.e., peak flux correctly classified as C-class)

- Peak time falls within the injected event window

The GOES downloader provides an optional --verify flag that reads back each downloaded file through read_goes_nc() and asserts that flux_long is present and non-empty. The preprocessing functions have been exercised against representative GOES-16 NetCDF4 files from the acquired corpus, confirming correct UTC index construction and background/noise column production.

*Section 10*

## Deliverables Summary

| DELIVERABLE | PATH |
| --- | --- |
| GOES automated downloader | scripts/download_goes.py |
|   | data/raw/goes/*.nc |
| GOES-16 XRS corpus (1,462 files) |   |
| SoLEXS FITS reader | pipeline/readers.py → |
|   | read_solexs() |
| HEL1OS FITS reader | pipeline/readers.py → read_helios() |
| GOES NetCDF4 / TXT / CSV readers | pipeline/readers.py → read_goes_nc/txt/csv() |
|   | Preprocessing pipeline (SoLEXS, HEL1OS, GOES) pipeline/preprocessor.py |
| Algorithmic nowcasting detector | pipeline/detector.py |
| Multi-instrument catalog merge | pipeline/catalog.py |
| Feature engineering + window builder | model/features.py |
| Central configuration | config.yaml |
| Aditya-L1 FITS files (SoLEXS + HEL1OS) | data/raw/{solexs,helios}/ |
| Processed Parquet files (Aditya-L1) | data/processed/YYYYMMDD_*.parque |


## Section 11

## Next Steps — Milestone 4: Model Training

With the data pipeline complete and the GOES corpus in place, the project moves to the CNN-LSTM model training phase. The following sequence is planned:

## Pre-train CNN-LSTM on GOES corpus

Run python model/train.py --phase pretrain --goes-dir data/raw/goes --epochs

30 . The trainer self-labels flares via the nowcasting detector (no external catalog required), builds 60-min windows, and saves the best checkpoint by validation loss.

## Download and process Aditya-L1 FITS data

Retrieve SoLEXS and HEL1OS Level-1 FITS files from ISSDC PRADAN, run through

process_and_save_solexs() and process_and_save_helios() , confirm HDU column names against notebooks/01_eda.ipynb .

## Fine-tune on Aditya-L1 data

Run python model/train.py --phase finetune --sxr-dir data/processed --epochs

10 . Convolutional layers are frozen; LSTM and head layers are re-trained at lr = 1×10⁻⁴ to adapt to instrument-specific calibration.

## Evaluate and report metrics

Run model/evaluate.py to produce TPR, FAR, AUC-ROC, and lead-time statistics across thresholds. Target: TPR ≥ 0.80, FAR ≤ 0.20, median lead time ≥ 20 min for M- class and above.

## 5 Validate dashboard end-to-end

Launch streamlit run dashboard/app.py against a processed Aditya-L1 day to

confirm the nowcasting alerts, ML probability curve, and master catalog table

render correctly in Historical mode.
