## PRODUCT REQU I REMENTS DOCUMENT

## SolarWatch

## Nowcasting & Forecasting of Solar Flares from Aditya‑L1 Soft and Hard X‑ray Data

An automated pipeline that fuses SoLEXS (soft X‑ray) and HEL1OS (hard X‑ray) light curves from ISRO's Aditya‑L1 to (a) detect and classify flares in real time into a master catalogue and (b) forecast the probability of an imminent flare with a quantifiable lead time — served through a live streaming dashboard with visual alerts.

Author

Milind (B.Tech, Artificial Intelligence & Robotics)

Institution

Madhav Institute of Technology & Science (MITS), Gwalior

Context

5th Semester Cornerstone Project · Solo

Problem Statement 15 — Combined Soft & Hard X‑ray Flare Forecasting/Now‐ casting

Reference problem

Document version

1.0 · August 2026

Status

Draft for review


## 0. Contents


## 1. Executive Summary

SolarWatch is an end-to-end system that turns the two X-ray eyes of Aditya‑L1 into an operational solar-flare monitor. Solar flares are sudden releases of magnetic energy that drive space-weather disruptions to satellites, GPS/GNSS, HF radio, aviation and power grids. Anticipating and cataloguing them is a prerequisite for protecting that infrastructure.

The system does two things that the problem statement weights equally:

- Nowcasting — a deterministic, interpretable pipeline detects flares independently in the soft X-ray (SoLEXS) and hard X-ray (HEL1OS) light curves, classifies each on the standard GOES A/B/C/M/X scale, and fuses the two into a single master flare catalogue with onset/peak/end times and a soft–hard association tag.

- Forecasting — a supervised time-series model reads a sliding window of the combined soft+hard signal and outputs the probability that a flare will begin within the next N minutes, producing a quantifiable lead time.

Both feed a real-time streaming dashboard that replays historical Aditya‑L1 data as a simulated live feed, plots the dual light curves, and raises visual alerts the moment a flare is nowcast or a forecast threshold is crossed.

## HEADLINE RECOMMENDATIONS (DECIDED IN THIS PRD)

Nowcasting is built as classical signal processing (robust, explainable, no training data needed). Forecasting is framed as sliding-window classification with a Temporal Convolutional Network (TCN) as the primary model, benchmarked against an LSTM and simple baselines. Data is resampled to a common 1-minute cadence; the primary forecast horizon is 30 minutes (reported also at 15 and 60 min); the headline skill metric is the True Skill Statistic (TSS). GOES/XRS is used as a supplementary dataset for labelling and cross-calibration only.

Scope is deliberately sized for a solo builder over one semester. Section 16 defines an explicit MVP cut line so that a working, demonstrable system exists well before the deadline, with stretch goals layered on top.

## 2. Background & Motivation

## 2.1 Solar flares and space weather

A solar flare is a rapid brightening on the Sun caused by magnetic reconnection releasing stored energy. The impulsive phase accelerates electrons that emit hard X-rays (HXR) as they slam into the dense chromosphere; that energy then heats coronal plasma to tens of millions of kelvin, which radiates soft X-rays (SXR) during a slower gradual phase. The consequences at Earth — sudden ionospheric disturbances, satellite drag and single-event upsets, GNSS positioning errors, HF radio blackout, and, for the largest events, geomagnetically induced currents in power grids — make early detection and forecasting operationally valuable.

## 2.2 Why Aditya‑L1, and why combine soft + hard X-rays

From the Sun–Earth L1 point (~1.5 million km from Earth) Aditya‑L1 has an uninterrupted view of the Sun — no orbital night, no South Atlantic Anomaly gaps that limit low-Earth-orbit monitors. It carries two full-disk ("Sun- as-a-star") X-ray spectrometers whose combination is the core of this project:

- SoLEXS captures the thermal SXR emission — the canonical flare light curve used for GOES-style classific‐ ation.


- HEL1OS captures the non-thermal HXR emission — the signature of the impulsive phase and particle acceleration, which the SXR channel alone cannot see.

Using both channels together is scientifically richer than either alone: the hard channel time-tags the energy- release moment, while the soft channel measures the thermal response and sets the flare class.

## 2.3 The scientific hook — the Neupert effect

## WHY THE COMBINATION HELPS FORECASTING

Empirically, the hard X-ray flux tracks the time derivative of the soft X-ray flux (the Neupert effect): HXR bursts during the impulsive rise, and the SXR peaks later, during decay. The practical consequence is that hard-X-ray activity systematically leads the soft-X-ray peak. That temporal ordering — plus pre-flare gradual rises and micro-flare "flickering" — is exactly the precursor structure a forecasting model can exploit, and is the reason a combined soft+hard model is expected to beat a soft-only one.

## 3. Problem Statement

Build an automated algorithmic pipeline that uses the combined time-series data from SoLEXS and HEL1OS to detect (nowcast) and predict (forecast) solar flares, and expose the results through a visual interface. Concretely, the system must:

- 1. Ingest and read SoLEXS (Level-1/Level-2) and HEL1OS (Level-1) time-series data from the ISSDC PRADAN portal.

- 2. Detect flares independently in soft and hard X-rays and fuse the two into a single automated master catalogue, classifying each event on the GOES A/B/C/M/X scale.

- 3. Forecast the probability of a flare occurring within the next N minutes, quantifying lead time before the flare peak.

- 4. Visualise the light curves and raise a visual alert whenever a flare is nowcast or forecast.

The central design tension is honest and worth stating up front: nowcasting is a tractable detection problem with a clear ground truth, whereas forecasting flares from disk-integrated X-rays alone is genuinely hard — the useful precursor information is limited and lead times are short. This PRD treats both as first-class but sets realistic, measurable targets for the forecasting side rather than over-promising (see §5 and §13).

## 4. Goals, Objectives & Scope

## 4.1 Primary objectives

| ID O1 O2 | Objective Success looks like Automated now‐ Master catalogue reproduces the GOES flare list with high recall on C-class and above. cast catalogue: de‐ tect & classify flares from combined soft + hard X-rays in real time. Automated forecast Positive TSS at a fixed operating point; measurable median lead time. model: predict P(flare within next N |
| --- | --- |


| O3 |   | min) with a quantifi‐ able lead time. Streaming dash‐ Live replay of historical data triggers correct, timely on-screen alerts. board that visual‐ ises dual light curves and raises |
| --- | --- | --- |
| O4 |   | nowcast/forecast alerts. Reproducibility: a A fresh clone reproduces the catalogue and model metrics from scripts. documented, re-run‐ |
|   |   | nable pipeline from raw PRADAN files to dashboard. |

## 4.2 Non-goals (explicitly out of scope for v1.0)

- Spectral fitting / plasma-parameter inversion (temperature, emission measure, abundances) — SolarWatch uses light curves, not full spectroscopy.

- Flare location or active-region identification (the payloads are full-disk integrated; no imaging).

- Magnetogram- or image-based forecasting (SDO/HMI, SUIT) — noted as future work in §20.

- A hardened 24×7 production service, user accounts, or alerting to external systems (email/SMS). The "real- time" feed is a faithful simulation that replays archived data.

- Downstream impact modelling (ionospheric/grid effects).

## MVP CUT LINE (SEE §16 FOR DETAIL)

Must-have MVP: soft+hard nowcast master catalogue validated against GOES + a streaming dashboard replay with working alerts + a baseline forecaster with reported TSS. Stretch: tuned TCN forecaster, multi-horizon lead-time analysis, Neupert-based precursor features, probability calibration.

## 5. Success Metrics & Evaluation Criteria

Metrics are grouped by subsystem. Flares are rare events, so accuracy is deliberately avoided as a headline number — it is trivially high by always predicting "no flare." Imbalance-robust skill scores are used instead.

## 5.1 Nowcasting metrics (vs. GOES event list as ground truth)

| Metric | Definition | Target (guidance) |
| --- | --- | --- |
| Detection recall | Fraction of true GOES flares the catalogue | ≥ 0.9 for C+; report separately for B, M, X. |
| (POD) | recovers, per class band. |   |
| Catalogue preci‐ | Fraction of detected events that are real flares | ≥ 0.85 overall. |
| sion | (not noise/spikes). |   |
| Classification ac‐ | Agreement of assigned GOES class vs. reference | ≥ 0.8 within one sub-class. |
| curacy | class (exact & ±1 sub-class). |   |
| Timing error | Median absolute error of onset/peak times vs. ref‐ | Report; aim ≤ a few minutes at 1-min cadence. |
|   | erence. |   |
| Soft–hard | Fraction of M/X flares with a correctly matched | Report; expected high for large flares. |
| association rate | HEL1OS counterpart. |   |


## 5.2 Forecasting metrics

|   | Metric | Definition / why it matters |
| --- | --- | --- |
| Rate | Score) | TSS (True Skill Stat‐ POD − POFD (= TPR − FPR). Robust to class imbalance; standard in flare-forecasting istic) — headline literature. Range −1..1; 0 = no skill. TPR / True Positive Probability of Detection — flares correctly warned. Directly named in the challenge criteria. FAR / False Alarm Rate False alarms as a share of warnings; the operational cost of crying wolf. Directly named in the challenge criteria. HSS (Heidke Skill Skill over random/climatology baseline; complements TSS under heavy imbalance. BSS (Brier Skill Score) Quality of the probabilities (calibration + sharpness), since the model outputs a probability. Precision / Recall / F1 Threshold-independent view under imbalance (PR-AUC preferred over ROC-AUC here). |
|   |   | & PR-AUC Lead time Minutes between the alert crossing threshold and the flare peak (and separately, the flare onset). Report the full distribution — median and IQR — not a single number. |

## OPERATING POINT

Because TPR and FAR trade off against each other, the model is tuned to a chosen operating point on the validation set (e.g. maximise TSS, or fix FAR ≤ a budget and maximise TPR), and that point is frozen before touching the test set. The dashboard alert threshold is this operating point.

## 5.3 System / product metrics

- Streaming latency: end-to-end lag from a replayed sample to its appearance + any alert on the dashboard (target < 1 s at demo replay speeds).

- Throughput: replay can run at ≥ 60× real time without dropping samples or stalling the UI.

- Reproducibility: catalogue & metrics regenerate from scripts on a clean checkout.

## 6. Users, Personas & Use Cases

| Persona | Needs | How SolarWatch serves them |
| --- | --- | --- |
| Space-weather analyst (primary) | Timely, trustworthy flare alerts | Live dashboard + queryable master |
|   | and a searchable event history. | catalogue with class, timing and soft/ |
|   |   | hard tags. |
| Solar physics researcher | A clean, combined soft+hard flare | Exportable catalogue (CSV/JSON) with |
|   | catalogue for downstream stud‐ | provenance and detection parameters. |
|   | ies. |   |
| Satellite / GNSS operator (downstream) Advance warning to safe-mode |   | Forecast probability + lead time sur‐ |
|   | assets or flag positioning risk. | faced as a graded alert. |
| Evaluator / faculty | Evidence the system meets the | Reproducible pipeline, metrics report, |
|   | stated objectives and metrics. | and a live demo. |


## Representative use cases

- UC-1 (nowcast): An M-class flare begins; the dashboard flags the rise, classifies it, associates the HEL1OS burst, and writes the event to the catalogue — all within the replay.

- UC-2 (forecast): Minutes before onset, precursor structure pushes the forecast probability over threshold; a "flare likely within 30 min" warning appears; lead time is logged when the flare later peaks.

- UC-3 (research): A user filters the catalogue for all C+ flares in a date range with a hard-X-ray counterpart and exports them.

- UC-4 (review): An evaluator replays a known active week and checks alerts against the GOES record.

## 7. Domain Background (The Science)

## 7.1 The two payloads at a glance

| Property | SoLEXS (soft X-ray) | HEL1OS (hard X-ray) |
| --- | --- | --- |
| Full name | Solar Low Energy X-ray Spectrometer | High Energy L1 Orbiting X-ray |
|   |   | Spectrometer |
| Energy range | ~1–30 keV (design); ~2–22 keV effective | ~8–150 keV (CdTe 8–70, CZT 20– |
|   |   | 150) |
| Emission probed Thermal (hot coronal plasma) |   | Non-thermal (accelerated electrons) |
| Detectors | Two SDDs, different apertures (large aperture → small flares; | CdTe + CZT compound-semicon‐ |
|   | small aperture → large flares) | ductor detectors |
| Cadence | Spectra at ~1 s during flares; onboard flare-start detection | Event list at 10 ms timing resolution |
| Flare sensitivity A-class to X-class |   | Down to ~B-class; shines on M/X |
|   |   | impulsive bursts |
| L1 products | Light curves / spectra (Level-1, Level-2) | Event list + spectral data (Level-1) |
| Role here | Sets the GOES class; primary detection channel | Time-tags energy release; precursor |
|   |   | + association channel |

## 7.2 GOES flare classification

Flare class is defined by peak flux in the GOES 1–8 Å soft X-ray band. SolarWatch adopts this scale for O1; because the SoLEXS bandpass differs from GOES/XRS, a cross-calibration mapping from SoLEXS flux to GOES-equivalent flux is derived from overlapping flares (see §12).

| Class | Peak flux (W m−2, 1–8 Å) | Typical impact |
| --- | --- | --- |
| A | < 10−7 | Background level; negligible. |
| B | 10−7 – 10−6 | Minor; usually no Earth effect. |
| C | 10−6 – 10−5 | Small; minor ionospheric effects. |
| M | 10−5 – 10−4 | Moderate; radio blackouts, radiation risk to polar flights. |
| X | ≥ 10−4 | Major; wide blackouts, satellite & grid risk. |

Within each letter a linear 1–9 sub-scale applies (e.g. M2.5); X may exceed 9 (e.g. X10). "Low-class" in the challenge criteria maps to A/B/C; "high-class" to M/X.


## 7.3 Anatomy of a flare light curve

- Pre-flare / precursor: gradual soft rise, occasional micro-flare flickering — the regime the forecaster must learn.

- Impulsive phase: sharp HXR burst (HEL1OS); soft flux rising fastest here (Neupert).

- Peak: soft flux maximum (SoLEXS) — sets the GOES class; HXR has usually already peaked.

- Decay / gradual phase: slow soft-flux decline back to background.

## DESIGN IMPLICATION

The hard channel leads, the soft channel classifies. Nowcasting keys the class off SoLEXS but uses HEL1OS to confirm and time the impulsive phase; forecasting leans on HEL1OS activity and the SoLEXS derivative as leading indicators.

## 8. Data Requirements & Access

## 8.1 Datasets

| Dataset | Role | Source |
| --- | --- | --- |
| SoLEXS L1/L2 | Primary soft X-ray light curve; detection + classification. | ISSDC PRADAN |
| HEL1OS L1 | Hard X-ray event list & light curve; impulsive-phase + precurs‐ | ISSDC PRADAN |
|   | ors. |   |
| GOES/XRS (supplementary) | Independent flare event list for labels + soft-flux cross-calib‐ | NOAA/SWPC (open) |
|   | ration. Not an input to the deployed model. |   |
| Vendor tools | SoLEXS analysis package ("SolexS" tools) & HEL1OS Python | PRADAN (per-payload) |
|   | utilities for reading/timing/spectra. |   |

## DATA-AVAILABILITY CAVEAT

Publicly released Aditya‑L1 science data covers roughly July 2024 onward; the performance-verification phase (Jan–Jun 2024) is available on request to the payload team. Plan the training window around the public range, and treat any gaps or instrument-off periods as first-class citizens in preprocessing.

## 8.2 Step-by-step: getting the data from PRADAN

- 1. Register / log in at the ISSDC PRADAN portal ( pradan.issdc.gov.in ; the Aditya‑L1 archive is under the AL1 path). Registration is required so ISRO can track usage.

- 2. Open the Aditya‑L1 collection and select the SoLEXS and HEL1OS payloads. Each payload page links its user manual, data-format document, and instrument paper — read these first; they define the file structure and columns.

- 3. Search & browse by date range. Prefer active-Sun intervals (more flares → more positive labels). Cross- reference the NOAA/GOES flare list to pick weeks rich in C/M/X events.

- 4. Preview then download the relevant Level-1 (and Level-2 for SoLEXS, if offered) products. Note the file format (FITS-style science products with light-curve / event-list extensions and housekeeping).

- 5. Download the payload utility tools from the same pages — the SoLEXS analysis package and the HEL1OS Python timing/spectral utilities — to parse the native products correctly.

- 6. Pull the matching GOES/XRS light curves and event list for the same dates (open NOAA/SWPC data) for labelling and cross-calibration.


- 7. Organise on disk as data/raw/solexs/… , data/raw/hel1os/… , data/raw/goes/… , and record exactly which product IDs/date ranges were pulled (a MANIFEST ) for reproducibility.

Exact menu labels, product levels, and file extensions can change; always defer to the on-portal user manuals rather than hard- coding assumptions. Build the reader against the documented format, not a guessed one.

## 8.3 Preprocessing pipeline

- 1. Parse SoLEXS light curves and HEL1OS event lists with the vendor tools / Astropy; extract time + flux (or counts) per band.

- 2. Bin HEL1OS from 10 ms events into a light curve; resample both payloads onto a common 1-minute grid (configurable), aligned to UTC.

- 3. Clean: flag/handle data gaps, instrument-off intervals, and PV-phase exclusions; remove obvious non-solar spikes (e.g. particle hits) via robust filtering.

- 4. Background & normalise: estimate a rolling quiescent background; produce both raw flux and background- subtracted series; log-scale soft flux (spans many decades).

- 5. Cross-calibrate SoLEXS soft flux to GOES-equivalent 1–8 Å flux using overlapping flares (regression), enabling GOES-class assignment.

- 6. Label: tag each timestamp using the GOES event list (flare / no-flare, class, onset/peak/end) for both catalogue validation and forecast supervision.

- 7. Persist tidy time-series as Parquet and the labelled flare list as CSV/JSON; everything downstream reads these.

## 9. System Architecture

SolarWatch is a linear data pipeline (offline) plus a serving layer (online replay). The same nowcast/forecast code runs in both batch (building the catalogue, training) and streaming (dashboard) modes.


## Modules

- Ingest & Readers — turn native PRADAN products into tidy DataFrames.

- Preprocess — resample, clean, background, cross-calibrate, label (§8).

- Nowcast Engine — SXR & HXR detectors, classifier, fusion → master catalogue (§12).

- Forecast Engine — windowed dataset, TCN/LSTM training + inference (§13).

- Replay Engine — streams archived data as a simulated live feed at controllable speed.

- Serving API — FastAPI endpoints + WebSocket for streaming, catalogue and forecasts.

- Dashboard — React/Vite/TS UI with dual light curves, gauges, alerts, catalogue table (§14).

## 10. Functional Requirements

| ID FR-1 FR-2 FR-3 FR-4 FR-5 FR-6 FR-7 FR-8 FR-9 | Requirement Priority Read SoLEXS L1/L2 and HEL1OS L1 products and expose them as time-aligned light curves. Must Resample soft & hard series to a common configurable cadence (default 1 min) with gap hand‐ Must ling. Detect flares in the SXR channel (onset, peak, end) with configurable sensitivity. Must Detect bursts in the HXR channel independently. Must Assign a GOES A/B/C/M/X class (with sub-class) to each SXR flare via cross-calibration. Must Fuse soft & hard detections by temporal association into a single master catalogue. Must Persist the master catalogue with a defined schema; expose query/filter/export. Must Build a windowed, labelled dataset for forecasting from the combined series. Must Train & evaluate a forecast model outputting P(flare within N min); persist weights. Must FR-10 Serve current forecast probability + lead-time estimate over the API. Must FR-11 Replay archived data as a simulated live stream at controllable speed (play/pause/seek). Must FR-12 Stream dual light curves to the dashboard and render them live. Must |
| --- | --- |
|   | FR-13 Raise a visual alert on nowcast detection and on forecast threshold crossing, colour-graded by Must class/probability. FR-14 Show a live catalogue panel of recent events with class & soft/hard tags. Should FR-15 Overlay the SXR derivative / Neupert view and a precursor indicator. Could FR-16 Export catalogue & a metrics report (CSV/JSON/PDF). Should |

## 11. Non-Functional Requirements

| ID | Requirement |
| --- | --- |
| NFR-1 | Reproducibility: every artefact (catalogue, model, metrics) regenerates from scripts on a clean checkout |
|   | with a pinned environment and data manifest. |
| NFR-2 | No temporal leakage: forecasting uses strictly causal features and a chronological train/val/test split; the |
|   | pipeline must make future-peeking structurally impossible. |


- NFR-3 Explainability: nowcasting is deterministic and inspectable; detection parameters are logged per event.

- NFR-4 Performance: replay ≥ 60× real time; dashboard render + alert latency < 1 s at demo speed.

- NFR-5 Robustness: graceful handling of gaps, instrument-off periods and malformed records — no crashes, explicit "no-data" states.

- NFR-6 Configurability: cadence, window size, horizon, thresholds set via a single config, not hard-coded.

- NFR-7 Portability: runs locally; containerised; no dependency on a live network feed.

- NFR-8 Documentation: READMEs, data manual notes, and a final report sufficient for a third party to re-run.

## 12. Detailed Design — Nowcasting Engine

Nowcasting is intentionally algorithmic, not learned: it is robust, needs no training labels, and every decision is inspectable — the right choice for a catalogue that must be trusted.

## 12.1 Soft X-ray (SoLEXS) flare detector

Operates on the background-subtracted, log-scaled 1-min soft flux. A flare candidate is declared when the flux rises significantly above a rolling quiescent background and its short-term derivative is positive and sustained. For each candidate the detector estimates:

- Onset — where flux departs the background trend and the derivative turns sharply positive.

- Peak — local flux maximum (sets the class).

- End — return toward background (e.g. flux decays to a defined fraction above pre-flare level).

Techniques: rolling median/MAD background, adaptive thresholding, derivative (rate-of-rise) gating, and minimum-duration / prominence filters to reject spikes. Overlapping/complex events are handled by peak- prominence separation.

## 12.2 Hard X-ray (HEL1OS) burst detector

Operates on the binned hard-X-ray light curve. Because HXR is impulsive and only appreciable for larger flares, the detector emphasises transient bursts: significant excess over a local Poisson/robust background, short rise time, and coincidence with elevated activity. Outputs burst start/peak/end and a peak-energy indicator (CdTe vs CZT bands).

## 12.3 GOES-style classification & SoLEXS cross-calibration

SoLEXS is not GOES, so a mapping is derived: for flares observed by both SoLEXS and GOES in an overlap window, regress SoLEXS peak flux (in its band) against the GOES 1–8 Å peak flux. The fitted relation converts a SoLEXS peak into a GOES-equivalent flux, which is then bucketed into A/B/C/M/X + sub-class. The fit and its residuals are reported as a calibration artefact.

## 12.4 Catalogue fusion (soft ↔ hard association)

Independent soft and hard detections are merged by temporal association: a hard burst is matched to a soft flare when their intervals overlap (allowing for the Neupert lead of HXR before the SXR peak). Each merged event records whether it was seen in soft only, hard only, or both, plus a Neupert-consistency indicator (HXR vs d/dt SXR).


## 12.5 Master catalogue schema

| Field |   | Description event_id Unique ID (e.g. date-sequence). sxr_start / peak / end Soft X-ray onset, peak, end times (UTC). goes_class , peak_flux Assigned class (e.g. M1.4) and GOES-equivalent peak flux. |
| --- | --- | --- |
|   |   | hxr_detected Boolean — HEL1OS counterpart present. hxr_start / peak , hxr_peak_band Hard-burst timing and dominant energy band (CdTe/CZT). neupert_corr Correlation of HXR with d/dt(SXR) — impulsiveness / consistency. duration , source , confidence Event duration, detection source (soft/hard/both), detector confidence. params Detection parameters used (for reproducibility). |

## 13. Detailed Design — Forecasting Engine

## 13.1 Problem framing

Forecasting is posed as causal sliding-window classification. At time t, the model sees the last W minutes of the combined feature series and predicts the probability that a flare (≥ chosen class) will begin within the next N minutes (the horizon). This yields a probability at every step and a natural lead time: the gap between the first threshold-crossing and the flare peak.

## DECIDED DEFAULTS (TUNABLE)

Cadence = 1 min; look-back window W = 120 min (120 steps); primary horizon N = 30 min, also reported at 15 & 60 min; positive threshold class = ≥ C (with a ≥M variant reported). Rationale: X-ray precursors are short-lived, so a modest horizon with a 2-hour context is a realistic, testable target.

## 13.2 Features (all strictly causal)

- Soft flux (log) and its 1st/2nd derivatives; short- & long-window rolling mean/variance (rise + "flickering").

- Hard-X-ray flux/counts and its derivatives; hardness proxy (CZT-band vs CdTe-band ratio).

- Neupert feature: running correlation of HXR with d/dt(SXR).

- Time-since-last-flare and recent flare-rate features from the catalogue.

## 13.3 Models (recommended stack)

| Model Role Why |
| --- |
| Persistence / climatology Baseline Sanity floor — any model must beat "assume |
| tomorrow like today." |
| Derivative threshold rule Baseline Physics-motivated: alert when soft rise-rate / |
| HXR exceeds a level. Logistic regression / GBDT on hand features Interpretable Strong, explainable reference; feature- baseline importance insight. TCN (Temporal Convolutional Network) — primary Main model |


|   | Dilated causal convolutions capture long look- backs efficiently; parallel, stable training; no future leakage by construction. |
| --- | --- |
|   | LSTM (+ optional attention) Comparison Sequential recurrent benchmark; attention adds interpretability over the window. |

## WHY TCN AS PRIMARY

For fixed-horizon forecasting on evenly-sampled series, TCNs match or beat LSTMs while training faster and — crucially — their causal convolutions make it structurally impossible to peek at the future. The architecture also aligns with the author's prior temporal-forecasting work, lowering execution risk. The LSTM is kept as a rigorous comparison, not an afterthought.

## 13.4 Training methodology

- 1. Chronological split (e.g. earliest 60% train / next 20% val / final 20% test) — never random — to prevent leakage across the time axis.

- 2. Class imbalance: flare-positive windows are rare → class weights / focal loss, and threshold selection on validation (not accuracy).

- 3. Calibration: post-hoc probability calibration (Platt/isotonic) so the dashboard probability is meaningful; assessed via reliability diagram + BSS.

- 4. Operating point: pick the alert threshold on validation (maximise TSS or fix FAR budget), then freeze before the test set.

- 5. Evaluation: TSS (headline), HSS, TPR, FAR, PR-AUC, BSS, and the lead-time distribution (median + IQR) at each horizon; report ≥C and ≥M separately.

## HONESTY ABOUT EXPECTED SKILL

Forecasting flares from disk-integrated X-rays alone is a hard, actively-researched problem; expect modest positive skill and short lead times, not near-perfect prediction. Success is a model that beats the baselines with a positive TSS and a measurable lead time — reported honestly — and a clear demonstration that the soft+hard combination outperforms soft-only (the key scientific claim). This ablation is a required result.

## 14. Interface & Dashboard Specification

## 14.1 Real-time streaming via replay

There is no public live Aditya‑L1 feed, so "real time" is a faithful simulation: the Replay Engine streams archived, preprocessed data in timestamp order over a WebSocket at a controllable speed (1× … 300×). Each tick pushes the newest soft/hard samples, the current nowcast state, and the current forecast probability. The same detection/inference code runs on the stream as on the batch data, guaranteeing consistency.

## 14.2 Panels

- Dual light-curve plot — soft (log-y) & hard curves scrolling together; flare spans shaded; class-coloured.

- Flare-state indicator — current condition (Quiet / Rising / Flaring) + live GOES-class readout for the active event.

- Forecast meter — probability gauge for "flare within N min," horizon selector, and a lead-time countdown/ estimate.

- Alert banner — colour-graded (green/amber/red by class or probability); fires on nowcast detection and on forecast threshold crossing; shows time, type, class/probability.


- Live catalogue table — most recent events with class, soft/hard tags, timing; filter + export.

- Replay controls — play/pause, speed, seek/scrub, jump-to-date.

- Neupert / derivative overlay (optional) — d/dt(SXR) vs HXR to visualise the precursor relationship.

## ALERT SEMANTICS

Two distinct alert types, never conflated: NOWCAST ("flare detected now," from the deterministic engine) and FORECAST ("flare likely within N min," from the model, with probability + lead time). Both are logged so alerts can be replayed and audited against the GOES record.

## 15. Technology Stack

| Layer Choice Language / data Python · NumPy · Pandas · Astropy (FITS) · SunPy · SciPy · vendor SoLEXS/HEL1OS tools ML PyTorch (TCN, LSTM) · scikit-learn (baselines, metrics, calibration) · matplotlib |
| --- |
| Storage Parquet (time series) · SQLite/DuckDB (catalogue) · JSON/CSV exports Backend FastAPI · Uvicorn · WebSockets · Pydantic Frontend React + Vite + TypeScript · fast streaming charts (uPlot / Plotly / Recharts) · Tailwind CSS Infra / tooling Docker · Git/GitHub · a config file (YAML) · reproducible env (requirements/lock) |

This mirrors the author's existing aditya-l1-flare-pipeline (FastAPI + React/Vite/TS), so the serving layer starts from familiar ground.

## 16. Phased Implementation Roadmap

Sized for a solo builder across a ~16-week semester. Each phase has a clear exit criterion — do not advance until it is met. The MVP (Phases 0–5, core paths) is prioritised so a working demo exists before polish.

Weeks 1–2

Objectives: access real data; set up the project skeleton.

Tasks: register on PRADAN; read SoLEXS & HEL1OS user manuals + instrument papers; download SoLEXS L1/L2 and HEL1OS L1 for active-Sun weeks; pull matching GOES data; grab vendor tools; init repo, environment, config, and a data MANIFEST .

Deliverables: raw data on disk; reader stubs; repo scaffold.

Exit: both payloads' files parse into a DataFrame; GOES labels available for the same dates.

## Phase 0 — Foundations & Data Acquisition


## Phase 1 — Data Pipeline & EDA

Objectives: a clean, labelled, unified time series.

Tasks: finalise readers; bin HEL1OS; resample both to 1-min; clean gaps/PV-phase/spikes; background + normalise; cross-match GOES event list; EDA of real soft & hard flare morphology (visualise a known M/X event across both channels).

Deliverables: Parquet time series + labelled flare list; an EDA notebook confirming the Neupert lead in real data.

Exit: one command produces the clean, labelled dataset; flare morphology understood.

Weeks 3–4

## Phase 2 — Nowcasting Engine (O1)

Objectives: the validated master catalogue.

Tasks: SXR detector (onset/peak/end); HXR burst detector; SoLEXS→GOES cross-calibration; A–X classification; soft↔hard fusion; write catalogue; validate recall/precision/timing vs GOES.

Deliverables: master catalogue + a nowcasting metrics report.

Exit: catalogue recovers GOES C+ flares at target recall with measured precision.

Weeks 5–7

## Phase 3 — Forecasting Model (O2)

Objectives: a trained, honestly-evaluated forecaster.

Tasks: windowed dataset builder (causal); baselines; TCN + LSTM; chronological split; imbalance handling; tuning; calibration; evaluate TSS/HSS/TPR/FAR/BSS + lead-time; run the soft+hard vs soft-only ablation.

Deliverables: trained weights; metrics report; ablation result.

Exit: primary model beats baselines with positive TSS and a reported lead-time distribution.

Weeks 8–10

## Phase 4 — Backend & Serving (O3a)

Objectives: the API + streaming layer.

Tasks: FastAPI catalogue endpoints; model inference endpoint; Replay Engine (history→live); WebSocket /stream emitting samples + nowcast + forecast; replay control endpoint.

Deliverables: running API serving catalogue & live forecast over WS.

Exit: a client can subscribe and receive a correct streamed flare sequence with alerts.

Weeks 11–12

## Phase 5 — Dashboard & Real-time Simulation (O3b)

Objectives: the end-to-end live demo.

Tasks: React/Vite/TS UI; dual streaming light curves; flare-state + forecast gauges; graded alert banner; live catalogue table; replay controls; wire to WebSocket.

Deliverables: working dashboard replaying archived data with live nowcast + forecast alerts.

Exit: replaying a known active week produces correct, timely NOWCAST and FORECAST alerts on screen.

Weeks 12–13


## Phase 6 — Evaluation, Hardening & Documentation (O4)

Objectives: rigour, reproducibility, and the deliverables.

Tasks: full-system evaluation on the held-out test period; error analysis (misses/false alarms); optional Neupert/ precursor extras; final report/paper; demo video; container + README; reproducibility pass on a clean checkout.

Deliverables: final report, reproducible repo, demo video, metrics appendix.

Exit: a third party reproduces the catalogue & metrics and runs the demo from the README.

| Milestone | Week | Gate |
| --- | --- | --- |
| M1 — Data ready | 4 | Clean labelled dataset + EDA. |
| M2 — Nowcast catalogue (MVP core) | 7 | Validated master catalogue. |
| M3 — Forecaster | 10 | Model beats baselines, TSS > 0. |
| M4 — Live demo (MVP complete) | 13 | Dashboard replay with alerts. |
| M5 — Final | 16 | Report + reproducible repo + demo. |

## 17. Risks & Mitigations

| Risk | Mitigation Data access / download friction or quota on PRADAN. Start Phase 0 immediately; download active-Sun weeks early; keep a local manifest; fall back to GOES-only prototyping for plumbing while data arrives. Sparse hard-X-ray signal (HEL1OS mainly triggers on larger flares). Make SXR the primary detection channel; treat HXR as confirmation/precursor; report soft-only vs soft+hard honestly. SoLEXS↔GOES cross-calibration uncertainty. Fit on many overlapping flares; report residuals; allow ±1 sub-class tolerance in the classification metric. Severe class imbalance in forecasting. Class weights/focal loss; TSS/HSS/PR-AUC over accuracy; careful threshold selection on valida‐ |
| --- | --- |
|   | tion. Low forecast skill / short lead time. Set modest targets; success = beat baselines with positive TSS; frame lead time as a distribu‐ tion; document limits. Temporal leakage inflating results. Causal features + chronological split enforced structurally; leakage check in tests. Browser streaming performance. Use a fast canvas charting lib (uPlot); cap on- screen points; downsample for display. |
|   | Solo scope / time pressure. Explicit MVP cut line; phase exit gates; stretch items clearly optional. |


## 18. Validation & Testing Plan

- Unit tests: readers (known file → expected frame), resampling/gap handling, detector on synthetic flares, windowizer (no future leakage), classification buckets.

- Nowcast validation: catalogue vs GOES event list over a held-out period — recall/precision/timing per class band; manual review of a sample of matches/misses.

- Forecast validation: metrics on the chronological test set only; reliability diagram; lead-time distribution; the required soft+hard vs soft-only ablation.

- Leakage audit: confirm no feature at time t uses data after t; shuffle/placebo test.

- System test: replay a known active week end-to-end; confirm alert timing/type against the GOES record; measure latency/throughput.

- Reproducibility test: clean checkout regenerates catalogue & metrics within tolerance.

## 19. Deliverables & Cornerstone Assessment Mapping

| Deliverable | Maps to |
| --- | --- |
| Reproducible data pipeline (PRADAN → clean labelled dataset) | O4, FR-1/2, Phase 1 |
| Automated master flare catalogue (soft+hard, GOES-classified) | O1, FR-3–7, Phase 2 |
| Trained forecast model + metrics + ablation | O2, FR-8–10, Phase 3 |
| FastAPI backend + WebSocket streaming | O3, FR-10–13, Phase 4 |
| React streaming dashboard with alerts | O3, FR-11–16, Phase 5 |
| Final report / paper, demo video, container | O4, Phase 6 |

These map directly onto the challenge's expected outcomes (nowcast database, forecast model with lead time, visualising interface with alerts) and evaluation criteria (low/high-class detection, high TPR & low FAR, quantified lead time).

## 20. Future Work

- Magnetogram-based forecasting — fuse active-region magnetic features (SDO/HMI) or SUIT imaging for longer lead times; the recognised path to real skill beyond X-ray-only.

- Spectral products — temperature / emission-measure time series from SoLEXS+HEL1OS spectra as richer features.

- Probabilistic forecasting — full predictive distributions / conformal prediction for calibrated uncertainty.

- Live ingestion — swap the replay for a near-real-time PRADAN pull when operationally available.

- External alerting — webhook/email/SMS notifications and an operator API.

- Multi-mission fusion — cross-check with GOES/other monitors for redundancy.

## 21. Glossary

| Term | Meaning SoLEXS Solar Low Energy X-ray Spectrometer — soft X-ray payload. |
| --- | --- |


| HEL1OS High Energy L1 Orbiting X-ray Spectrometer — hard X-ray payload. |
| --- |
| SXR / HXR Soft / Hard X-ray. Nowcasting Real-time detection & classification of flares happening now. Forecasting Predicting the probability of a flare within a future horizon. Neupert effect HXR flux ≈ time-derivative of SXR flux; HXR leads the SXR peak. GOES class A/B/C/M/X flare scale from peak 1–8 Å soft X-ray flux. Lead time Minutes between an alert and the flare peak (or onset). TSS / HSS / BSS True Skill Statistic / Heidke / Brier Skill Score — imbalance-robust skill metrics. TPR / FAR True Positive Rate / False Alarm Rate. TCN Temporal Convolutional Network — dilated causal 1-D CNN for sequences. PRADAN ISSDC's Policy-based data Retrieval, Analytics, Dissemination And Notification system — the Aditya-L1 data portal. |

## 22. References

- 1. ISRO / ISSDC — PRADAN data dissemination portal for Aditya-L1. pradan.issdc.gov.in (registration required; per- payload user manuals & tools).

- 2. Aditya-L1 science-data release announcement, Astronomical Society of India / SolarNews (Jan 2025) — public data availability from July 2024 onward.

- 3. Aditya-L1 mission overview and X-ray payloads (SoLEXS & HEL1OS) — mission description paper (arXiv:2212.13046).

- 4. SoLEXS — ground calibration & in-flight performance (arXiv:2509.26292); energy range, resolution, dual-SDD design, "SolexS" analysis package.

- 5. HEL1OS — A Hard X-ray Spectrometer on Board Aditya-L1, Solar Physics (2025) / arXiv:2512.12679 — 8–150 keV, CdTe/ CZT, 10 ms event-list timing, GOES consistency.

- 6. Sankarasubramanian et al. (2017) — X-ray Spectrometers On-Board Aditya-L1 for Solar Flare Studies.

- 7. NOAA/SWPC — GOES/XRS flare classification scale and event lists (supplementary labelling & cross-calibration).

- 8. Neupert, W. M. (1968) — relationship between microwave/HXR and the time-derivative of SXR emission (the Neupert ef‐ fect).

Prepared as a cornerstone-project planning document. Instrument specifications are drawn from published Aditya-L1 payload literature and the ISSDC portal as of August 2026; confirm exact data levels, formats and portal procedures against the current on-portal user manuals before implementation.
