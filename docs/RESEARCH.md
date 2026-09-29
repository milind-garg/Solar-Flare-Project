# Research Dossier — ISRO BAH 2026 Problem Statement 15

Deep research on the problem statement, existing implementations, scientific state-of-the-art, and Aditya-L1 data specifics. Every claim is linked to its source.

> **Timeline reality check:** BAH 2026 registration opened 10 Jun 2026 and the Grand Finale ran 6–7 Aug 2026 — the hackathon has **concluded** ([ISRO Grand Finale page](https://www.isro.gov.in/Bharatiya_Antariksh_Hackathon_2026_Grand_Finale.html)). This dossier is therefore for building a portfolio-grade / extensible solution against the same well-defined problem, and for learning from the teams who competed.

---

## 1. The official problem statement

**PS-15 (verbatim):** *"Forecasting and/or Nowcasting of Solar Flares using combined Soft and Hard X-ray data from Aditya-L1."*
Source: [Hack2skill BAH 2026 event page](https://hack2skill.com/event/bah2026) (challenge list, #15).

### 1.1 What ISRO actually asked for (from the official explainer, Session 2)

The [PS Explainer Session 2 video](https://www.youtube.com/watch?v=FwPmc6iqbag) ([structured summary](https://youtubesummary.com/summary/FwPmc6iqbag)) is the authoritative spec. PS-15 requirements:

**Prediction targets:**
1. Solar flare **detection / nowcasting**.
2. Build a **unified flare catalog combining SoLEXS (soft) and HEL1OS (hard)**.
3. **Forecast probability** that a flare occurs in the next *N* minutes.
4. **Multi-class probabilities for flare intensity / energy class** (A/B/C/M/X).

**Fine structures ISRO explicitly wants captured:**
- **Pre-flare sudden increases** (early rise before the main soft-X-ray peak).
- **Quasi-periodic pulsations (QPPs) in hard X-rays.**

**Evaluation criteria named by ISRO:**
- Detection accuracy across **both low- and high-class** flares.
- **High true-positive rate, low false-positive rate.**
- Forecast with **15–30 minutes lead time**.

**Physical motivation given:** flares drive geomagnetic storms, power-grid damage, GPS disruption, satellite-comms effects. SoLEXS sees **thermal plasma heating (gradual enhancements)**; HEL1OS sees **non-thermal particle acceleration (impulsive signatures)** — the two are complementary, which is why ISRO wants them fused.

**Data source:** Aditya-L1 SoLEXS + HEL1OS **Level-1 data via the ISSDC PRADAN portal**.

**Logistics:** teams of 3–4 students, India only, 30-hour finale ([event page](https://hack2skill.com/event/bah2026)). Shortlisted-teams list is public ([ISRO PDF](https://www.isro.gov.in/media_isro/pdf/BAH_2026_Shortlisted_Teams_16072026_Publication.pdf)).

---

## 2. How **this project** maps to PS-15 (gap analysis)

This is the most actionable section. Comparing ISRO's ask to what's already in the repo:

| ISRO requirement | This project | Status |
|---|---|---|
| Nowcast / detect flares | `pipeline/detector.py` (σ-threshold + rise guard) | ✅ Done |
| Unified SoLEXS+HEL1OS catalog | `pipeline/catalog.py` `merge_catalogs` (±3 min match) | ✅ Done (HEL1OS pending data) |
| Forecast P(flare in next N min) | CNN-LSTM, `lead_time_min=30` | ✅ Done (AUC=0.818, TSS=0.361) |
| **Multi-class intensity probabilities (A/B/C/M/X)** | `class_head` Linear(128→5) in model; `n_classes: 5` in config | ✅ Done |
| High TPR / low FPR | `evaluate.py` reports TPR, FAR, TSS, AUC | ✅ Done |
| 15–30 min lead time | `lead_time_min=30`; `compute_lead_times` measures it (P90=27 min) | 🟡 3 min short of target |
| **Pre-flare sudden increases** | `dflux_1min/5min` derivatives capture rise | 🟡 Partial |
| **QPPs in hard X-rays** | `qpp_power` feature (feature #8): rolling FFT peak power in HEL1OS | 🟡 Partial (HEL1OS data absent) |
| Standard skill metrics (TSS/HSS) | TSS added to `evaluate.py`; HSS not yet added | 🟡 TSS done, HSS pending |

**Remaining gaps to close** (in priority order):
1. ~~**Add a multi-class head.**~~ ✅ Done — `class_head` Linear(128→5), `n_classes: 5`.
2. ~~**Add TSS to `evaluate.py`.**~~ ✅ Done — TSS computed. **Add HSS** — one more metric (H(+) skill score) still missing.
3. ~~**Add a QPP / periodicity feature**~~ ✅ Done — `qpp_power` feature (#8). Activates when HEL1OS data is present.
4. **Push P90 lead time from 27 → 30 min** — currently 3 min short of the ISRO target.
5. **Consider the official SoLEXS flare catalog as ground-truth labels** (§5.3) instead of only self-detected labels — cleaner supervision.

---

## 3. Existing implementations & prior work

### 3.1 Directly on Aditya-L1 data
- **`thecont1/aditya-L1-solar-explorer`** (GitHub, Sep 2023) — early exploratory repo studying SoLEXS light curves; notes SoLEXS covers A-class to X-class dynamic range. Paired with a **[Kaggle "ISRO Aditya-L1 SoLEXS LightCurve Data" dataset](https://www.kaggle.com/)**. Good reference for SoLEXS `.lc` parsing and EDA, not a forecasting model.
- **"Aditya-L1 Solar Shield: A Multi-Modal ML Architecture"** ([IJERT v15](https://www.ijert.org/aditya-l1-solar-shield-a-multi-modal-machine-learning-architecture-for-48-hour-coronal-mass-ejection-ijertv15is043432)) — a multi-payload "space-weather command centre." Note it targets **48-hour CME prognostics**, not flare nowcasting specifically, but it's the closest published Aditya-L1 multi-instrument ML effort and shows the "fuse all payloads + dashboard" framing judges respond to.
- BAH 2026 competing teams (e.g. "The Fab Four" [LinkedIn writeup](https://www.linkedin.com/posts/mohitha-s-302baa2a6_certificate-activity-7497571803712770048-03z6), and a public [ideation doc on Scribd](https://www.scribd.com/document/1060780502/Ideation-for-Isro-Hackathon)) — confirm the winning framing is: unified catalog + probabilistic forecast + live dashboard, which matches this repo's architecture.

### 3.2 Takeaway
No dominant open-source SoLEXS/HEL1OS *forecasting* baseline exists — the field is wide open. The standard, proven recipe (below) comes from the **GOES/SDO** literature, which is exactly why this project **pre-trains on GOES then fine-tunes on Aditya** — the right call given how little labeled Aditya data exists.

---

## 4. Scientific state-of-the-art

### 4.1 The two problem framings
1. **Active-region / image-based** (SDO/HMI magnetograms, SHARP parameters) → "will AR X produce an M/X flare in next 24 h." Dominant in literature; needs imagery this project doesn't use.
2. **Light-curve / time-series** (GOES XRS soft X-ray flux) → "flare in next N min from the flux history." **This is exactly PS-15's regime and this project's approach.**

### 4.2 Standard metrics — use TSS and HSS
Solar-flare forecasting is evaluated with **class-imbalance-robust skill scores**, not accuracy:
- **TSS (True Skill Statistic)** = TPR − FPR. The community standard; insensitive to class ratio.
- **HSS (Heidke Skill Score)** — skill over random/climatology baseline.
Nearly every SOTA paper reports `min(TSS, HSS)` or both (e.g. [Yu et al. 2026, *Sci. Reports*](https://www.nature.com/articles/s41598-026-66110-2); [Trital et al. 2026, *Pattern Recognition Letters*](https://www.sciencedirect.com/science/article/pii/S0167865526003260)). Representative numbers: TSS ≈ 0.44–0.72 depending on task/dataset ([Surya evaluation](https://essopenarchive.org/doi/abs/10.22541/essoar.15008581); [Carrascosa 2026](https://link.springer.com/chapter/10.1007/978-3-032-36806-5_6)). **Add TSS/HSS to this project to be legible to domain experts.**

### 4.3 Models — the field's trajectory
- **Time-series baselines → CNN/LSTM hybrids → transformers/foundation models.** See the survey [Riggi et al. 2025, *Astronomy & Computing*](https://www.sciencedirect.com/science/article/pii/S2213133725001155) ("from early CNN models to hybrid… high-cadence soft X-ray XRS time series… baseline estimators and CNN/LSTM networks") and the review [Huang et al. 2024, *Science China Earth Sci.*](https://link.springer.com/article/10.1007/s11430-023-1375-2).
- **LSTM is a proven, competitive choice** for XRS/parameter time series ([Deshmukh et al. 2023, *A&A*](https://www.aanda.org/articles/aa/abs/2023/06/aa45742-22/aa45742-22.html) — LSTM among the strongest; [Li et al. 2024](https://arxiv.org/abs/2410.18562) — BiLSTM-Attention on SHARP+HED).
- **Bigger isn't automatically better.** An [independent evaluation of NASA-IBM's 366M-param *Surya* foundation model](https://essopenarchive.org/doi/abs/10.22541/essoar.15008581) found **cheap baselines match it** on flare forecasting (TSS 0.436). This directly **validates this project's deliberately small CNN-LSTM** — the `cnn_lstm.py` comment ("swap to a Transformer only above ~500k samples") is aligned with the evidence.
- **Onboard / compressed models** are a live research theme ([N-slim, Frontiers 2026](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2026.1866227/full); [Carrascosa 2026](https://link.springer.com/chapter/10.1007/978-3-032-36806-5_6)) — relevant if "runs at L1 / low compute" is ever a judging point.
- **Emerging:** multimodal LLM approaches ([Jw-flare, arXiv 2025](https://arxiv.org/abs/2511.08970)) — bleeding edge, not needed here.

---

## 5. Aditya-L1 data specifics

### 5.1 The instruments
- **SoLEXS** — Solar Low Energy X-ray Spectrometer. Soft X-rays; **A-class to X-class dynamic range**; uses SDD detectors with continuous solar viewing. Full instrument/calibration paper: [Sarwade et al. 2025, arXiv](https://arxiv.org/) ("Solar Low Energy X-ray Spectrometer on board Aditya-L1"; ground + onboard calibration establishing quantitative spectroscopy). This is your reference for **counts→physical-flux calibration** (the `solexs_counts_to_wm2` placeholder in `config.yaml`).
- **HEL1OS** — High Energy L1 Orbiting X-ray Spectrometer. Hard X-rays; impulsive/non-thermal signatures; operations + data paper: [HEL1OS arXiv (2026)](https://arxiv.org/). This is where QPPs and pre-flare hard-X-ray bursts live.

### 5.2 Data formats & access — an important discrepancy to verify
- ISSDC PRADAN [FAQ](https://pradan1.issdc.gov.in) states Aditya payload data is **netCDF at L1 and L2**, accessed after registration.
- **But** the SoLEXS data on disk here is `.lc.gz` **FITS** light curves (`AL1_SLX_L1_*/SDD1,SDD2/*.lc.gz`) — a different product line than the netCDF the FAQ describes. **Action:** confirm which SoLEXS product you have vs. what PRADAN currently serves; the `read_solexs` reader is already written for the FITS `.lc.gz` form, so you're fine as long as that's the product you keep using.
- GOES reference data is netCDF (already handled by `read_goes_nc`).

### 5.3 There is an official SoLEXS flare catalog — use it
URSC/ISRO published a **SoLEXS light-curve product spanning Jan 2024 – May 2026 with 15,000+ automatically detected flares** (search result from [ursc.gov.in](https://www.ursc.gov.in/)). Your local SoLEXS data (Feb 2024 onward) overlaps this window. **Two high-value uses:**
1. **Ground-truth labels** for fine-tuning (better than self-detected labels).
2. **Validation** of your own detector's catalog against the official one (precision/recall of your nowcaster).
> Verify the exact URL/product on PRADAN/URSC before relying on it (the specific page 404'd during this research; the catalog is referenced but locate the live link).

---

## 6. Recommended next actions for this project

Ordered by impact-per-effort:

1. **Add TSS + HSS** to `model/evaluate.py` — trivial, and it's the language domain judges/reviewers speak.
2. **Add a multi-class flare-intensity head** (A/B/C/M/X) — directly closes an explicit ISRO requirement; labels already exist via the detector's GOES-equivalent classification.
3. **Get HEL1OS data in**, then add a **QPP/periodicity feature** (rolling FFT/wavelet peak power on the hard-X-ray channel) — closes the other explicit ISRO signature and is a genuine differentiator.
4. **Fetch the official SoLEXS flare catalog** from PRADAN/URSC and wire it in as label + validation source.
5. **Tune `solexs_counts_to_wm2`** using the Sarwade et al. calibration paper + overlapping GOES/SoLEXS flares (already flagged in `config.yaml`).
6. Keep the small CNN-LSTM — the Surya evaluation is direct evidence that heavier models don't buy much here.

---

## Source index
- Official PS list & logistics — [Hack2skill BAH 2026](https://hack2skill.com/event/bah2026)
- **PS-15 authoritative requirements** — [Explainer Session 2 summary](https://youtubesummary.com/summary/FwPmc6iqbag) · [video](https://www.youtube.com/watch?v=FwPmc6iqbag)
- Hackathon concluded — [ISRO Grand Finale](https://www.isro.gov.in/Bharatiya_Antariksh_Hackathon_2026_Grand_Finale.html) · [shortlist PDF](https://www.isro.gov.in/media_isro/pdf/BAH_2026_Shortlisted_Teams_16072026_Publication.pdf)
- SoLEXS instrument/calibration — [Sarwade et al. 2025 (arXiv/ADS)](http://ui.adsabs.harvard.edu)
- PRADAN/ISSDC data access & format — [PRADAN FAQ](https://pradan1.issdc.gov.in)
- Existing Aditya-L1 work — [aditya-L1-solar-explorer (GitHub)](https://github.com) · [Kaggle SoLEXS data](https://www.kaggle.com/) · [IJERT Solar Shield](https://www.ijert.org/aditya-l1-solar-shield-a-multi-modal-machine-learning-architecture-for-48-hour-coronal-mass-ejection-ijertv15is043432)
- SOTA & metrics — [Riggi et al. 2025](https://www.sciencedirect.com/science/article/pii/S2213133725001155) · [Deshmukh et al. 2023](https://www.aanda.org/articles/aa/abs/2023/06/aa45742-22/aa45742-22.html) · [Huang et al. 2024 review](https://link.springer.com/article/10.1007/s11430-023-1375-2) · [Surya independent eval](https://essopenarchive.org/doi/abs/10.22541/essoar.15008581) · [Li et al. 2024](https://arxiv.org/abs/2410.18562)
