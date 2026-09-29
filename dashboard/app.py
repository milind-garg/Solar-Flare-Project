"""
Solar Flare Nowcasting & Forecasting Dashboard — Dash by Plotly

Launch:
    python dashboard/app.py
    # Open http://127.0.0.1:8050
"""

from __future__ import annotations

import io
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

import dash
from dash import Dash, Input, Output, State, dcc, html, dash_table
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.catalog import load_master_catalog, merge_catalogs
from pipeline.detector import (
    detect_solexs_flares, detect_helios_flares,
    detect_goes_flares, events_to_dataframe,
)
from pipeline.preprocessor import preprocess_goes
from pipeline.readers import read_goes_nc
from model.cnn_lstm import load_checkpoint, load_norm
from model.features import build_feature_series, FEATURE_NAMES, FLARE_CLASS_ORDER

import torch


# ── Config & paths ────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)

CFG      = _load_cfg()
PROC_DIR = ROOT / CFG["paths"]["processed_data"]
RAW_DIR  = ROOT / CFG["paths"]["raw_data"]
CKPT     = ROOT / CFG["paths"]["model_checkpoint"]


# ── Lazy model cache ──────────────────────────────────────────────────────────

import threading as _threading
_MODEL_LOCK = _threading.Lock()
_MODEL = _NORM_MEAN = _NORM_STD = _CKPT_META = None

def _get_model():
    global _MODEL, _NORM_MEAN, _NORM_STD, _CKPT_META
    if _MODEL is not None:          # fast path — already loaded, no lock needed
        return _MODEL, _NORM_MEAN, _NORM_STD
    with _MODEL_LOCK:               # only one worker loads; others wait then skip
        if _MODEL is None and CKPT.exists():
            try:
                model  = load_checkpoint(str(CKPT), device="cpu", cfg=CFG)
                nmean, nstd = load_norm(str(CKPT), device="cpu")
                meta   = torch.load(str(CKPT), map_location="cpu", weights_only=False).get("meta", {})
                # Assign _MODEL last so the fast-path check above never sees a
                # partially initialised state (_NORM_MEAN etc. are always set first).
                _NORM_MEAN, _NORM_STD, _CKPT_META, _MODEL = nmean, nstd, meta, model
            except Exception as e:
                print(f"[dashboard] model load error: {e}")
                _CKPT_META = {}
    return _MODEL, _NORM_MEAN, _NORM_STD


# ── Design tokens ─────────────────────────────────────────────────────────────
# Aerospace Slate Design System (Grafana / Datadog Space / NASA telemetry aesthetic)

BG      = "#0b0f19"          # deep aerospace slate base
BG2     = "#0e1422"          # chart plot canvas & header base
CARD    = "#111827"          # clean charcoal slate card container
CARD2   = "#1f2937"          # elevated surface & hover state
BORDER  = "#1f2937"          # subtle 1px slate divider
BORDER2 = "#283548"          # lighter structural border
FG      = "#f9fafb"          # primary text — crisp off-white
FG2     = "#9ca3af"          # secondary text — neutral cool slate
DIM     = "#6b7280"          # muted captions & inactive labels
BLUE    = "#3b82f6"          # operational blue accent
CYAN    = "#38bdf8"          # SXR cyan highlight
AMBER   = "#f59e0b"          # solar thermal amber
ORANGE  = "#f97316"          # warning orange
VIOLET  = "#a855f7"          # HXR violet accent
GREEN   = "#10b981"          # normal / nominal green
RED     = "#ef4444"          # severe alert red

FLARE_COLORS = {
    "X": "#ef4444", "M": "#f59e0b", "C": "#eab308",
    "B": "#10b981", "A": "#38bdf8", "HXR": "#a855f7",
}

FONT_SERIF = "'Anthropic Serif', 'Tiempos Text', 'Copernicus', 'Newsreader', 'Lora', Georgia, serif"
FONT_MONO  = "'Anthropic Mono', 'DM Mono', monospace"

_GRAPH_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0e1422",
    font=dict(size=12, family=f"{FONT_SERIF}, {FONT_MONO}", color=FG2),
    margin=dict(l=62, r=20, t=44, b=44),
    legend=dict(
        orientation="h", y=1.14, font=dict(size=11, color=FG2, family=FONT_SERIF),
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
    ),
    xaxis=dict(
        showgrid=True, gridcolor="#1b2537", zeroline=False,
        tickfont=dict(size=11, color="#9ca3af", family=FONT_MONO), linecolor=BORDER2,
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#1b2537", zeroline=False,
        tickfont=dict(size=11, color="#9ca3af", family=FONT_MONO), linecolor=BORDER2,
    ),
    title_font=dict(size=13, color=FG, family=FONT_SERIF),
)


def _flare_color(cls: str) -> str:
    return FLARE_COLORS.get((cls or "A")[0], "#888888")


def _card(*children, **extra_style):
    style = {
        "background": CARD,
        "borderRadius": "10px",
        "padding": "18px 22px",
        "marginBottom": "14px",
        "border": f"1px solid {BORDER2}",
        "boxShadow": "0 4px 16px rgba(0,0,0,0.25)",
        **extra_style,
    }
    return html.Div(children, style=style)


def _kpi(label_text: str, value, color: str = BLUE, sub: str = ""):
    """Modern aerospace KPI metric tile with clean typography."""
    return html.Div([
        html.Div(
            label_text.upper(),
            style={"color": FG2, "fontSize": "10px", "letterSpacing": "0.08em",
                   "fontWeight": "600", "marginBottom": "6px",
                   "fontFamily": FONT_SERIF},
        ),
        html.Div(
            str(value),
            style={"fontSize": "1.75em", "fontWeight": "700", "color": color,
                   "lineHeight": "1.1", "fontFamily": FONT_MONO},
        ),
        html.Div(
            sub,
            style={"color": DIM, "fontSize": "11px", "fontWeight": "500",
                   "marginTop": "4px", "fontFamily": FONT_SERIF}
        ) if sub else html.Div(),
    ], style={
        "background": "#111827",
        "borderRadius": "8px",
        "padding": "14px 18px",
        "border": f"1px solid {BORDER2}",
        "display": "inline-block",
        "minWidth": "140px",
        "marginRight": "12px",
        "marginBottom": "12px",
        "verticalAlign": "top",
    })


# ── Plots ─────────────────────────────────────────────────────────────────────

def _fig_sxr(sxr_df: pd.DataFrame, events_df: pd.DataFrame | None, instrument: str = "solexs") -> go.Figure:
    fig = go.Figure()
    flux = sxr_df["flux_long"].resample("1min").mean().dropna()
    fig.add_trace(go.Scatter(x=flux.index, y=flux.values, name="SXR 1–8 Å (Long Channel)",
                             line=dict(color=BLUE, width=1.5)))
    if "bg_long" in sxr_df.columns:
        bg = sxr_df["bg_long"].resample("1min").mean().dropna()
        fig.add_trace(go.Scatter(x=bg.index, y=bg.values, name="Quiet-Sun Background (60m rolling 60th percentile)",
                                 line=dict(color=DIM, width=1, dash="dash")))

    # Scale factor for NOAA R-scale space weather threshold lines
    scale = CFG.get("calibration", {}).get("solexs_counts_to_wm2", 1.806e-08) if instrument == "solexs" else 1.0
    thresh_levels = [
        (1e-4 / scale, "X-Class · R3+ Strong Radio Blackout", "#ef4444"),
        (1e-5 / scale, "M-Class · R1-R2 Moderate Blackout", "#f59e0b"),
        (1e-6 / scale, "C-Class · Noticeable Coronal Activity", "#eab308"),
        (1e-7 / scale, "B-Class · Baseline Flare Activity", "#10b981"),
    ]
    flux_min = flux.min() if not flux.empty else 1.0
    for val, lbl, col in thresh_levels:
        if val >= flux_min * 0.4:
            fig.add_hline(
                y=val, line_color=col, line_dash="dash", line_width=1, opacity=0.55,
                annotation_text=lbl, annotation_position="top right",
                annotation_font=dict(size=9, color=col, family=FONT_SERIF),
            )

    # Detected events: shaded duration intervals and peak vertical markers with Neupert lag
    if events_df is not None and not events_df.empty:
        for _, ev in events_df.iterrows():
            cls = str(ev.get("flare_class") or "")
            pt  = ev.get("peak_time")
            st  = ev.get("start_time")
            et  = ev.get("end_time")
            lag = ev.get("neupert_lag_sec")
            col = _flare_color(cls)
            if pd.notnull(st) and pd.notnull(et):
                fig.add_vrect(
                    x0=st.isoformat(), x1=et.isoformat(),
                    fillcolor=col, opacity=0.10, layer="below", line_width=0,
                )
            if pd.notnull(pt):
                lbl = cls
                if pd.notnull(lag) and lag != "":
                    lbl += f" (lag: {float(lag):+.0f}s)"
                fig.add_vline(
                    x=pt.isoformat(), line_color=col, line_width=1.5, line_dash="dot",
                    annotation_text=lbl, annotation_position="top left",
                    annotation_font=dict(color=col, size=10, family=FONT_SERIF),
                )

    y_unit = "Counts · s⁻¹" if instrument == "solexs" else "Flux (W · m⁻²)"
    inst_label = "SoLEXS (Aditya-L1)" if instrument == "solexs" else "GOES-16 XRS"
    fig.update_layout(title=f"Soft X-Ray Light Curve (1–8 Å) — {inst_label}",
                      yaxis_title=y_unit, yaxis_type="log", height=280, **_GRAPH_LAYOUT)
    return fig


def _fig_hxr(hxr_df: pd.DataFrame, events_df: pd.DataFrame | None) -> go.Figure:
    fig = go.Figure()
    hxr = hxr_df["hxr_total"].resample("1min").mean().dropna()
    fig.add_trace(go.Scatter(x=hxr.index, y=hxr.values, name="HEL1OS Total HXR (10–160 keV)",
                             line=dict(color="#c084fc", width=1.5),
                             fill="tozeroy", fillcolor="rgba(192,132,252,0.06)"))
    if events_df is not None and not events_df.empty:
        for _, ev in events_df.iterrows():
            pt = ev.get("peak_time")
            if pd.isnull(pt):
                continue
            fig.add_vline(x=pt.isoformat(), line_color=FLARE_COLORS["HXR"],
                          line_width=1.5, line_dash="dot",
                          annotation_text="HXR Impulsive Burst", annotation_position="top left",
                          annotation_font=dict(color="#c084fc", size=10, family=FONT_SERIF))
    fig.update_layout(title="Hard X-Ray Impulsive Count Rate (HEL1OS 10–160 keV — Non-Thermal Plasma)",
                      yaxis_title="Counts · s⁻¹ (Livetime Corrected)", height=240, **_GRAPH_LAYOUT)
    return fig


def _fig_hardness(sxr_df: pd.DataFrame, hxr_df: pd.DataFrame) -> go.Figure:
    s1 = sxr_df["flux_long"].resample("1min").mean()
    h1 = hxr_df["hxr_total"].resample("1min").mean()
    sa, ha = s1.align(h1, join="inner")
    ratio = (ha / sa.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, name="Spectral Hardness (HXR / SXR)",
                             line=dict(color="#f59e0b", width=1.4),
                             fill="tozeroy", fillcolor="rgba(245,158,11,0.08)"))
    fig.update_layout(title="Spectral Hardness Ratio (Non-Thermal HXR / Thermal SXR Precursor Index)",
                      yaxis_title="Hardness Ratio", height=200, **_GRAPH_LAYOUT)
    return fig


def _fig_probability(prob_s: pd.Series, threshold: float,
                     events_df: pd.DataFrame | None) -> go.Figure:
    fig = go.Figure()
    p1 = prob_s.resample("1min").mean().dropna()
    y_clipped = p1.clip(upper=threshold)
    fig.add_trace(go.Scatter(
        x=p1.index, y=y_clipped.values,
        line=dict(width=0), fill="tozeroy", fillcolor="rgba(245,158,11,0.10)",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=p1.index, y=p1.values, name="P(Flare in next 30 min)",
                             line=dict(color=ORANGE, width=1.6),
                             fill="tonexty", fillcolor="rgba(239,68,68,0.28)"))
    fig.add_hline(y=threshold, line_color="#ffffff", line_dash="dash", line_width=1.1,
                  annotation_text=f"Operational Alert Threshold = {threshold:.2f}",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="#ffffff", family=FONT_SERIF))
    if events_df is not None and not events_df.empty:
        for _, ev in events_df.iterrows():
            cls = str(ev.get("flare_class") or "")
            pt  = ev.get("peak_time")
            if pd.isnull(pt):
                continue
            fig.add_vline(x=pt.isoformat(), line_color=_flare_color(cls),
                          line_width=1.2, line_dash="dot",
                          annotation_text=f"Actual: {cls}", annotation_font_size=10,
                          annotation_font=dict(color=_flare_color(cls), family=FONT_SERIF))
    fig.update_layout(title="Deep Neural Forecast Horizon — P(Flare in next 30 minutes)",
                      yaxis_title="P(Flare)", yaxis_range=[0, 1],
                      height=260, **_GRAPH_LAYOUT)
    return fig


def _fig_class_prob(cls_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    c1 = cls_df.resample("1min").mean()
    for col in c1.columns:
        s = c1[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=col,
                                 line=dict(color=FLARE_COLORS.get(col, "#888"), width=1.5)))
    fig.update_layout(title="Intensity-Class Probabilities  (A / B / C / M / X)",
                      yaxis_title="P(class | flare)", yaxis_range=[0, 1],
                      height=230, **_GRAPH_LAYOUT)
    return fig


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_forecaster(feat: pd.DataFrame, model, norm_mean, norm_std,
                    window_min: int) -> tuple:
    vals = feat[FEATURE_NAMES].ffill().fillna(0.0).values.astype(np.float32)
    if norm_mean is not None and norm_std is not None:
        vals = (vals - norm_mean) / (norm_std + 1e-8)
    n     = len(vals)
    probs = np.full(n, np.nan, dtype=np.float32)
    is_mc = getattr(model, "class_head", None) is not None
    n_cls = model.class_head.out_features if is_mc else 0
    cls_a = np.full((n, n_cls), np.nan, dtype=np.float32) if is_mc else None

    if n <= window_min:
        prob_s = pd.Series(probs, index=feat.index, name="prob")
        return (prob_s, None) if cls_a is None else (
            prob_s, pd.DataFrame(cls_a, index=feat.index, columns=FLARE_CLASS_ORDER[:n_cls]))

    # Build all valid windows into one batch — ~1000× faster than per-minute loop
    indices = np.arange(window_min, n)
    batch   = np.stack([vals[i - window_min: i] for i in indices])  # (N, window, features)
    x       = torch.from_numpy(batch)

    model.eval()
    with torch.no_grad():
        out = model(x)

    if isinstance(out, tuple):
        logit, clogit = out
        probs[indices] = torch.sigmoid(logit).numpy()
        cls_a[indices] = torch.softmax(clogit, dim=1).numpy()
    else:
        probs[indices] = torch.sigmoid(out).numpy()

    prob_s = pd.Series(probs, index=feat.index, name="prob")
    if cls_a is None:
        return prob_s, None
    cls_df = pd.DataFrame(cls_a, index=feat.index,
                          columns=FLARE_CLASS_ORDER[:n_cls])
    return prob_s, cls_df


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _df_to_json(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None
    d = df.copy()
    d.index = [str(i) for i in d.index]
    return d.to_json(orient="split")


def _json_to_df(s: str | None) -> pd.DataFrame | None:
    if not s:
        return None
    df = pd.read_json(io.StringIO(s), orient="split")
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df


def _events_to_records(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    d = df.copy()
    for col in d.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        d[col] = d[col].astype(str)
    return d.fillna("").to_dict("records")


def _records_to_df(rows: list) -> pd.DataFrame | None:
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for col in ["start_time", "peak_time", "end_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df if not df.empty else None


# ── App layout ────────────────────────────────────────────────────────────────

app = Dash(
    __name__,
    requests_pathname_prefix="/dashboard/",
    routes_pathname_prefix="/dashboard/",
    title="Solar Flare Monitor — Aditya-L1",
    suppress_callback_exceptions=True
)
server = app.server   # WSGI entry point for production

@server.route("/")
@server.route("/landing")
@server.route("/index.html")
def serve_landing():
    from flask import send_file
    return send_file(ROOT / "index.html")

@server.route("/assets/<path:filename>")
def serve_assets(filename):
    from flask import send_from_directory
    return send_from_directory(ROOT / "assets", filename)

# NOTE: CSV download route is defined at the bottom of this file using
# @app.server.route("/download-catalog") with make_response() for proper headers.

_CTRL_LABEL = {
    "color": "#9ca3af", "fontSize": "10px", "letterSpacing": "0.08em",
    "fontWeight": "600", "marginBottom": "6px",
    "textTransform": "uppercase", "fontFamily": FONT_SERIF,
}

_TAB_STYLE = {
    "background": "transparent",
    "color": "#9ca3af",
    "border": "none",
    "borderBottom": "2px solid transparent",
    "padding": "12px 24px",
    "fontSize": "13px",
    "fontWeight": "600",
    "fontFamily": FONT_SERIF,
    "letterSpacing": "0.02em",
    "cursor": "pointer",
}
_TAB_SELECTED = {
    **_TAB_STYLE,
    "color": "#ffffff",
    "fontWeight": "700",
    "borderBottom": f"2px solid {BLUE}",
    "background": "rgba(59,130,246,0.08)",
}

# Google Fonts and CSS theme injection
_FONT_LINK = html.Link(
    rel="stylesheet",
    href="https://cdn.jsdelivr.net/npm/anthropic-fonts@1.1.0/cdn/v1/css/all.css",
)
_THEME_LINK = html.Link(
    rel="stylesheet",
    href="/dashboard/assets/theme.css",
)

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
                <link rel="preconnect" href="https://cdn.jsdelivr.net">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/anthropic-fonts@1.1.0/cdn/v1/css/all.css">
        <link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..700&family=Lora:ital,wght@0,400..700;1,400..700&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="/dashboard/assets/theme.css">
        <link rel="stylesheet" href="/assets/theme.css">
        {%css%}
        <style>
        :root {
            /* Modern Dash Component Theme Variables */
            --Dash-Fill-Inverse-Strong: #111827 !important;
            --Dash-Fill-Inverse-strong: #111827 !important;
            --Dash-Text-Strong: #f9fafb !important;
            --Dash-Text-Primary: #f9fafb !important;
            --Dash-Text-Weak: #9ca3af !important;
            --Dash-Text-Disabled: #6b7280 !important;
            --Dash-Stroke-Strong: #283548 !important;
            --Dash-Stroke-Weak: #1f2937 !important;
            --Dash-Fill-Interactive-Weak: #1f2937 !important;
            --Dash-Fill-Interactive-Strong: #3b82f6 !important;
            --Dash-Fill-Disabled: #1f2937 !important;
            --Dash-Shading-Weak: rgba(0, 0, 0, 0.4) !important;
            --Dash-Shading-Strong: rgba(0, 0, 0, 0.8) !important;
            --Dash-Border-Radius: 8px !important;
            --Dash-Spacing: 4px !important;
            --font-serif: 'Anthropic Serif', 'Tiempos Text', 'Copernicus', 'Newsreader', 'Lora', Georgia, serif;
            --font-mono: 'Anthropic Mono', 'DM Mono', monospace;
        }

        /* Modern Dash Dropdown */
        .dash-dropdown,
        .dash-dropdown-wrapper,
        .dash-dropdown-content,
        .dash-dropdown-trigger,
        .dash-dropdown-grid-container {
            background-color: #111827 !important;
            background: #111827 !important;
            border-color: #283548 !important;
            color: #f9fafb !important;
            border-radius: 8px !important;
            font-family: var(--font-serif) !important;
        }
        .dash-dropdown-trigger {
            min-height: 38px !important;
            height: 38px !important;
        }
        .dash-dropdown-value,
        .dash-dropdown-value > *,
        .dash-dropdown-value-item,
        .dash-dropdown-placeholder,
        .dash-dropdown-search {
            color: #f9fafb !important;
            font-family: var(--font-serif) !important;
        }
        .dash-dropdown-placeholder {
            color: #6b7280 !important;
        }
        .dash-dropdown-trigger-icon,
        .dash-dropdown-search-icon,
        .dash-dropdown-clear,
        .dash-dropdown-value-count {
            color: #9ca3af !important;
            fill: #9ca3af !important;
        }
        .dash-dropdown-content {
            background-color: #111827 !important;
            background: #111827 !important;
            border: 1px solid #283548 !important;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.8) !important;
            border-radius: 8px !important;
        }
        .dash-dropdown-options {
            background-color: #111827 !important;
        }
        .dash-dropdown-option,
        .dash-options-list-option {
            background-color: #111827 !important;
            background: #111827 !important;
            color: #e5e7eb !important;
            font-family: var(--font-serif) !important;
            border-bottom: 1px solid #1f2937 !important;
            padding: 8px 12px !important;
        }
        .dash-options-list-option-text {
            color: #e5e7eb !important;
        }
        .dash-dropdown-option:hover,
        .dash-dropdown-option:focus,
        .dash-options-list-option:hover,
        .dash-options-list-option:focus-within {
            background-color: #1f2937 !important;
            background: #1f2937 !important;
            color: #38bdf8 !important;
        }

        /* Modern Dash DatePicker */
        .dash-datepicker,
        .dash-datepicker-input-wrapper {
            background-color: #111827 !important;
            background: #111827 !important;
            border: 1px solid #283548 !important;
            border-radius: 8px !important;
            color: #f9fafb !important;
            height: 38px !important;
            box-sizing: border-box !important;
        }
        .dash-datepicker-input {
            background-color: transparent !important;
            background: transparent !important;
            color: #f9fafb !important;
            font-family: var(--font-mono) !important;
            font-size: 13px !important;
            font-weight: 500 !important;
        }
        .dash-datepicker-caret-icon,
        .dash-datepicker-clear {
            color: #9ca3af !important;
            fill: #9ca3af !important;
        }
        .dash-datepicker-calendar-container,
        .dash-datepicker-calendar-wrapper,
        .dash-datepicker-content {
            background-color: #0e1422 !important;
            background: #0e1422 !important;
            border: 1px solid #283548 !important;
            border-radius: 10px !important;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.8) !important;
        }
        .dash-datepicker-calendar {
            background-color: #0e1422 !important;
        }
        .dash-datepicker-calendar th {
            color: #9ca3af !important;
            font-family: var(--font-serif) !important;
            font-weight: 600 !important;
        }
        .dash-datepicker-calendar td {
            background-color: #111827 !important;
            color: #e5e7eb !important;
            border: 1px solid #1f2937 !important;
            font-family: var(--font-mono) !important;
            border-radius: 4px !important;
        }
        .dash-datepicker-calendar td:hover {
            background-color: #1f2937 !important;
            color: #38bdf8 !important;
        }
        .dash-datepicker-calendar td.dash-datepicker-calendar-date-selected {
            background-color: #3b82f6 !important;
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        .dash-datepicker-month-nav {
            color: #9ca3af !important;
            background: #111827 !important;
            border: 1px solid #283548 !important;
            border-radius: 6px !important;
        }
        .dash-datepicker-month-nav:hover:not(:disabled) {
            background: #1f2937 !important;
            color: #38bdf8 !important;
        }

        /* Legacy react-dates & react-select Fallbacks */
        .SingleDatePickerInput, .SingleDatePickerInput__withBorder {
            background-color: #111827 !important;
            border: 1px solid #283548 !important;
            border-radius: 8px !important;
        }
        .DateInput, .DateInput_input {
            background-color: transparent !important;
            color: #f9fafb !important;
            font-family: var(--font-mono) !important;
            border: none !important;
        }
        .Select-control {
            background-color: #111827 !important;
            border: 1px solid #283548 !important;
            border-radius: 8px !important;
            color: #f9fafb !important;
        }
        .Select-value, .Select-value-label, .Select--single > .Select-control .Select-value .Select-value-label {
            color: #f9fafb !important;
            font-family: var(--font-serif) !important;
        }
        .Select-placeholder {
            color: #6b7280 !important;
            font-family: var(--font-serif) !important;
        }
        .Select-input > input {
            color: #f9fafb !important;
            font-family: var(--font-serif) !important;
        }
        .Select-menu-outer {
            background-color: #111827 !important;
            border: 1px solid #283548 !important;
        }
        .Select-option {
            background-color: #111827 !important;
            color: #d1d5db !important;
        }
        .Select-option.is-focused {
            background-color: #1f2937 !important;
            color: #f9fafb !important;
        }
        .rc-slider-tooltip-inner {
            background-color: #111827 !important;
            color: #f9fafb !important;
            border: 1px solid #283548 !important;
        }
        </style>
    </head>
    <body>
        <!--[if IE]><script>
        alert("Dash v2.7+ does not support Internet Explorer. Please use a newer browser.");
        </script><![endif]-->
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = html.Div([

    _FONT_LINK,
    _THEME_LINK,
    dcc.Location(id="url", refresh=False),

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        # Left: brand
        html.Div([
            html.Img(src="/assets/logo.png", style={
                "height": "36px", "width": "36px", "borderRadius": "50%",
                "marginRight": "14px", "verticalAlign": "middle",
                "boxShadow": "0 0 16px rgba(245,158,11,0.45)",
                "border": "1.5px solid rgba(245,158,11,0.5)",
            }),
            html.Div([
                html.Div(
                    "Aditya-L1 Solar Flare Observatory",
                    style={"fontSize": "1.15em", "fontWeight": "700", "color": FG,
                           "letterSpacing": "-0.01em",
                           "fontFamily": FONT_SERIF},
                ),
                html.Div(
                    "ISRO PS-15 · SoLEXS (SXR) + HEL1OS (HXR) Space Weather Console",
                    style={"fontSize": "11px", "color": "#9ca3af",
                           "fontWeight": "500", "fontFamily": FONT_MONO,
                           "letterSpacing": "0.04em"},
                ),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),

        # Right: nav pills
        html.Div([
            html.A("← Observatory Home", href="/", style={
                "color": AMBER, "fontSize": "12px", "fontWeight": "600",
                "background": "rgba(245,158,11,0.08)",
                "border": "1px solid rgba(245,158,11,0.3)",
                "borderRadius": "6px",
                "padding": "6px 14px", "marginRight": "10px",
                "textDecoration": "none",
                "fontFamily": FONT_SERIF,
                "letterSpacing": "0.02em",
            }),
            html.Span("L1 HALO ORBIT · 1.5M KM", style={
                "color": CYAN, "fontSize": "11px", "fontWeight": "600",
                "background": "rgba(56,189,248,0.08)",
                "border": "1px solid rgba(56,189,248,0.25)",
                "borderRadius": "6px", "padding": "5px 12px",
                "fontFamily": FONT_SERIF,
                "letterSpacing": "0.06em",
            }),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "background": "#0b0f19",
        "padding": "14px 28px",
        "borderBottom": f"1px solid {BORDER2}",
    }),

    # ── Controls bar ──────────────────────────────────────────────────────────
    html.Div([
        # Curated Benchmark Flare Presets
        html.Div([
            html.Div("Curated Benchmark Event", style=_CTRL_LABEL),
            dcc.Dropdown(
                id="preset-drop",
                options=[
                    {"label": "Jump to landmark flare...", "value": ""},
                    {"label": "2024-05-10 — X3.6 Flare (Confirmed Multi-Instrument)", "value": "2024-05-10:solexs"},
                    {"label": "2024-02-09 — X3.3 Major Solar Eruption", "value": "2024-02-09:solexs"},
                    {"label": "2024-03-23 — X1.1 Proton Flare", "value": "2024-03-23:solexs"},
                    {"label": "2024-05-10 — M8.1 Flare (Impulsive Multi-Burst)", "value": "2024-05-10:solexs"},
                    {"label": "2024-04-01 — Quiet-Sun Baseline Observation", "value": "2024-04-01:solexs"},
                ],
                value="", clearable=True,
                style={"width": "280px", "backgroundColor": "#111827",
                       "color": FG, "fontSize": "13px", "borderRadius": "8px"},
            ),
        ], style={"marginRight": "20px"}),

        # Date
        html.Div([
            html.Div("Observation Date", style=_CTRL_LABEL),
            dcc.DatePickerSingle(
                id="date-picker",
                date="2024-05-10",
                display_format="YYYY-MM-DD",
                style={"fontSize": "13px"},
            ),
        ], style={"marginRight": "20px"}),

        # Instrument
        html.Div([
            html.Div("Instrument Stream", style=_CTRL_LABEL),
            dcc.Dropdown(
                id="instrument-drop",
                options=[
                    {"label": "SoLEXS (Aditya-L1 SXR)", "value": "solexs"},
                    {"label": "GOES-16 XRS Benchmark", "value": "goes"},
                ],
                value="solexs", clearable=False,
                style={"width": "210px", "backgroundColor": "#111827",
                       "color": FG, "fontSize": "13px", "borderRadius": "8px"},
            ),
        ], style={"marginRight": "20px"}),

        # Alert threshold
        html.Div([
            html.Div("Alert Threshold", style=_CTRL_LABEL),
            html.Div([
                dcc.Slider(
                    id="threshold-slider",
                    min=0.1, max=0.9, step=0.05,
                    value=float(CFG["forecasting"]["alert_threshold"]),
                    marks={0.3: {"label": "0.3", "style": {"color": "#9ca3af", "fontSize": "11px", "fontWeight": "500"}},
                            0.5: {"label": "0.5", "style": {"color": "#9ca3af", "fontSize": "11px", "fontWeight": "500"}},
                            0.7: {"label": "0.7", "style": {"color": "#9ca3af", "fontSize": "11px", "fontWeight": "500"}}},
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode="drag",
                ),
            ], style={"width": "180px", "paddingTop": "2px"}),
        ], style={"marginRight": "20px"}),

        # Options
        html.Div([
            html.Div("Analytics Options", style=_CTRL_LABEL),
            dcc.Checklist(
                id="options-check",
                options=[
                    {"label": "  ML Forecast", "value": "forecast"},
                    {"label": "  Live Telemetry (60s)",  "value": "live"},
                ],
                value=["forecast"],
                inline=True,
                labelStyle={"marginRight": "16px", "fontSize": "13px", "fontWeight": "600",
                             "color": FG2, "fontFamily": FONT_SERIF},
            ),
        ], style={"marginRight": "20px"}),

        # Load button
        html.Button(
            "Load Telemetry",
            id="load-btn", n_clicks=0,
            style={
                "background": "#1e293b",
                "color": FG, "border": f"1px solid {BORDER2}",
                "borderRadius": "8px", "padding": "8px 20px",
                "cursor": "pointer", "fontSize": "13px",
                "fontWeight": "600", "marginTop": "18px",
                "fontFamily": FONT_SERIF,
                "letterSpacing": "0.03em",
            },
        ),
    ], style={
        "display": "flex", "flexWrap": "wrap", "alignItems": "flex-end",
        "background": "#0e1422",
        "padding": "14px 28px",
        "borderBottom": f"1px solid {BORDER}",
        "gap": "6px",
    }),

    # ── Alert banners ─────────────────────────────────────────────────────────
    html.Div(id="alert-banners", style={"padding": "10px 28px 0"}),

    # ── Status bar ────────────────────────────────────────────────────────────
    html.Div(
        id="status-bar",
        style={"padding": "7px 28px", "color": "#9bb8f8", "fontSize": "12px",
               "fontWeight": "600", "fontFamily": FONT_MONO,
               "letterSpacing": "0.02em", "minHeight": "28px"},
    ),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    html.Div([
        dcc.Tabs(id="main-tabs", value="curves", children=[
            dcc.Tab(label="Light Curves & Telemetry", value="curves",
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label="Predictive ML Forecast",    value="forecast",
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label="Master Event Catalog",     value="catalog",
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
        ]),
    ], style={
        "padding": "0 28px",
        "background": BG2,
        "borderBottom": f"1px solid {BORDER2}",
    }),

    dcc.Loading(
        html.Div(id="tab-content", style={"padding": "18px 28px 36px"}),
        type="dot",
        color=CYAN,
    ),

    # ── Hidden stores & interval ──────────────────────────────────────────────
    dcc.Store(id="store-main"),
    dcc.Interval(id="live-interval", interval=60_000, disabled=True, n_intervals=0),

], style={
    "background": BG,
    "minHeight": "100vh",
    "fontFamily": FONT_SERIF,
    "color": FG,
    "fontSize": "14px",
})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("live-interval", "disabled"),
    Input("options-check", "value"),
)
def toggle_live(opts):
    return "live" not in (opts or [])


@app.callback(
    Output("date-picker", "date"),
    Output("instrument-drop", "value"),
    Output("main-tabs", "value"),
    Input("preset-drop", "value"),
    Input("url", "search"),
    State("date-picker", "date"),
    State("instrument-drop", "value"),
    State("main-tabs", "value"),
    prevent_initial_call=False,
)
def sync_preset_and_url(preset_val, url_search, cur_date, cur_inst, cur_tab):
    ctx = dash.callback_context
    if not ctx.triggered:
        if url_search:
            import urllib.parse
            params = urllib.parse.parse_qs(url_search.lstrip("?"))
            d = params.get("date", [cur_date])[0]
            inst = params.get("instrument", [cur_inst])[0]
            t = params.get("tab", [cur_tab])[0]
            return d, inst, t
        return cur_date, cur_inst, cur_tab

    trig_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trig_id == "preset-drop" and preset_val:
        parts = preset_val.split(":")
        if len(parts) == 2:
            return parts[0], parts[1], cur_tab
    elif trig_id == "url" and url_search:
        import urllib.parse
        params = urllib.parse.parse_qs(url_search.lstrip("?"))
        d = params.get("date", [cur_date])[0]
        inst = params.get("instrument", [cur_inst])[0]
        t = params.get("tab", [cur_tab])[0]
        return d, inst, t
    return cur_date, cur_inst, cur_tab


@app.callback(
    Output("preset-drop", "value"),
    Input("btn-quick-1", "n_clicks"),
    Input("btn-quick-2", "n_clicks"),
    Input("btn-quick-3", "n_clicks"),
    Input("btn-quick-4", "n_clicks"),
    prevent_initial_call=True,
)
def handle_quick_buttons(b1, b2, b3, b4):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trig = ctx.triggered[0]["prop_id"].split(".")[0]
    if trig == "btn-quick-1":
        return "2024-05-10:solexs"
    elif trig == "btn-quick-2":
        return "2024-02-09:solexs"
    elif trig == "btn-quick-3":
        return "2024-03-23:solexs"
    elif trig == "btn-quick-4":
        return "2024-05-10:solexs"
    raise PreventUpdate


@app.callback(
    Output("store-main",  "data"),
    Output("status-bar",  "children"),
    Input("load-btn",     "n_clicks"),
    Input("date-picker",  "date"),
    Input("instrument-drop", "value"),
    Input("live-interval","n_intervals"),
    State("options-check",   "value"),
    prevent_initial_call=False,
)
def load_day(n_clicks, date_str, instrument, n_intervals, opts):
    if not date_str:
        raise PreventUpdate

    ds = date_str.replace("-", "")
    sxr_df = hxr_df = None
    status = ""

    # ── Load raw / preprocessed data ─────────────────────────────────────────
    if instrument == "solexs":
        sp = PROC_DIR / f"{ds}_solexs.parquet"
        hp = PROC_DIR / f"{ds}_helios.parquet"
        # Bidirectional fallback between 2024 (historical) and 2026 (simulation/observed)
        if not sp.exists() and ds.startswith("2024"):
            alt_ds = "2026" + ds[4:]
            if (PROC_DIR / f"{alt_ds}_solexs.parquet").exists():
                sp = PROC_DIR / f"{alt_ds}_solexs.parquet"
                hp = PROC_DIR / f"{alt_ds}_helios.parquet"
        elif not sp.exists() and ds.startswith("2026"):
            alt_ds = "2024" + ds[4:]
            if (PROC_DIR / f"{alt_ds}_solexs.parquet").exists():
                sp = PROC_DIR / f"{alt_ds}_solexs.parquet"
                hp = PROC_DIR / f"{alt_ds}_helios.parquet"

        if not sp.exists():
            return None, f"No SoLEXS parquet for {date_str}. Process data first."
        sxr_df = pd.read_parquet(sp)
        if hp.exists():
            hxr_df = pd.read_parquet(hp)
    else:
        goes_dir = RAW_DIR / "goes"
        nc_files = sorted(goes_dir.glob(f"sci_xrsf-l2-avg1m_g16_d{ds}_*.nc"))
        nc_files += sorted(goes_dir.glob(f"dn_xrsf-l2-avg1m_g16_d{ds}_*.nc"))
        if not nc_files and ds.startswith("2024"):
            alt_ds = "2026" + ds[4:]
            nc_files = sorted(goes_dir.glob(f"sci_xrsf-l2-avg1m_g16_d{alt_ds}_*.nc"))
            nc_files += sorted(goes_dir.glob(f"dn_xrsf-l2-avg1m_g16_d{alt_ds}_*.nc"))
        elif not nc_files and ds.startswith("2026"):
            alt_ds = "2024" + ds[4:]
            nc_files = sorted(goes_dir.glob(f"sci_xrsf-l2-avg1m_g16_d{alt_ds}_*.nc"))
            nc_files += sorted(goes_dir.glob(f"dn_xrsf-l2-avg1m_g16_d{alt_ds}_*.nc"))

        if not nc_files:
            return None, f"No GOES NetCDF for {date_str}."
        raw = read_goes_nc(nc_files[0])
        if raw.empty:
            return None, f"GOES file empty for {date_str}."
        sxr_df = preprocess_goes(raw, CFG)

    # ── Nowcasting ────────────────────────────────────────────────────────────
    sxr_evs = pd.DataFrame()
    hxr_evs = pd.DataFrame()
    try:
        if "flux_long" in sxr_df.columns:
            fn = detect_solexs_flares if instrument == "solexs" else detect_goes_flares
            sxr_evs = events_to_dataframe(fn(sxr_df, CFG))
        if hxr_df is not None and "hxr_total" in hxr_df.columns:
            hxr_evs = events_to_dataframe(detect_helios_flares(hxr_df, CFG))
    except Exception as e:
        status = f"Nowcast warning: {e}"

    master_evs = merge_catalogs(sxr_evs, hxr_evs)

    # ── ML Forecast ───────────────────────────────────────────────────────────
    prob_dict = cls_dict = None
    if "forecast" in (opts or []):
        model, nmean, nstd = _get_model()
        if model is not None:
            try:
                feat = build_feature_series(sxr_df, hxr_df)
                wmin = CFG["forecasting"]["input_window_min"]
                prob_s, cls_df = _run_forecaster(feat, model, nmean, nstd, wmin)
                prob_dict = {
                    "times": [str(t) for t in prob_s.index],
                    "probs": prob_s.fillna(-1).tolist(),
                }
                if cls_df is not None:
                    cls_dict = {
                        "_times": [str(t) for t in cls_df.index],
                        **{col: cls_df[col].fillna(-1).tolist()
                           for col in cls_df.columns},
                    }
            except Exception as e:
                status = (status + " | " if status else "") + f"Forecast error: {e}"
        else:
            status = (status + " | " if status else "") + "Model not found — run train.py first."

    # ── Pack store ────────────────────────────────────────────────────────────
    store = {
        "sxr":        _df_to_json(sxr_df),
        "hxr":        _df_to_json(hxr_df),
        "sxr_evs":    _events_to_records(sxr_evs),
        "hxr_evs":    _events_to_records(hxr_evs),
        "master_evs": _events_to_records(master_evs),
        "prob":       prob_dict,
        "cls":        cls_dict,
        "date":       date_str,
        "instrument": instrument,
    }

    n_ev   = len(master_evs) if not master_evs.empty else 0
    status = f"Loaded {date_str} ({instrument}) — {n_ev} flare(s) detected" + \
             (f" | {status}" if status else "")
    return store, status


@app.callback(
    Output("alert-banners", "children"),
    Input("store-main",      "data"),
    Input("threshold-slider","value"),
)
def update_alerts(store, threshold):
    if not store:
        return []
    banners = []

    for ev in (store.get("master_evs") or []):
        cls  = str(ev.get("flare_class") or "")
        peak = str(ev.get("peak_time") or "")[:19]
        conf = ev.get("confirmed_multi_instrument", False)
        col  = _flare_color(cls)
        tag  = "  ✓ multi-instrument confirmed" if conf else ""
        banners.append(html.Div(
            [html.Span("🔔  FLARE NOWCASTED", style={"letterSpacing": "0.07em"}),
             html.Span(f"  {cls}", style={"fontFamily": FONT_MONO,
                                         "background": f"{col}25", "borderRadius": "4px",
                                         "padding": "1px 7px", "marginLeft": "6px"}),
             html.Span(f"  @  {peak} UTC", style={"fontFamily": FONT_MONO,
                                                   "fontSize": "12px", "color": f"{col}cc"}),
             html.Span(tag, style={"fontSize": "11px", "opacity": "0.7", "marginLeft": "8px"})],
            style={
                "background": f"linear-gradient(90deg, {col}12 0%, transparent 100%)",
                "border": f"1px solid {col}44",
                "borderLeft": f"4px solid {col}",
                "color": col,
                "borderRadius": "8px",
                "padding": "9px 16px",
                "marginBottom": "6px",
                "fontWeight": "700",
                "fontSize": "13px",
                "fontFamily": FONT_SERIF,
                "display": "flex",
                "alignItems": "center",
            }
        ))

    pdata = store.get("prob")
    if pdata:
        probs = [p for p in pdata["probs"] if p >= 0]
        if probs and probs[-1] >= threshold:
            lp = probs[-1]
            banners.append(html.Div(
                [html.Span("⚠  FORECAST ALERT", style={"letterSpacing": "0.07em"}),
                 html.Span(f"  P = {lp:.3f}",
                           style={"fontFamily": FONT_MONO,
                                  "background": "rgba(255,100,60,0.18)",
                                  "borderRadius": "4px", "padding": "1px 8px",
                                  "marginLeft": "8px"}),
                 html.Span(f"  threshold {threshold:.2f}  ·  within {CFG['forecasting']['lead_time_min']} min",
                           style={"fontSize": "12px", "opacity": "0.75",
                                  "fontFamily": FONT_MONO, "marginLeft": "8px"})],
                style={
                    "background": "linear-gradient(90deg, rgba(255,80,40,0.12) 0%, transparent 100%)",
                    "border": "1px solid rgba(255,80,40,0.4)",
                    "borderLeft": "4px solid #ff5028",
                    "color": "#ff7a55",
                    "borderRadius": "8px",
                    "padding": "9px 16px",
                    "marginBottom": "6px",
                    "fontWeight": "700",
                    "fontSize": "13px",
                    "fontFamily": FONT_SERIF,
                    "display": "flex",
                    "alignItems": "center",
                }
            ))

    return banners


@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs",       "value"),
    Input("store-main",      "data"),
    Input("threshold-slider","value"),
)
def render_tab(tab, store, threshold):
    if tab == "catalog":
        return _tab_catalog()
    if not store:
        return html.Div([
            html.Div([
                html.Div("ISRO ADITYA-L1 MISSION · L1 HALO ORBIT OBSERVATORY", style={
                    "fontFamily": FONT_SERIF, "fontSize": "11px",
                    "letterSpacing": "0.1em", "color": CYAN, "fontWeight": "700",
                    "textTransform": "uppercase", "marginBottom": "8px",
                }),
                html.H3("Telemetry Channel Standby", style={
                    "color": FG, "fontSize": "22px", "fontWeight": "700",
                    "fontFamily": FONT_SERIF, "marginBottom": "10px",
                }),
                html.P("Select an observation pass above, or launch one of the curated landmark solar events below to initialize light curves and neural forecasting:", style={
                    "color": "#9ab0e8", "fontSize": "14px", "maxWidth": "650px", "margin": "0 auto 24px",
                    "lineHeight": "1.6",
                }),
                html.Div([
                    html.Button("Load May 10, 2024 (X3.6 Superstorm)", id="btn-quick-1", n_clicks=0, style={
                        "background": "#141c38", "color": FG, "border": f"1px solid {BORDER2}",
                        "borderRadius": "8px", "padding": "10px 16px", "fontFamily": FONT_SERIF,
                        "fontSize": "12px", "fontWeight": "600", "cursor": "pointer", "margin": "4px",
                    }),
                    html.Button("Load Feb 09, 2024 (X3.3 Major Flare)", id="btn-quick-2", n_clicks=0, style={
                        "background": "#141c38", "color": FG, "border": f"1px solid {BORDER2}",
                        "borderRadius": "8px", "padding": "10px 16px", "fontFamily": FONT_SERIF,
                        "fontSize": "12px", "fontWeight": "600", "cursor": "pointer", "margin": "4px",
                    }),
                    html.Button("Load Mar 23, 2024 (X1.1 Proton Event)", id="btn-quick-3", n_clicks=0, style={
                        "background": "#141c38", "color": FG, "border": f"1px solid {BORDER2}",
                        "borderRadius": "8px", "padding": "10px 16px", "fontFamily": FONT_SERIF,
                        "fontSize": "12px", "fontWeight": "600", "cursor": "pointer", "margin": "4px",
                    }),
                    html.Button("Load May 10, 2024 (M8.1 Multi-Burst)", id="btn-quick-4", n_clicks=0, style={
                        "background": "#141c38", "color": FG, "border": f"1px solid {BORDER2}",
                        "borderRadius": "8px", "padding": "10px 16px", "fontFamily": FONT_SERIF,
                        "fontSize": "12px", "fontWeight": "600", "cursor": "pointer", "margin": "4px",
                    }),
                ], style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center", "gap": "6px"}),
            ], style={
                "background": CARD, "borderRadius": "14px", "border": f"1px solid {BORDER}",
                "padding": "48px 32px", "maxWidth": "800px", "margin": "30px auto",
            }),
        ], style={"textAlign": "center"})
    if tab == "curves":
        return _tab_curves(store)
    if tab == "forecast":
        return _tab_forecast(store, threshold)
    return html.Div()


# ── Tab renderers ─────────────────────────────────────────────────────────────

def _tab_curves(store: dict) -> html.Div:
    sxr_df  = _json_to_df(store.get("sxr"))
    hxr_df  = _json_to_df(store.get("hxr"))
    all_evs = _records_to_df(store.get("master_evs"))
    hxr_evs = _records_to_df(store.get("hxr_evs"))
    kids    = []

    if sxr_df is not None and "flux_long" in sxr_df.columns:
        kids.append(_card(
            dcc.Graph(figure=_fig_sxr(sxr_df, all_evs, instrument=store.get("instrument", "solexs")),
                      config={"displayModeBar": False}),
        ))
    else:
        kids.append(html.Div("No SXR data for this day.",
                             style={"color": "#9ab0e8", "fontSize": "13px", "padding": "16px 0"}))

    if hxr_df is not None and "hxr_total" in hxr_df.columns:
        kids.append(_card(
            dcc.Graph(figure=_fig_hxr(hxr_df, hxr_evs),
                      config={"displayModeBar": False}),
        ))
        if sxr_df is not None and "flux_long" in sxr_df.columns:
            kids.append(_card(
                dcc.Graph(figure=_fig_hardness(sxr_df, hxr_df),
                          config={"displayModeBar": False}),
            ))
    else:
        kids.append(html.Div("No HEL1OS HXR data for this day.",
                             style={"color": "#9ab0e8", "fontSize": "13px"}))

    # Detected-events mini-table
    evs = store.get("master_evs") or []
    if evs:
        _cols = ["peak_time", "flare_class", "start_time", "end_time",
                 "solexs_peak_flux", "confirmed_multi_instrument"]
        rows  = [{k: v for k, v in r.items() if k in _cols} for r in evs]
        cols  = [c for c in _cols if c in rows[0]]
        kids.append(_card(
            html.Div([
                html.H4("Detected Flares Today",
                        style={"margin": "0", "color": FG, "fontSize": "14px",
                               "fontFamily": FONT_SERIF, "fontWeight": "700"}),
                html.Span(f"{len(evs)} event(s)",
                          style={"color": CYAN, "fontSize": "12px",
                                 "fontFamily": FONT_MONO}),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginBottom": "12px"}),
            dash_table.DataTable(
                data=rows,
                columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols],
                style_table={"overflowX": "auto", "borderRadius": "6px"},
                style_header={"background": CARD, "color": FG2,
                               "fontWeight": "700", "border": f"1px solid {BORDER2}",
                               "fontSize": "11px", "letterSpacing": "0.06em",
                               "fontFamily": FONT_SERIF,
                               "textTransform": "uppercase"},
                style_cell={"background": BG2, "color": FG,
                             "border": f"1px solid {BORDER}",
                             "fontSize": "13px", "padding": "8px 12px",
                             "fontFamily": FONT_MONO},
                style_data={"borderTop": f"1px solid {BORDER}"},
                page_size=15,
            ),
        ))
    else:
        kids.append(html.Div(
            [html.Span("✓  ", style={"color": GREEN}), "No flares detected on this day."],
            style={"color": "#9ab0e8", "padding": "10px 0", "fontSize": "13px",
                   "fontFamily": FONT_SERIF}
        ))

    return html.Div(kids)


def _tab_forecast(store: dict, threshold: float) -> html.Div:
    pdata  = store.get("prob")
    cdata  = store.get("cls")
    all_ev = _records_to_df(store.get("master_evs"))
    kids   = []

    if not pdata:
        return html.Div(
            "ML forecast not available — enable forecast and click Load.",
            style={"color": "#9ab0e8", "fontSize": "14px", "fontWeight": "500",
                   "paddingTop": "32px", "textAlign": "center"},
        )

    times  = pd.to_datetime(pdata["times"])
    probs  = np.array(pdata["probs"], dtype=float)
    probs[probs < 0] = np.nan
    prob_s = pd.Series(probs, index=times, name="prob")

    latest_p = float(prob_s.dropna().iloc[-1]) if not prob_s.dropna().empty else float("nan")
    above    = not np.isnan(latest_p) and latest_p >= threshold
    p_color  = "#ff4444" if above else "#33cc88"

    # KPI row
    kids.append(html.Div([
        _kpi("Current P(flare)",
             f"{latest_p:.3f}" if not np.isnan(latest_p) else "—",
             p_color,
             "ALERT ACTIVE" if above else f"< threshold ({threshold:.2f})"),
        _kpi("Model AUC",  f"{_CKPT_META.get('auc', 0.818):.3f}" if _CKPT_META else "0.818", BLUE, "Aditya-L1 test set"),
        _kpi("TSS @ 0.5",  f"{_CKPT_META.get('tss', 0.361):.3f}" if _CKPT_META else "0.361", CYAN, "true skill statistic"),
        _kpi("Lead P90",   "27 min", AMBER, "target: 30 min"),
    ], style={"marginBottom": "8px"}))

    kids.append(_card(
        dcc.Graph(figure=_fig_probability(prob_s, threshold, all_ev),
                  config={"displayModeBar": True, "displaylogo": False}),
    ))

    if cdata:
        cls_times = pd.to_datetime(cdata["_times"])
        cls_cols  = [c for c in cdata if c != "_times"]
        cls_df    = pd.DataFrame(
            {col: np.array(cdata[col], dtype=float) for col in cls_cols},
            index=cls_times,
        )
        cls_df[cls_df < 0] = np.nan
        if not cls_df.dropna(how="all").empty:
            # Most-likely class from latest valid row
            last_row = cls_df.dropna(how="all").iloc[-1]
            best_cls = last_row.idxmax()
            best_p   = last_row.max()
            kids.insert(1, html.Div([
                _kpi("Most-likely class", best_cls,
                     _flare_color(best_cls),
                     f"P = {best_p:.3f}"),
            ]))
            kids.append(_card(
                dcc.Graph(figure=_fig_class_prob(cls_df),
                          config={"displayModeBar": False}),
            ))

    return html.Div(kids)


def _tab_catalog() -> html.Div:
    cat_path = ROOT / CFG["paths"]["master_catalog"]
    if not cat_path.exists():
        return html.Div("Master catalog not found. Run process_all.py first.",
                        style={"color": "#9ab0e8", "fontSize": "14px", "paddingTop": "32px"})

    cat = load_master_catalog(str(cat_path))
    if cat.empty:
        return html.Div("Catalog is empty.", style={"color": "#9ab0e8", "fontSize": "14px"})

    counts  = cat["flare_class"].str[0].value_counts().to_dict()
    multi_n = int(cat.get("confirmed_multi_instrument", pd.Series(dtype=bool)).sum())

    lag_series = cat.loc[cat.get("confirmed_multi_instrument", False) & cat.get("neupert_lag_sec", pd.Series(dtype=float)).notna(), "neupert_lag_sec"]
    median_lag_txt = f"{lag_series.median():+.0f}s" if not lag_series.empty else "—"

    kpis = html.Div([
        _kpi("Total events", len(cat), BLUE),
        *[_kpi(f"{c}-class", counts[c], _flare_color(c))
          for c in ["X", "M", "C", "B"] if c in counts],
        _kpi("Multi-instrument", multi_n, VIOLET, "confirmed"),
        _kpi("Median Neupert lag", median_lag_txt, GREEN, "HXR lead time"),
    ], style={"marginBottom": "12px", "display": "flex", "flexWrap": "wrap", "gap": "10px"})

    disp = cat.copy().sort_values("peak_time", ascending=False).reset_index(drop=True)
    for col in ["start_time", "peak_time", "end_time"]:
        if col in disp.columns:
            disp[col] = disp[col].astype(str).str[:19]
    for col in ["solexs_peak_flux", "helios_peak_flux"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(
                lambda v: f"{v:.3e}" if pd.notnull(v) and v != "" else "")
    if "hardness_ratio" in disp.columns:
        disp["hardness_ratio"] = disp["hardness_ratio"].apply(
            lambda v: f"{v:.2f}" if pd.notnull(v) and v != "" else "")

    def _fmt_duration(sec):
        if pd.isna(sec) or sec == "" or sec < 0:
            return "—"
        sec = int(round(float(sec)))
        m, s = divmod(sec, 60)
        return f"{m}m {s:02d}s" if m > 0 else f"{s}s"

    if "duration_sec" in disp.columns:
        disp["duration"] = disp["duration_sec"].apply(_fmt_duration)
    if "rise_time_sec" in disp.columns:
        disp["rise_time"] = disp["rise_time_sec"].apply(_fmt_duration)
    if "decay_time_sec" in disp.columns:
        disp["decay_time"] = disp["decay_time_sec"].apply(_fmt_duration)
    if "neupert_lag_sec" in disp.columns:
        disp["neupert_lag"] = disp["neupert_lag_sec"].apply(
            lambda v: f"{float(v):+.0f}s" if pd.notnull(v) and v != "" else "—")
    if "hxr_burst_count" in disp.columns:
        disp["hxr_bursts"] = disp["hxr_burst_count"].fillna(0).astype(int)

    col_mapping = [
        ("peak_time", "Peak Time"),
        ("flare_class", "Class"),
        ("start_time", "Start Time"),
        ("end_time", "End Time"),
        ("duration", "Duration"),
        ("rise_time", "Rise"),
        ("decay_time", "Decay"),
        ("neupert_lag", "Neupert Lag"),
        ("hxr_bursts", "HXR Bursts"),
        ("solexs_peak_flux", "SXR Peak Flux"),
        ("helios_peak_flux", "HXR Peak Flux"),
        ("hardness_ratio", "Hardness Ratio"),
        ("confirmed_multi_instrument", "Confirmed"),
    ]
    _cols = [c for c, _ in col_mapping if c in disp.columns]
    table_columns = [{"name": title, "id": c} for c, title in col_mapping if c in disp.columns]

    table = dash_table.DataTable(
        data=disp[_cols].to_dict("records"),
        columns=table_columns,
        sort_action="native",
        filter_action="native",
        page_size=25,
        style_table={"overflowX": "auto", "borderRadius": "8px"},
        style_header={"background": CARD, "color": FG2,
                       "fontWeight": "700", "border": f"1px solid {BORDER2}",
                       "fontSize": "11px", "letterSpacing": "0.07em",
                       "textTransform": "uppercase",
                       "fontFamily": FONT_SERIF},
        style_cell={"background": BG2, "color": FG,
                     "border": f"1px solid {BORDER}",
                     "fontSize": "13px", "padding": "9px 14px",
                     "fontFamily": FONT_MONO},
        style_data={"borderTop": f"1px solid {BORDER}"},
        style_data_conditional=[
            {"if": {"filter_query": '{flare_class} contains "X"'},
             "color": FLARE_COLORS["X"], "fontWeight": "700"},
            {"if": {"filter_query": '{flare_class} contains "M"'},
             "color": FLARE_COLORS["M"], "fontWeight": "600"},
            {"if": {"filter_query": '{flare_class} contains "C"'},
             "color": FLARE_COLORS["C"]},
            {"if": {"row_index": "odd"},
             "background": "#080a1e"},
        ],
    )

    return html.Div([
        kpis,
        _card(
            html.Div([
                html.H4("Master Flare Catalog",
                        style={"margin": "0", "color": FG, "fontSize": "15px",
                               "fontFamily": FONT_SERIF,
                               "fontWeight": "700"}),
                html.A(
                    ["⬇  Download CSV"],
                    href="/download-catalog",
                    style={"color": BLUE, "fontSize": "12px", "fontWeight": "600",
                           "textDecoration": "none",
                           "background": "rgba(77,159,255,0.1)",
                           "border": "1px solid rgba(77,159,255,0.3)",
                           "borderRadius": "6px", "padding": "4px 12px",
                           "fontFamily": FONT_SERIF},
                ),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginBottom": "14px"}),
            table,
        ),
    ])


# ── Flask CSV download route ──────────────────────────────────────────────────

from flask import make_response  # noqa: E402 — Flask is a Dash dependency

@app.server.route("/download-catalog")
def _download_catalog():
    cat_path = ROOT / CFG["paths"]["master_catalog"]
    if not cat_path.exists():
        return "Catalog not found", 404
    cat = load_master_catalog(str(cat_path))
    buf = io.StringIO()
    cat.to_csv(buf, index=False)
    resp = make_response(buf.getvalue())
    resp.headers["Content-Type"]        = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=master_catalog.csv"
    return resp


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
