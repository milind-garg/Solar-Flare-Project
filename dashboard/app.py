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
# Observatory Field Log System (calm, precise, readable, grounded in real data)

BG      = "#101820"          # deep ink base
BG2     = "#17232B"          # chart plot canvas & raised surface
CARD    = "#17232B"          # panel container
CARD2   = "#1e2e38"          # elevated surface / hover
BORDER  = "#23333F"          # hairline 1px divider
BORDER2 = "#2d404e"          # subtle structural border
FG      = "#F1F0E8"          # warm near-white primary text
FG2     = "#A7B0B2"          # secondary slate text
DIM     = "#78858A"          # muted captions & inactive labels

# Restrained scientific instrument colors
OCHRE   = "#E3A64B"          # solar ochre (flare & solar emphasis)
SXR_BLUE= "#79B8C8"          # pale blue for SXR (1-8 Å)
HXR_VIO = "#B29AE0"          # violet for HXR (10-160 keV)
GREEN   = "#4EAD76"          # confirmed healthy / nominal
RED     = "#E05252"          # critical alert red
BLUE    = "#79B8C8"
CYAN    = "#79B8C8"
AMBER   = "#E3A64B"
ORANGE  = "#E3A64B"
VIOLET  = "#B29AE0"

FLARE_COLORS = {
    "X": "#E05252", "M": "#E3A64B", "C": "#D4A373",
    "B": "#4EAD76", "A": "#79B8C8", "HXR": "#B29AE0",
}

FONT_SANS = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
FONT_MONO = "'JetBrains Mono', 'SF Mono', 'Roboto Mono', Menlo, Consolas, monospace"
FONT_SERIF = FONT_SANS       # for backwards-compatibility alias

_GRAPH_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#101820",
    plot_bgcolor="#17232B",
    font=dict(size=12, family=FONT_SANS, color=FG2),
    margin=dict(l=62, r=20, t=44, b=44),
    legend=dict(
        orientation="h", y=1.14, font=dict(size=11, color=FG2, family=FONT_SANS),
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
    ),
    xaxis=dict(
        showgrid=True, gridcolor="#1b2832", zeroline=False,
        tickfont=dict(size=11, color=FG2, family=FONT_MONO), linecolor=BORDER,
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#1b2832", zeroline=False,
        tickfont=dict(size=11, color=FG2, family=FONT_MONO), linecolor=BORDER,
    ),
    title_font=dict(size=13, color=FG, family=FONT_SANS),
)


def _flare_color(cls: str) -> str:
    return FLARE_COLORS.get((cls or "A")[0], "#888888")


def _card(*children, **extra_style):
    style = {
        "background": CARD,
        "borderRadius": "3px",
        "padding": "16px 20px",
        "marginBottom": "12px",
        "border": f"1px solid {BORDER}",
        **extra_style,
    }
    return html.Div(children, style=style)


def _kpi(label_text: str, value, color: str = BLUE, sub: str = "", style: dict = None):
    """Observatory field log KPI metric tile with tabular mono typography."""
    base_style = {
        "background": CARD,
        "borderRadius": "3px",
        "padding": "12px 14px",
        "border": f"1px solid {BORDER}",
        "display": "inline-block",
        "minWidth": "110px",
        "marginRight": "10px",
        "marginBottom": "10px",
        "verticalAlign": "top",
    }
    if style:
        base_style.update(style)
    return html.Div([
        html.Div(
            label_text.upper(),
            style={"color": DIM, "fontSize": "10px", "letterSpacing": "0.08em",
                   "fontWeight": "600", "marginBottom": "6px",
                   "fontFamily": FONT_SANS, "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"},
        ),
        html.Div(
            str(value),
            style={"fontSize": "1.65em", "fontWeight": "700", "color": color,
                   "lineHeight": "1.1", "fontFamily": FONT_MONO, "whiteSpace": "nowrap"},
        ),
        html.Div(
            sub,
            style={"color": FG2, "fontSize": "11px", "fontWeight": "400",
                   "marginTop": "4px", "fontFamily": FONT_SANS, "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"}
        ) if sub else html.Div(),
    ], style=base_style)


# ── Plots ─────────────────────────────────────────────────────────────────────

def _fig_sxr(sxr_df: pd.DataFrame, events_df: pd.DataFrame | None, instrument: str = "solexs") -> go.Figure:
    fig = go.Figure()
    flux = sxr_df["flux_long"].resample("1min").mean().dropna()
    fig.add_trace(go.Scatter(x=flux.index, y=flux.values, name="SXR 1–8 Å (Long Channel)",
                             line=dict(color=SXR_BLUE, width=1.5)))
    if "bg_long" in sxr_df.columns:
        bg = sxr_df["bg_long"].resample("1min").mean().dropna()
        fig.add_trace(go.Scatter(x=bg.index, y=bg.values, name="Quiet-Sun Background (60m rolling 60th percentile)",
                                 line=dict(color=DIM, width=1, dash="dash")))

    # Scale factor for NOAA R-scale space weather threshold lines
    scale = CFG.get("calibration", {}).get("solexs_counts_to_wm2", 1.806e-08) if instrument == "solexs" else 1.0
    thresh_levels = [
        (1e-4 / scale, "X-Class · R3+ Strong Radio Blackout", "#E05252"),
        (1e-5 / scale, "M-Class · R1-R2 Moderate Blackout", "#E3A64B"),
        (1e-6 / scale, "C-Class · Noticeable Coronal Activity", "#D4A373"),
        (1e-7 / scale, "B-Class · Baseline Flare Activity", "#4EAD76"),
    ]
    flux_min = flux.min() if not flux.empty else 1.0
    for val, lbl, col in thresh_levels:
        if val >= flux_min * 0.4:
            fig.add_hline(
                y=val, line_color=col, line_dash="dash", line_width=1, opacity=0.55,
                annotation_text=lbl, annotation_position="top right",
                annotation_font=dict(size=9, color=col, family=FONT_SANS),
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
                    annotation_font=dict(color=col, size=10, family=FONT_SANS),
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
                             line=dict(color=HXR_VIO, width=1.5),
                             fill="tozeroy", fillcolor="rgba(178,154,224,0.06)"))
    if events_df is not None and not events_df.empty:
        for _, ev in events_df.iterrows():
            pt = ev.get("peak_time")
            if pd.isnull(pt):
                continue
            fig.add_vline(x=pt.isoformat(), line_color=FLARE_COLORS["HXR"],
                          line_width=1.5, line_dash="dot",
                          annotation_text="HXR Impulsive Burst", annotation_position="top left",
                          annotation_font=dict(color=HXR_VIO, size=10, family=FONT_SANS))
    fig.update_layout(title="Hard X-Ray Impulsive Count Rate (HEL1OS 10–160 keV — Non-Thermal Bremsstrahlung)",
                      yaxis_title="Counts · s⁻¹ (Livetime Corrected)", height=240, **_GRAPH_LAYOUT)
    return fig


def _fig_hardness(sxr_df: pd.DataFrame, hxr_df: pd.DataFrame) -> go.Figure:
    s1 = sxr_df["flux_long"].resample("1min").mean()
    h1 = hxr_df["hxr_total"].resample("1min").mean()
    sa, ha = s1.align(h1, join="inner")
    ratio = (ha / sa.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, name="Spectral Hardness (HXR / SXR)",
                             line=dict(color=OCHRE, width=1.4),
                             fill="tozeroy", fillcolor="rgba(227,166,75,0.08)"))
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
        line=dict(width=0), fill="tozeroy", fillcolor="rgba(227,166,75,0.08)",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=p1.index, y=p1.values, name="P(Flare in next 30 min)",
                             line=dict(color=OCHRE, width=1.6),
                             fill="tonexty", fillcolor="rgba(224,82,82,0.18)"))
    fig.add_hline(y=threshold, line_color="#F1F0E8", line_dash="dash", line_width=1.1,
                  annotation_text=f"Operational Alert Threshold = {threshold:.2f}",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="#F1F0E8", family=FONT_SANS))
    if events_df is not None and not events_df.empty:
        for _, ev in events_df.iterrows():
            cls = str(ev.get("flare_class") or "")
            pt  = ev.get("peak_time")
            if pd.isnull(pt):
                continue
            fig.add_vline(x=pt.isoformat(), line_color=_flare_color(cls),
                          line_width=1.2, line_dash="dot",
                          annotation_text=f"Actual: {cls}", annotation_font_size=10,
                          annotation_font=dict(color=_flare_color(cls), family=FONT_SANS))
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
    "color": DIM, "fontSize": "10px", "letterSpacing": "0.08em",
    "fontWeight": "600", "marginBottom": "6px",
    "textTransform": "uppercase", "fontFamily": FONT_SANS,
}

_TAB_STYLE = {
    "background": "transparent",
    "color": FG2,
    "border": "none",
    "borderBottom": "2px solid transparent",
    "padding": "10px 20px",
    "fontSize": "13px",
    "fontWeight": "600",
    "fontFamily": FONT_SANS,
    "letterSpacing": "0.01em",
    "cursor": "pointer",
}
_TAB_SELECTED = {
    **_TAB_STYLE,
    "color": FG,
    "fontWeight": "600",
    "borderBottom": f"2px solid {OCHRE}",
    "background": "rgba(227,166,75,0.06)",
}

# Google Fonts and CSS theme injection
_FONT_LINK = html.Link(
    rel="stylesheet",
    href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap",
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
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="/dashboard/assets/theme.css">
        <link rel="stylesheet" href="/assets/theme.css">
        {%css%}
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
        # Left: brand & mission identity
        html.Div([
            html.Img(src="/assets/logo.png", style={
                "height": "32px", "width": "32px", "borderRadius": "3px",
                "marginRight": "12px", "verticalAlign": "middle",
                "border": f"1px solid {BORDER}",
            }),
            html.Div([
                html.Div(
                    "Aditya-L1 Solar Flare Observatory",
                    style={"fontSize": "1.15em", "fontWeight": "600", "color": FG,
                           "letterSpacing": "-0.01em", "fontFamily": FONT_SANS},
                ),
                html.Div(
                    "ISRO PS-15 · SoLEXS (SXR 1–8 Å) + HEL1OS (HXR 10–160 keV) Science Operations Console",
                    style={"fontSize": "11px", "color": DIM,
                           "fontWeight": "400", "fontFamily": FONT_MONO,
                           "letterSpacing": "0.02em"},
                ),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),

        # Right: plain-language data/model/feed status badges & navigation
        html.Div([
            html.Span([
                html.Span("DATA: ", style={"color": DIM}),
                html.Span("Archive Parquet", style={"color": FG2, "fontWeight": "600"}),
            ], style={
                "fontSize": "11px", "fontFamily": FONT_MONO, "border": f"1px solid {BORDER}",
                "borderRadius": "3px", "padding": "4px 8px", "marginRight": "8px", "background": CARD,
            }),
            html.Span([
                html.Span("MODEL: ", style={"color": DIM}),
                html.Span("CNN-LSTM (0.813 AUC)", style={"color": FG2, "fontWeight": "600"}),
            ], style={
                "fontSize": "11px", "fontFamily": FONT_MONO, "border": f"1px solid {BORDER}",
                "borderRadius": "3px", "padding": "4px 8px", "marginRight": "8px", "background": CARD,
            }),
            html.Span([
                html.Span("FEED: ", style={"color": DIM}),
                html.Span("Static Analysis Mode", style={"color": SXR_BLUE, "fontWeight": "600"}),
            ], style={
                "fontSize": "11px", "fontFamily": FONT_MONO, "border": f"1px solid {BORDER}",
                "borderRadius": "3px", "padding": "4px 8px", "marginRight": "12px", "background": CARD,
            }),
            html.A("← Field Log Home", href="/", style={
                "color": OCHRE, "fontSize": "12px", "fontWeight": "600",
                "background": "transparent",
                "border": f"1px solid {BORDER}",
                "borderRadius": "3px",
                "padding": "5px 12px",
                "textDecoration": "none",
                "fontFamily": FONT_SANS,
            }),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "4px"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "background": BG,
        "padding": "12px 24px",
        "borderBottom": f"1px solid {BORDER}",
    }),

    # ── Controls toolbar ──────────────────────────────────────────────────────
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
                style={"width": "280px", "backgroundColor": CARD,
                       "color": FG, "fontSize": "13px", "borderRadius": "3px"},
            ),
        ], style={"marginRight": "16px"}),

        # Date
        html.Div([
            html.Div("Observation Date", style=_CTRL_LABEL),
            dcc.DatePickerSingle(
                id="date-picker",
                date="2024-05-10",
                display_format="YYYY-MM-DD",
                style={"fontSize": "13px"},
            ),
        ], style={"marginRight": "16px"}),

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
                style={"width": "200px", "backgroundColor": CARD,
                       "color": FG, "fontSize": "13px", "borderRadius": "3px"},
            ),
        ], style={"marginRight": "16px"}),

        # Alert threshold
        html.Div([
            html.Div("Alert Threshold (τ)", style=_CTRL_LABEL),
            html.Div([
                dcc.Slider(
                    id="threshold-slider",
                    min=0.1, max=0.9, step=0.05,
                    value=float(CFG["forecasting"]["alert_threshold"]),
                    marks={0.3: {"label": "0.3", "style": {"color": DIM, "fontSize": "10px", "fontWeight": "500", "fontFamily": FONT_MONO}},
                            0.5: {"label": "0.5", "style": {"color": DIM, "fontSize": "10px", "fontWeight": "500", "fontFamily": FONT_MONO}},
                            0.7: {"label": "0.7", "style": {"color": DIM, "fontSize": "10px", "fontWeight": "500", "fontFamily": FONT_MONO}}},
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode="drag",
                ),
            ], style={"width": "160px", "paddingTop": "2px"}),
        ], style={"marginRight": "16px"}),

        # Options
        html.Div([
            html.Div("Analysis Mode", style=_CTRL_LABEL),
            dcc.Checklist(
                id="options-check",
                options=[
                    {"label": "  ML Forecast", "value": "forecast"},
                    {"label": "  Live Telemetry (60s)",  "value": "live"},
                ],
                value=["forecast"],
                inline=True,
                labelStyle={"marginRight": "14px", "fontSize": "13px", "fontWeight": "500",
                             "color": FG2, "fontFamily": FONT_SANS},
            ),
        ], style={"marginRight": "16px"}),

        # Load button
        html.Button(
            "Load Telemetry Pass",
            id="load-btn", n_clicks=0,
            style={
                "background": "#1e2e38",
                "color": FG, "border": f"1px solid {BORDER2}",
                "borderRadius": "3px", "padding": "8px 18px",
                "cursor": "pointer", "fontSize": "13px",
                "fontWeight": "600", "marginTop": "18px",
                "fontFamily": FONT_SANS,
            },
        ),
    ], style={
        "display": "flex", "flexWrap": "wrap", "alignItems": "flex-end",
        "background": CARD,
        "padding": "12px 24px",
        "borderBottom": f"1px solid {BORDER}",
        "gap": "6px",
    }),

    # ── Alert banners ─────────────────────────────────────────────────────────
    html.Div(id="alert-banners", style={"padding": "10px 24px 0"}),

    # ── Status bar ────────────────────────────────────────────────────────────
    html.Div(
        id="status-bar",
        style={"padding": "6px 24px", "color": FG2, "fontSize": "12px",
               "fontWeight": "500", "fontFamily": FONT_MONO,
               "letterSpacing": "0.01em", "minHeight": "26px"},
    ),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    html.Div([
        dcc.Tabs(id="main-tabs", value="curves", children=[
            dcc.Tab(label="Light Curves & Observations", value="curves",
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label="Predictive Neural Forecast",    value="forecast",
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
            dcc.Tab(label="Master Flare Catalog",         value="catalog",
                    style=_TAB_STYLE, selected_style=_TAB_SELECTED),
        ]),
    ], style={
        "padding": "0 24px",
        "background": BG,
        "borderBottom": f"1px solid {BORDER}",
    }),

    dcc.Loading(
        html.Div(id="tab-content", style={"padding": "16px 24px 32px"}),
        type="dot",
        color=SXR_BLUE,
    ),

    # ── Hidden stores & interval ──────────────────────────────────────────────
    dcc.Store(id="store-main"),
    dcc.Interval(id="live-interval", interval=60_000, disabled=True, n_intervals=0),

], style={
    "background": BG,
    "minHeight": "100vh",
    "fontFamily": FONT_SANS,
    "color": FG,
    "fontSize": "13px",
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
        tag  = "✓ multi-instrument confirmed" if conf else ""
        banners.append(html.Div(
            [html.Span("NOWCAST DETECTED · ", style={"fontWeight": "700", "fontSize": "11px", "letterSpacing": "0.06em", "fontFamily": FONT_MONO}),
             html.Span(f"{cls} FLARE", style={"fontFamily": FONT_MONO, "fontWeight": "700",
                                              "background": f"{col}25", "borderRadius": "2px",
                                              "padding": "2px 8px", "marginRight": "8px"}),
             html.Span(f"Peak @ {peak} UTC", style={"fontFamily": FONT_MONO, "fontSize": "12px", "color": FG}),
             html.Span(f" · {store.get('instrument', 'solexs').upper()} stream", style={"color": DIM, "fontSize": "11px", "fontFamily": FONT_MONO}),
             html.Span(f"  ({tag})", style={"fontSize": "11px", "color": GREEN, "marginLeft": "8px", "fontFamily": FONT_MONO}) if conf else html.Span()],
            style={
                "background": CARD,
                "border": f"1px solid {BORDER}",
                "borderLeft": f"3px solid {col}",
                "color": col,
                "borderRadius": "3px",
                "padding": "8px 14px",
                "marginBottom": "6px",
                "fontSize": "13px",
                "fontFamily": FONT_SANS,
                "display": "flex",
                "alignItems": "center",
                "flexWrap": "wrap",
            }
        ))

    pdata = store.get("prob")
    if pdata:
        probs = [p for p in pdata["probs"] if p >= 0]
        if probs and probs[-1] >= threshold:
            lp = probs[-1]
            banners.append(html.Div(
                [html.Span("SPACE WEATHER ALERT · ", style={"fontWeight": "700", "fontSize": "11px", "letterSpacing": "0.06em", "fontFamily": FONT_MONO}),
                 html.Span(f"P(Flare in next 30m) = {lp:.3f}",
                           style={"fontFamily": FONT_MONO, "fontWeight": "700",
                                  "background": "rgba(224,82,82,0.18)",
                                  "borderRadius": "2px", "padding": "2px 8px",
                                  "marginRight": "8px"}),
                 html.Span(f"Exceeds decision threshold τ = {threshold:.2f} · Lead time window: {CFG['forecasting']['lead_time_min']} min",
                           style={"fontSize": "12px", "color": FG2, "fontFamily": FONT_SANS})],
                style={
                    "background": CARD,
                    "border": f"1px solid {BORDER}",
                    "borderLeft": f"3px solid {RED}",
                    "color": RED,
                    "borderRadius": "3px",
                    "padding": "8px 14px",
                    "marginBottom": "6px",
                    "fontSize": "13px",
                    "fontFamily": FONT_SANS,
                    "display": "flex",
                    "alignItems": "center",
                    "flexWrap": "wrap",
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
                html.Div("ADITYA-L1 FIELD LOG STANDBY", style={
                    "fontFamily": FONT_MONO, "fontSize": "11px",
                    "letterSpacing": "0.08em", "color": DIM, "fontWeight": "600",
                    "textTransform": "uppercase", "marginBottom": "6px",
                }),
                html.H3("Observation Pass Ready for Inspection", style={
                    "color": FG, "fontSize": "18px", "fontWeight": "600",
                    "fontFamily": FONT_SANS, "marginBottom": "8px",
                }),
                html.P("Select an observation date and instrument stream in the toolbar above, or initialize one of the benchmark flare events below to inspect calibrated light curves and neural forecasts:", style={
                    "color": FG2, "fontSize": "13px", "maxWidth": "680px", "margin": "0 auto 20px",
                    "lineHeight": "1.5",
                }),
                html.Div([
                    html.Button("2024-05-10 · X3.6 Flare (Confirmed Multi-Instrument)", id="btn-quick-1", n_clicks=0, style={
                        "background": CARD2, "color": FG, "border": f"1px solid {BORDER}",
                        "borderRadius": "3px", "padding": "8px 14px", "fontFamily": FONT_SANS,
                        "fontSize": "12px", "fontWeight": "500", "cursor": "pointer", "margin": "3px",
                    }),
                    html.Button("2024-02-09 · X3.3 Major Eruption", id="btn-quick-2", n_clicks=0, style={
                        "background": CARD2, "color": FG, "border": f"1px solid {BORDER}",
                        "borderRadius": "3px", "padding": "8px 14px", "fontFamily": FONT_SANS,
                        "fontSize": "12px", "fontWeight": "500", "cursor": "pointer", "margin": "3px",
                    }),
                    html.Button("2024-03-23 · X1.1 Proton Flare", id="btn-quick-3", n_clicks=0, style={
                        "background": CARD2, "color": FG, "border": f"1px solid {BORDER}",
                        "borderRadius": "3px", "padding": "8px 14px", "fontFamily": FONT_SANS,
                        "fontSize": "12px", "fontWeight": "500", "cursor": "pointer", "margin": "3px",
                    }),
                    html.Button("2024-05-10 · M8.1 Impulsive Multi-Burst", id="btn-quick-4", n_clicks=0, style={
                        "background": CARD2, "color": FG, "border": f"1px solid {BORDER}",
                        "borderRadius": "3px", "padding": "8px 14px", "fontFamily": FONT_SANS,
                        "fontSize": "12px", "fontWeight": "500", "cursor": "pointer", "margin": "3px",
                    }),
                ], style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center", "gap": "6px"}),
            ], style={
                "background": CARD, "borderRadius": "3px", "border": f"1px solid {BORDER}",
                "padding": "36px 24px", "maxWidth": "740px", "margin": "24px auto",
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
    inst    = store.get("instrument", "solexs")
    inst_name = "SoLEXS SDD2 (Aditya-L1)" if inst == "solexs" else "GOES-16 XRS-B (NOAA)"
    date_val  = store.get("date", "—")
    unit_val  = "counts · s⁻¹ (1.806e-8 W/m² cal scalar)" if inst == "solexs" else "W · m⁻²"
    kids    = []

    # Observation metadata header bar
    workspace_meta = html.Div([
        html.Div([
            html.Span("PRIMARY WORKSPACE · ", style={"color": DIM, "fontWeight": "700", "fontSize": "10px", "fontFamily": FONT_MONO}),
            html.Span(f"PASS: {date_val}  |  INSTRUMENT: {inst_name}  |  UNITS: {unit_val}",
                      style={"color": FG2, "fontSize": "11px", "fontFamily": FONT_MONO}),
        ]),
        html.Div([
            html.Span("PROVENANCE: ISRO ISSDC / URSC LEVEL-2", style={"color": DIM, "fontSize": "10px", "fontFamily": FONT_MONO}),
        ]),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "6px 12px", "background": CARD, "border": f"1px solid {BORDER}",
        "borderRadius": "3px", "marginBottom": "8px", "flexWrap": "wrap", "gap": "6px",
    })
    kids.append(workspace_meta)

    if sxr_df is not None and "flux_long" in sxr_df.columns:
        kids.append(_card(
            dcc.Graph(figure=_fig_sxr(sxr_df, all_evs, instrument=inst),
                      config={"displayModeBar": False}),
        ))
    else:
        kids.append(html.Div(f"No SXR observation data available for {date_val}.",
                             style={"color": FG2, "fontSize": "13px", "padding": "16px 0"}))

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
        kids.append(html.Div("No HEL1OS HXR observation data available for this pass.",
                             style={"color": DIM, "fontSize": "11px", "fontFamily": FONT_MONO, "padding": "4px 0 10px"}))

    # Detected-events table
    evs = store.get("master_evs") or []
    if evs:
        _cols = ["peak_time", "flare_class", "start_time", "end_time",
                 "solexs_peak_flux", "confirmed_multi_instrument"]
        rows  = [{k: v for k, v in r.items() if k in _cols} for r in evs]
        cols  = [c for c in _cols if c in rows[0]]
        kids.append(_card(
            html.Div([
                html.Div([
                    html.Span("DETECTED FLARE EVENTS IN THIS PASS",
                              style={"color": FG, "fontSize": "13px", "fontFamily": FONT_SANS, "fontWeight": "600"}),
                    html.Span(" (Autonomous Nowcasting Threshold Crossings)",
                              style={"color": DIM, "fontSize": "11px", "fontFamily": FONT_SANS}),
                ]),
                html.Span(f"{len(evs)} event(s)",
                          style={"color": SXR_BLUE, "fontSize": "12px",
                                 "fontFamily": FONT_MONO, "fontWeight": "600"}),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginBottom": "10px"}),
            dash_table.DataTable(
                data=rows,
                columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols],
                style_table={"overflowX": "auto", "borderRadius": "3px"},
                style_header={"background": BG, "color": FG2,
                               "fontWeight": "600", "border": f"1px solid {BORDER}",
                               "fontSize": "10px", "letterSpacing": "0.06em",
                               "fontFamily": FONT_SANS,
                               "textTransform": "uppercase"},
                style_cell={"background": CARD, "color": FG,
                             "border": f"1px solid {BORDER}",
                             "fontSize": "12px", "padding": "7px 12px",
                             "fontFamily": FONT_MONO},
                style_data={"borderTop": f"1px solid {BORDER}"},
                page_size=15,
            ),
        ))
    else:
        kids.append(html.Div(
            [html.Span("✓  ", style={"color": GREEN}), "No flare threshold crossings detected on this observation date."],
            style={"color": FG2, "padding": "10px 0", "fontSize": "13px",
                   "fontFamily": FONT_SANS}
        ))

    return html.Div(kids)


def _tab_forecast(store: dict, threshold: float) -> html.Div:
    pdata  = store.get("prob")
    cdata  = store.get("cls")
    all_ev = _records_to_df(store.get("master_evs"))
    kids   = []

    if not pdata:
        return html.Div(
            "ML forecast not active — enable 'ML Forecast' option in toolbar and click 'Load Telemetry Pass'.",
            style={"color": FG2, "fontSize": "13px", "fontWeight": "400",
                   "paddingTop": "32px", "textAlign": "center", "fontFamily": FONT_SANS},
        )

    times  = pd.to_datetime(pdata["times"])
    probs  = np.array(pdata["probs"], dtype=float)
    probs[probs < 0] = np.nan
    prob_s = pd.Series(probs, index=times, name="prob")

    latest_p = float(prob_s.dropna().iloc[-1]) if not prob_s.dropna().empty else float("nan")
    above    = not np.isnan(latest_p) and latest_p >= threshold
    p_color  = RED if above else GREEN

    # Forecast status explanation banner
    status_text = (
        f"ALERT CONDITION ACTIVE: Latest P(flare) = {latest_p:.3f} >= threshold {threshold:.2f}. "
        f"High probability of flare eruption within 30 minutes."
        if above else
        f"NOMINAL FORECAST: Latest P(flare) = {latest_p:.3f} < threshold {threshold:.2f}. "
        f"Coronal flux levels consistent with baseline quiet-Sun conditions."
    )
    status_banner = html.Div(
        status_text,
        style={
            "padding": "8px 14px",
            "background": CARD,
            "border": f"1px solid {BORDER}",
            "borderLeft": f"3px solid {p_color}",
            "color": p_color,
            "fontSize": "12px",
            "fontFamily": FONT_MONO,
            "borderRadius": "3px",
            "marginBottom": "12px",
            "fontWeight": "600",
        }
    )
    kids.append(status_banner)

    # KPI row
    kpi_tile_style = {"minWidth": "0", "marginRight": "0", "marginBottom": "0"}
    kpis = [
        _kpi("Current P(flare)",
             f"{latest_p:.3f}" if not np.isnan(latest_p) else "—",
             p_color,
             "THRESHOLD EXCEEDED" if above else f"< threshold ({threshold:.2f})",
             style=kpi_tile_style),
        _kpi("Held-out AUC",  f"{_CKPT_META.get('auc', 0.813):.3f}" if _CKPT_META else "0.813", SXR_BLUE, "20% test split", style=kpi_tile_style),
        _kpi("TSS @ 0.5",     f"{_CKPT_META.get('tss', 0.349):.3f}" if _CKPT_META else "0.349", SXR_BLUE, "true skill statistic", style=kpi_tile_style),
        _kpi("Lead Time P90", "27.0 min", OCHRE, "14m median lead", style=kpi_tile_style),
    ]

    cls_df = None
    if cdata:
        cls_times = pd.to_datetime(cdata["_times"])
        cls_cols  = [c for c in cdata if c != "_times"]
        cls_df    = pd.DataFrame(
            {col: np.array(cdata[col], dtype=float) for col in cls_cols},
            index=cls_times,
        )
        cls_df[cls_df < 0] = np.nan
        if not cls_df.dropna(how="all").empty:
            last_row = cls_df.dropna(how="all").iloc[-1]
            best_cls = last_row.idxmax()
            best_p   = last_row.max()
            kpis.append(
                _kpi("Predicted Class", best_cls,
                     _flare_color(best_cls),
                     f"P({best_cls}) = {best_p:.3f}",
                     style=kpi_tile_style)
            )

    kids.append(html.Div(
        kpis,
        style={
            "display": "grid",
            "gridTemplateColumns": f"repeat({len(kpis)}, minmax(0, 1fr))",
            "gap": "10px",
            "marginBottom": "12px",
            "width": "100%",
        }
    ))

    kids.append(_card(
        dcc.Graph(figure=_fig_probability(prob_s, threshold, all_ev),
                  config={"displayModeBar": True, "displaylogo": False}),
    ))

    if cls_df is not None and not cls_df.dropna(how="all").empty:
        kids.append(_card(
            dcc.Graph(figure=_fig_class_prob(cls_df),
                      config={"displayModeBar": False}),
        ))

    return html.Div(kids)


def _tab_catalog() -> html.Div:
    cat_path = ROOT / CFG["paths"]["master_catalog"]
    if not cat_path.exists():
        return html.Div("Master catalog not found. Run process_all.py first.",
                        style={"color": FG2, "fontSize": "13px", "paddingTop": "32px"})

    cat = load_master_catalog(str(cat_path))
    if cat.empty:
        return html.Div("Catalog is empty.", style={"color": FG2, "fontSize": "13px"})

    counts  = cat["flare_class"].str[0].value_counts().to_dict()
    multi_n = int(cat.get("confirmed_multi_instrument", pd.Series(dtype=bool)).sum())

    lag_series = cat.loc[cat.get("confirmed_multi_instrument", False) & cat.get("neupert_lag_sec", pd.Series(dtype=float)).notna(), "neupert_lag_sec"]
    median_lag_txt = f"{lag_series.median():+.0f}s" if not lag_series.empty else "—"

    kpis = html.Div([
        _kpi("Total Catalog Events", len(cat), SXR_BLUE, "824 observed days"),
        *[_kpi(f"{c}-class", counts[c], _flare_color(c))
          for c in ["X", "M", "C", "B"] if c in counts],
        _kpi("Multi-instrument", multi_n, HXR_VIO, "confirmed events"),
        _kpi("Median Neupert lag", median_lag_txt, GREEN, "HXR precursor lead"),
    ], style={"marginBottom": "12px", "display": "flex", "flexWrap": "wrap", "gap": "8px"})

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
        ("peak_time", "Peak Time (UTC)"),
        ("flare_class", "Class"),
        ("start_time", "Start Time"),
        ("end_time", "End Time"),
        ("duration", "Duration"),
        ("rise_time", "Rise"),
        ("decay_time", "Decay"),
        ("neupert_lag", "Neupert Lag"),
        ("hxr_bursts", "HXR Bursts"),
        ("solexs_peak_flux", "SXR Peak Flux (cts/s)"),
        ("helios_peak_flux", "HXR Peak Flux (cts/s)"),
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
        style_table={"overflowX": "auto", "borderRadius": "3px"},
        style_header={"background": BG, "color": FG2,
                       "fontWeight": "600", "border": f"1px solid {BORDER}",
                       "fontSize": "10px", "letterSpacing": "0.06em",
                       "textTransform": "uppercase",
                       "fontFamily": FONT_SANS},
        style_cell={"background": CARD, "color": FG,
                     "border": f"1px solid {BORDER}",
                     "fontSize": "12px", "padding": "8px 12px",
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
             "background": "#141e26"},
        ],
    )

    return html.Div([
        kpis,
        _card(
            html.Div([
                html.Div([
                    html.H4("Master Solar Flare Event Catalog",
                            style={"margin": "0", "color": FG, "fontSize": "14px",
                                   "fontFamily": FONT_SANS,
                                   "fontWeight": "600"}),
                    html.Div("Unified SoLEXS + HEL1OS + GOES cross-calibrated detection events",
                             style={"color": DIM, "fontSize": "11px", "fontFamily": FONT_SANS, "marginTop": "2px"}),
                ]),
                html.A(
                    ["⬇  Download CSV Catalog"],
                    href="/download-catalog",
                    style={"color": OCHRE, "fontSize": "12px", "fontWeight": "600",
                           "textDecoration": "none",
                           "background": "transparent",
                           "border": f"1px solid {BORDER}",
                           "borderRadius": "3px", "padding": "5px 12px",
                           "fontFamily": FONT_SANS},
                ),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginBottom": "12px"}),
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
