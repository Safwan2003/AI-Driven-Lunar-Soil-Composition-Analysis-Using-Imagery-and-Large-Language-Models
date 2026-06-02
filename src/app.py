"""
SUPARCO AI Soil Composition Analysis — Streamlit Exhibition App
Run: streamlit run src/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.inference import LunarAnalysisPipeline
from src.llm_reporter import SoilLLMReporter, SAFE_THRESHOLDS, ELEMENT_FULL
from src.model import ELEMENTS, TERRAIN_CLASSES
from src.dataset import ELEMENT_UNITS

import io
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
COMP_MODEL   = ROOT / 'models' / 'composition_model.pth'
TERR_MODEL   = ROOT / 'models' / 'terrain_classifier.pth'
METRICS_FILE = ROOT / 'models' / 'training_metrics.json'
SOIL_DIR     = ROOT / 'Soil_data_Academia-20260601T122708Z-3-001' / 'Soil_data_Academia'
EXCEL_PATH   = SOIL_DIR / 'Soil_Analysis.xlsx'
IMG_DIR      = SOIL_DIR / 'images'
CLASSES_IMG  = ROOT / 'classses_terrain.png'

TERRAIN_ICONS = {
    'Rocky Region': '🪨',
    'Crater':       '🕳️',
    'Big Rock':     '🗿',
    'Artifact':     '🛸',
}

ELEMENT_COLORS = {
    'Cd': '#6366f1',  # indigo
    'Cu': '#f59e0b',  # amber
    'Ni': '#10b981',  # emerald
    'Mn': '#3b82f6',  # blue
    'Fe': '#ef4444',  # red
    'Zn': '#8b5cf6',  # violet
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SUPARCO · Lunar Soil AI",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base & Background ───────────────────────────────────────── */
  [data-testid="stAppViewContainer"] {
      background: linear-gradient(160deg, #f0f4ff 0%, #fafbff 50%, #f5f0ff 100%);
      min-height: 100vh;
  }

  /* ── Global text color fix — all custom HTML in main area ──── */
  [data-testid="stMain"] div,
  [data-testid="stMain"] span,
  [data-testid="stMain"] p,
  [data-testid="stMain"] b,
  [data-testid="stMain"] small,
  [data-testid="stMain"] li {
      color: #1e293b;
  }

  /* ── Sidebar (override global above for dark sidebar) ────────── */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 60%, #0f172a 100%) !important;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stMarkdown h2 { color: #a5b4fc !important; }
  [data-testid="stSidebar"] .stMarkdown h3 { color: #c7d2fe !important; }
  [data-testid="stSidebar"] hr { border-color: #334155 !important; }

  /* ── Hero Banner ─────────────────────────────────────────────── */
  .hero-banner {
      background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #312e81 70%, #1e3a5f 100%);
      padding: 2.2rem 2.5rem;
      border-radius: 20px;
      margin-bottom: 1.8rem;
      position: relative;
      overflow: hidden;
      box-shadow: 0 8px 32px rgba(99,102,241,0.25);
  }
  .hero-banner::before {
      content: '✦ ·  · ✦  ·    ·  ✦ ·   · ✦     ✦  ·  ✦   ·  ✦ ·  ✦';
      position: absolute;
      top: 12px; left: 0; right: 0;
      font-size: 0.6rem;
      color: rgba(255,255,255,0.18);
      letter-spacing: 6px;
      text-align: center;
  }
  .hero-title {
      font-size: 2.1rem;
      font-weight: 800;
      color: #ffffff;
      margin: 0 0 0.3rem 0;
      letter-spacing: -0.5px;
  }
  .hero-subtitle {
      font-size: 1rem;
      color: #a5b4fc;
      margin: 0 0 1rem 0;
  }
  .hero-tags {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
  }
  .hero-tag {
      background: rgba(255,255,255,0.10);
      border: 1px solid rgba(165,180,252,0.35);
      color: #c7d2fe;
      padding: 0.25rem 0.75rem;
      border-radius: 999px;
      font-size: 0.78rem;
      font-weight: 600;
      backdrop-filter: blur(4px);
  }
  .uni-badges {
      display: flex;
      gap: 0.8rem;
      margin-top: 1rem;
      flex-wrap: wrap;
  }
  .uni-badge {
      background: rgba(99,102,241,0.20);
      border: 1px solid rgba(165,180,252,0.4);
      color: #e0e7ff;
      padding: 0.3rem 1rem;
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.3px;
  }

  /* ── Glass Cards ─────────────────────────────────────────────── */
  .glass-card {
      background: rgba(255,255,255,0.80);
      border: 1px solid rgba(199,210,254,0.5);
      border-radius: 16px;
      padding: 1.4rem 1.6rem;
      box-shadow: 0 4px 24px rgba(99,102,241,0.08);
      backdrop-filter: blur(8px);
      margin-bottom: 1rem;
  }
  .glass-card-dark {
      background: linear-gradient(135deg, #0f172a, #1e1b4b);
      border: 1px solid rgba(165,180,252,0.25);
      border-radius: 16px;
      padding: 1.4rem 1.6rem;
      box-shadow: 0 4px 24px rgba(0,0,0,0.25);
      color: white;
      margin-bottom: 1rem;
  }

  /* ── Metric Element Cards ────────────────────────────────────── */
  .elem-card {
      background: white;
      border-left: 6px solid #6366f1;
      border-radius: 12px;
      padding: 0.7rem 1rem;
      margin: 0.35rem 0;
      box-shadow: 0 2px 12px rgba(0,0,0,0.07);
      transition: transform 0.15s, box-shadow 0.15s;
      color: #1e293b !important;
      font-family: inherit;
  }
  .elem-card:hover { transform: translateX(4px); box-shadow: 0 4px 18px rgba(0,0,0,0.12); }
  .elem-normal   { border-left-color: #10b981 !important; background: linear-gradient(135deg, #f0fdf4, #ecfdf5) !important; }
  .elem-elevated { border-left-color: #f59e0b !important; background: linear-gradient(135deg, #fffbeb, #fef9c3) !important; }
  .elem-critical { border-left-color: #ef4444 !important; background: linear-gradient(135deg, #fef2f2, #fee2e2) !important; }
  .e-sym  { font-size: 1rem; font-weight: 900; letter-spacing: -0.3px; }
  .e-name { font-size: 0.88rem; font-weight: 600; color: #334155 !important; }
  .e-val  { font-size: 1.1rem; font-weight: 800; color: #0f172a !important; }
  .e-unit { font-size: 0.72rem; font-weight: 500; color: #64748b !important; }
  .e-lim  { font-size: 0.71rem; color: #94a3b8 !important; margin-top: 0.15rem; }
  .e-pill-safe     { background:#dcfce7; color:#15803d !important; border-radius:999px; padding:0.15rem 0.55rem; font-size:0.72rem; font-weight:700; }
  .e-pill-elevated { background:#fef9c3; color:#92400e !important; border-radius:999px; padding:0.15rem 0.55rem; font-size:0.72rem; font-weight:700; }
  .e-pill-critical { background:#fee2e2; color:#991b1b !important; border-radius:999px; padding:0.15rem 0.55rem; font-size:0.72rem; font-weight:700; }

  /* ── Terrain Badge ───────────────────────────────────────────── */
  .terrain-pill {
      display: inline-block;
      background: linear-gradient(135deg, #4f46e5, #7c3aed);
      color: white;
      padding: 0.55rem 1.4rem;
      border-radius: 999px;
      font-size: 1.15rem;
      font-weight: 800;
      box-shadow: 0 4px 16px rgba(99,102,241,0.4);
      letter-spacing: 0.3px;
      margin: 0.4rem 0 0.8rem 0;
  }
  .conf-label {
      font-size: 0.78rem;
      color: #64748b !important;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
  }
  .prob-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.3rem 0;
      border-bottom: 1px solid #e2e8f0;
      font-size: 0.87rem;
      color: #1e293b !important;
  }
  .prob-row span { color: #1e293b !important; }
  .prob-val { font-weight: 800 !important; color: #4f46e5 !important; font-size: 0.9rem; }

  /* ── Section Headers ─────────────────────────────────────────── */
  .section-title {
      font-size: 1.15rem;
      font-weight: 800;
      color: #1e293b;
      letter-spacing: -0.3px;
      margin-bottom: 0.8rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
  }
  .divider {
      height: 2px;
      background: linear-gradient(90deg, #6366f1, #8b5cf6, transparent);
      border: none;
      border-radius: 2px;
      margin: 1.8rem 0;
  }

  /* ── Upload Zone ─────────────────────────────────────────────── */
  [data-testid="stFileUploader"] {
      background: rgba(255,255,255,0.7) !important;
      border: 2px dashed #a5b4fc !important;
      border-radius: 16px !important;
      padding: 1rem !important;
  }

  /* ── Tab styling ─────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
      gap: 6px;
      background: rgba(255,255,255,0.6);
      padding: 4px 8px;
      border-radius: 12px;
      border: 1px solid rgba(199,210,254,0.5);
      backdrop-filter: blur(6px);
  }
  .stTabs [data-baseweb="tab"] {
      font-weight: 700;
      font-size: 0.9rem;
      border-radius: 10px;
      color: #475569;
      padding: 0.4rem 1rem;
  }
  .stTabs [aria-selected="true"] {
      background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
      color: white !important;
  }

  /* ── Sidebar model status chips ──────────────────────────────── */
  .status-chip {
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      padding: 0.25rem 0.7rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      margin: 0.2rem 0;
  }
  .status-ok  { background: rgba(16,185,129,0.2); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }
  .status-err { background: rgba(239,68,68,0.2);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }

  /* ── Report section ──────────────────────────────────────────── */
  .report-box {
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 16px;
      padding: 1.5rem 2rem;
      box-shadow: 0 4px 24px rgba(0,0,0,0.06);
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌕 SUPARCO AI")
    st.markdown("##### Lunar Soil Analysis System")
    st.markdown("---")
    st.image(str(CLASSES_IMG), caption="Terrain Classification Classes", use_container_width=True)
    st.markdown("---")
    st.markdown("### Model Status")
    comp_ok = COMP_MODEL.exists()
    terr_ok = TERR_MODEL.exists()
    st.markdown(
        f'<span class="status-chip {"status-ok" if comp_ok else "status-err"}">{"✅" if comp_ok else "❌"} Composition CNN</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="status-chip {"status-ok" if terr_ok else "status-err"}">{"✅" if terr_ok else "❌"} Terrain CNN</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    gemini_key = st.text_input(
        "🔑 Gemini API Key",
        type="password",
        placeholder="AIza… (optional, uses .env)",
        key="gemini_key",
    )
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;color:#94a3b8;line-height:1.6'>"
        "🏛️ Salim Habib University<br>"
        "🛰️ SUPARCO Collaboration<br>"
        "🎓 FYP Exhibition · June 2026"
        "</p>",
        unsafe_allow_html=True,
    )


# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return LunarAnalysisPipeline(
        composition_model_path=str(COMP_MODEL) if COMP_MODEL.exists() else None,
        terrain_model_path=str(TERR_MODEL) if TERR_MODEL.exists() else None,
    )

@st.cache_resource
def load_reporter(api_key):
    return SoilLLMReporter(api_key=api_key or None)

@st.cache_data
def load_dataset():
    if not EXCEL_PATH.exists():
        return None
    df = pd.read_excel(EXCEL_PATH)
    df['Sample ID'] = df['Sample ID'].astype(str).str.strip()
    return df

@st.cache_data
def load_metrics():
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            return json.load(f)
    return None

@st.cache_data
def get_dataset_stats(df):
    return {
        e: {
            'mean': float(df[e].mean()),
            'std':  float(df[e].std()),
            'min':  float(df[e].min()),
            'max':  float(df[e].max()),
        }
        for e in ELEMENTS
    }


# ── Helper functions ───────────────────────────────────────────────────────────
def report_to_pdf(markdown_text: str, terrain_class: str | None, predictions: dict) -> bytes:
    """Convert the markdown report + composition table to a styled PDF."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Title'],
        fontSize=20, textColor=colors.HexColor('#0f172a'),
        spaceAfter=6, alignment=TA_CENTER)
    sub_style = ParagraphStyle('Sub', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor('#475569'),
        spaceAfter=14, alignment=TA_CENTER)
    h1_style = ParagraphStyle('H1', parent=styles['Heading1'],
        fontSize=13, textColor=colors.HexColor('#312e81'),
        spaceBefore=14, spaceAfter=4)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
        fontSize=11, textColor=colors.HexColor('#4f46e5'),
        spaceBefore=10, spaceAfter=3)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=9.5, textColor=colors.HexColor('#1e293b'),
        spaceAfter=4, leading=14)
    small_style = ParagraphStyle('Small', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#64748b'),
        spaceAfter=2, leading=12)

    story = []

    # ── Cover header ──────────────────────────────────────────────────────
    story.append(Paragraph("🌕 SUPARCO AI Lunar Soil Analysis", title_style))
    story.append(Paragraph(
        "Salim Habib University &nbsp;·&nbsp; SUPARCO &nbsp;·&nbsp; FYP Exhibition 2026",
        sub_style,
    ))
    story.append(HRFlowable(width='100%', color=colors.HexColor('#6366f1'), thickness=2))
    story.append(Spacer(1, 0.3*cm))

    # ── Composition summary table ─────────────────────────────────────────
    story.append(Paragraph("Predicted Heavy Metal Composition", h1_style))
    if terrain_class:
        story.append(Paragraph(f"Terrain: <b>{terrain_class}</b>", body_style))

    tdata = [['Element', 'Symbol', 'Predicted (mg/kg)', 'Safe Limit (mg/kg)', 'Status']]
    row_colors = []
    for i, e in enumerate(ELEMENTS):
        v = predictions.get(e, 0)
        thresh = SAFE_THRESHOLDS.get(e, 1e9)
        if v > thresh * 2:
            status = 'Critical'; bg = colors.HexColor('#fee2e2')
        elif v > thresh:
            status = 'Elevated'; bg = colors.HexColor('#fef9c3')
        else:
            status = 'Safe';     bg = colors.HexColor('#dcfce7')
        tdata.append([ELEMENT_FULL[e], e, f'{v:.3f}', str(thresh), status])
        row_colors.append((i + 1, bg))

    t = Table(tdata, colWidths=[4.5*cm, 1.5*cm, 3.5*cm, 4*cm, 2.5*cm])
    ts = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#312e81')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ALIGN',      (2,1), (-1,-1), 'CENTER'),
        ('GRID',       (0,0), (-1,-1), 0.4, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ])
    for row_idx, bg_color in row_colors:
        ts.add('BACKGROUND', (4, row_idx), (4, row_idx), bg_color)
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width='100%', color=colors.HexColor('#e2e8f0'), thickness=1))

    # ── Markdown report body ──────────────────────────────────────────────
    story.append(Paragraph("AI Scientific Report", h1_style))
    for line in markdown_text.splitlines():
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.15*cm))
            continue
        # Strip markdown symbols and render with styles
        if line.startswith('### '):
            story.append(Paragraph(line[4:], h2_style))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], h1_style))
        elif line.startswith('# '):
            story.append(Paragraph(line[2:], h1_style))
        elif line.startswith('**') and line.endswith('**'):
            story.append(Paragraph(f'<b>{line[2:-2]}</b>', body_style))
        elif re.match(r'^\|.+\|$', line):
            pass  # skip markdown tables (already rendered above)
        elif line.startswith('---'):
            story.append(HRFlowable(width='100%', color=colors.HexColor('#e2e8f0'), thickness=0.5))
        else:
            clean = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
            clean = re.sub(r'\*(.+?)\*', r'<i>\1</i>', clean)
            story.append(Paragraph(clean, body_style))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width='100%', color=colors.HexColor('#6366f1'), thickness=1.5))
    story.append(Paragraph(
        "Generated by SUPARCO AI Lunar Soil Analysis System · Salim Habib University FYP 2026",
        small_style,
    ))

    doc.build(story)
    return buf.getvalue()


def contamination_class(element, value):
    thresh = SAFE_THRESHOLDS.get(element, 1e9)
    if value > thresh * 2:
        return 'elem-critical', 'e-pill-critical', '🔴 Critical'
    if value > thresh:
        return 'elem-elevated', 'e-pill-elevated', '🟠 Elevated'
    return 'elem-normal', 'e-pill-safe', '🟢 Safe'


def composition_bar_chart(predictions, dataset_stats):
    elements  = list(predictions.keys())
    predicted = [predictions[e] for e in elements]
    means     = [dataset_stats[e]['mean'] for e in elements]
    colors    = [ELEMENT_COLORS.get(e, '#6366f1') for e in elements]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='AI Prediction',
        x=[ELEMENT_FULL[e] for e in elements],
        y=predicted,
        marker=dict(
            color=colors,
            line=dict(width=0),
            opacity=0.9,
        ),
        text=[f'{v:.3f}' for v in predicted],
        textposition='outside',
        textfont=dict(size=11, color='#1e293b'),
    ))
    fig.add_trace(go.Scatter(
        name='Dataset Mean',
        x=[ELEMENT_FULL[e] for e in elements],
        y=means,
        mode='markers+lines',
        marker=dict(size=11, symbol='diamond', color='#10b981'),
        line=dict(dash='dot', color='#10b981', width=2),
    ))
    fig.add_trace(go.Scatter(
        name='Safe Threshold',
        x=[ELEMENT_FULL[e] for e in elements],
        y=[SAFE_THRESHOLDS.get(e, 0) for e in elements],
        mode='markers+lines',
        marker=dict(size=8, symbol='line-ew-open', color='#ef4444'),
        line=dict(dash='dash', color='#ef4444', width=1.8),
    ))
    fig.update_layout(
        title=dict(text='Predicted Heavy Metal Concentrations (mg/kg)', font=dict(size=14, color='#1e293b')),
        xaxis_title='Element',
        yaxis_title='Concentration (mg/kg)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=11)),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        height=390,
        bargap=0.35,
        margin=dict(t=80, b=40),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor='#f1f5f9', gridwidth=1)
    return fig


def radar_chart(predictions, dataset_stats):
    elements    = ELEMENTS
    norm_pred   = [predictions[e] / max(dataset_stats[e]['max'], 1e-9) for e in elements]
    norm_mean   = [dataset_stats[e]['mean'] / max(dataset_stats[e]['max'], 1e-9) for e in elements]
    cats        = [ELEMENT_FULL[e] for e in elements] + [ELEMENT_FULL[elements[0]]]
    norm_pred_r = norm_pred + [norm_pred[0]]
    norm_mean_r = norm_mean + [norm_mean[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_pred_r, theta=cats, fill='toself', name='AI Prediction',
        line=dict(color='#6366f1', width=2.5),
        fillcolor='rgba(99,102,241,0.18)',
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm_mean_r, theta=cats, fill='toself', name='Dataset Mean',
        line=dict(color='#10b981', width=2, dash='dot'),
        fillcolor='rgba(16,185,129,0.10)',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9), gridcolor='#e2e8f0'),
            angularaxis=dict(tickfont=dict(size=10)),
            bgcolor='rgba(248,250,255,0.5)',
        ),
        showlegend=True,
        legend=dict(font=dict(size=11)),
        title=dict(text='Element Profile (normalised)', font=dict(size=14, color='#1e293b')),
        paper_bgcolor='rgba(0,0,0,0)',
        height=380,
        margin=dict(t=60, b=20, l=40, r=40),
    )
    return fig


def metrics_heatmap(metrics):
    elements  = [e for e in ELEMENTS if e in metrics]
    r2_vals   = [metrics[e].get('r2', 0)   for e in elements]
    mae_vals  = [metrics[e].get('mae', 0)  for e in elements]
    rmse_vals = [metrics[e].get('rmse', 0) for e in elements]

    fig = go.Figure(data=go.Heatmap(
        z=[[r2_vals], [mae_vals], [rmse_vals]],
        x=[ELEMENT_FULL[e] for e in elements],
        y=['R² Score', 'MAE (mg/kg)', 'RMSE (mg/kg)'],
        colorscale='Viridis',
        text=[[f'{v:.4f}' for v in r2_vals],
              [f'{v:.4f}' for v in mae_vals],
              [f'{v:.4f}' for v in rmse_vals]],
        texttemplate='%{text}',
        hovertemplate='%{y} — %{x}: %{z:.4f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Model Performance per Element (Validation Set)', font=dict(size=14)),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ── Load everything ────────────────────────────────────────────────────────────
pipeline      = load_pipeline()
df            = load_dataset()
metrics       = load_metrics()
dataset_stats = get_dataset_stats(df) if df is not None else None

# ── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🌕 AI-Driven Lunar Soil Analysis</div>
  <div class="hero-subtitle">
      Upload a lunar satellite image — get terrain classification, heavy metal composition &amp; AI scientific report instantly.
  </div>
  <div class="hero-tags">
    <span class="hero-tag">🛰️ Satellite Imagery</span>
    <span class="hero-tag">🧪 Heavy Metal Detection</span>
    <span class="hero-tag">🏔️ Terrain AI</span>
    <span class="hero-tag">🤖 Gemini LLM Reports</span>
    <span class="hero-tag">🔬 ResNet-18 CNN</span>
  </div>
  <div class="uni-badges">
    <span class="uni-badge">🏛️ Salim Habib University</span>
    <span class="uni-badge">🛰️ SUPARCO</span>
    <span class="uni-badge">🎓 FYP Exhibition · 2026</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔬  Analysis Pipeline",
    "📊  Dataset Explorer",
    "📈  Model Performance",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Unified Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="glass-card" style="margin-bottom:1.2rem;">
      <div class="section-title">🛸 Upload Lunar Image</div>
      <p style="margin:0;color:#475569;font-size:0.9rem;">
        Upload any lunar surface image — grayscale satellite imagery, rover photos, or soil samples.
        The AI pipeline runs <strong>terrain classification</strong> and <strong>heavy metal composition estimation</strong> simultaneously.
      </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag & drop or click to upload (JPG / PNG / grayscale PCAM imagery)",
        type=['jpg', 'jpeg', 'png'],
        key='main_upload',
        label_visibility='visible',
    )

    if uploaded_file:
        raw_img = Image.open(uploaded_file)
        orig_mode = raw_img.mode

        # Handle 16-bit and float images (e.g. PCAM satellite PNGs: mode I;16, I, F)
        if orig_mode in ('I;16', 'I', 'F'):
            arr = np.array(raw_img, dtype=np.float32)
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = (arr - lo) / (hi - lo) * 255.0
            arr = arr.astype(np.uint8)
            pil_img = Image.fromarray(arr).convert('RGB')
        else:
            pil_img = raw_img.convert('RGB')

        img_np = np.array(pil_img)

        # ── Run both models ────────────────────────────────────────────────────
        terrain_result     = None
        composition_result = None

        with st.spinner("🤖 Running AI analysis pipeline..."):
            if pipeline.terrain_ready:
                terrain_result = pipeline.predict_terrain(pil_img)
            if pipeline.composition_ready:
                composition_result = pipeline.predict_composition(pil_img)

        # ── Row 1: Image + Terrain + Composition ───────────────────────────────
        col_img, col_terrain, col_comp = st.columns([1.1, 1, 1.3], gap="large")

        with col_img:
            st.markdown('<div class="section-title">🖼️ Input Image</div>', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True,
                     caption=f"{pil_img.size[0]}×{pil_img.size[1]}px · {orig_mode}")

        with col_terrain:
            st.markdown('<div class="section-title">🏔️ Terrain Classification</div>', unsafe_allow_html=True)
            if terrain_result:
                cls  = terrain_result['class_name']
                conf = terrain_result['confidence']
                icon = TERRAIN_ICONS.get(cls, '🏔️')
                st.markdown(
                    f'<div class="terrain-pill">{icon} &nbsp; {cls}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="conf-label" style="color:#64748b;">Confidence</div>',
                    unsafe_allow_html=True,
                )
                st.progress(conf, text=f"{conf*100:.1f}%")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="conf-label" style="color:#64748b;">All Probabilities</div>', unsafe_allow_html=True)
                for c, prob in sorted(terrain_result['all_probs'].items(), key=lambda x: -x[1]):
                    t_icon = TERRAIN_ICONS.get(c, '')
                    st.markdown(
                        f'<div class="prob-row">'
                        f'<span style="color:#1e293b;font-weight:600;">{t_icon} {c}</span>'
                        f'<span style="color:#4f46e5;font-weight:800;font-size:0.9rem;">{prob*100:.1f}%</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("Terrain model not loaded.")

        with col_comp:
            st.markdown('<div class="section-title">🧪 Heavy Metal Composition</div>', unsafe_allow_html=True)
            if composition_result:
                preds = composition_result['predictions']
                for e in ELEMENTS:
                    v = preds[e]
                    css_class, pill_cls, badge = contamination_class(e, v)
                    color = ELEMENT_COLORS.get(e, '#6366f1')
                    st.markdown(
                        f'<div class="elem-card {css_class}" style="border-left-color:{color};">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'  <span>'
                        f'    <span class="e-sym" style="color:{color};">{e}</span>'
                        f'    <span class="e-name"> &nbsp;{ELEMENT_FULL[e]}</span>'
                        f'  </span>'
                        f'  <span class="{pill_cls}">{badge}</span>'
                        f'</div>'
                        f'<div style="margin-top:0.25rem;">'
                        f'  <span class="e-val">{v:.3f}</span>'
                        f'  <span class="e-unit"> mg/kg</span>'
                        f'</div>'
                        f'<div class="e-lim">Safe limit: {SAFE_THRESHOLDS.get(e, "N/A")} mg/kg</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("Composition model not loaded.")

        # ── Row 2: Charts ──────────────────────────────────────────────────────
        if composition_result and dataset_stats:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Composition Visualisation</div>', unsafe_allow_html=True)
            col_bar, col_radar = st.columns(2, gap="medium")
            with col_bar:
                st.plotly_chart(composition_bar_chart(preds, dataset_stats), use_container_width=True)
            with col_radar:
                st.plotly_chart(radar_chart(preds, dataset_stats), use_container_width=True)

        # ── Row 3: Report ──────────────────────────────────────────────────────
        if composition_result:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📝 AI Scientific Report</div>', unsafe_allow_html=True)
            st.markdown(
                '<p style="color:#475569;font-size:0.88rem;margin-bottom:1rem;">'
                'Powered by Google Gemini — generates a full scientific interpretation including contamination risk, '
                'terrain context, and remediation recommendations.</p>',
                unsafe_allow_html=True,
            )

            reporter = load_reporter(gemini_key)
            if st.button("✨ Generate Scientific Report", type="primary", use_container_width=True):
                with st.spinner("🤖 Gemini AI is writing your report..."):
                    per_element_metrics = metrics.get('per_element') if metrics else None
                    report = reporter.generate_report(
                        composition=preds,
                        terrain_class=terrain_result['class_name'] if terrain_result else None,
                        image=img_np,
                        model_metrics=per_element_metrics,
                    )
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.markdown(report)
                st.markdown('</div>', unsafe_allow_html=True)

                dl_col1, dl_col2 = st.columns(2, gap="small")
                with dl_col1:
                    st.download_button(
                        "⬇️ Download as Markdown",
                        report,
                        file_name="suparco_lunar_soil_report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with dl_col2:
                    pdf_bytes = report_to_pdf(
                        report,
                        terrain_class=terrain_result['class_name'] if terrain_result else None,
                        predictions=preds,
                    )
                    st.download_button(
                        "📄 Download as PDF",
                        pdf_bytes,
                        file_name="suparco_lunar_soil_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                    )

    else:
        # ── No upload: landing state ───────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;">
          <div style="font-size:4rem;margin-bottom:1rem;">🛸</div>
          <div style="font-size:1.3rem;font-weight:700;color:#1e293b;margin-bottom:0.5rem;">
            Ready for Analysis
          </div>
          <div style="color:#64748b;font-size:0.95rem;max-width:480px;margin:0 auto;">
            Upload a lunar satellite image or soil sample photo above.<br>
            The AI pipeline will run automatically — no extra steps needed.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if df is not None and IMG_DIR.exists():
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🔭 Sample from Dataset</div>', unsafe_allow_html=True)
            sample_row      = df.iloc[0]
            sample_img_path = IMG_DIR / f"{sample_row['Sample ID']}.jpg"
            if sample_img_path.exists():
                col_s1, col_s2 = st.columns([1, 2])
                with col_s1:
                    st.image(str(sample_img_path),
                             caption=f"Sample: {sample_row['Sample ID']}",
                             use_container_width=True)
                with col_s2:
                    st.markdown("**Ground Truth Composition from SUPARCO Dataset:**")
                    for e in ELEMENTS:
                        v = float(sample_row[e])
                        css, pill_cls, badge = contamination_class(e, v)
                        color = ELEMENT_COLORS.get(e, '#6366f1')
                        st.markdown(
                            f'<div class="elem-card {css}" style="border-left-color:{color};">'
                            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                            f'  <span>'
                            f'    <span class="e-sym" style="color:{color};">{e}</span>'
                            f'    <span class="e-name"> &nbsp;{ELEMENT_FULL[e]}</span>'
                            f'  </span>'
                            f'  <span class="{pill_cls}">{badge}</span>'
                            f'</div>'
                            f'<div style="margin-top:0.2rem;">'
                            f'  <span class="e-val">{v:.3f}</span>'
                            f'  <span class="e-unit"> mg/kg</span>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Dataset Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📦 SUPARCO Soil Academia Dataset</div>', unsafe_allow_html=True)

    if df is None:
        st.error("Dataset Excel file not found.")
    else:
        # ── Stats strip ────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                '<div class="glass-card" style="text-align:center;">'
                f'<div style="font-size:2rem;font-weight:900;color:#4f46e5;">{len(df)}</div>'
                '<div style="font-size:0.82rem;color:#64748b;font-weight:600;">Total Images</div>'
                '</div>', unsafe_allow_html=True)
        with c2:
            n_unique = len(df[ELEMENTS].drop_duplicates())
            st.markdown(
                '<div class="glass-card" style="text-align:center;">'
                f'<div style="font-size:2rem;font-weight:900;color:#7c3aed;">{n_unique}</div>'
                '<div style="font-size:0.82rem;color:#64748b;font-weight:600;">Unique Compositions</div>'
                '</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(
                '<div class="glass-card" style="text-align:center;">'
                f'<div style="font-size:2rem;font-weight:900;color:#0891b2;">{len(ELEMENTS)}</div>'
                '<div style="font-size:0.82rem;color:#64748b;font-weight:600;">Heavy Metals</div>'
                '</div>', unsafe_allow_html=True)
        with c4:
            st.markdown(
                '<div class="glass-card" style="text-align:center;">'
                '<div style="font-size:2rem;font-weight:900;color:#059669;">180+</div>'
                '<div style="font-size:0.82rem;color:#64748b;font-weight:600;">Soil Samples</div>'
                '</div>', unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Distribution plot ──────────────────────────────────────────────────
        st.markdown('<div class="section-title">📉 Concentration Distributions</div>', unsafe_allow_html=True)
        melt = df[ELEMENTS].melt(var_name='Element', value_name='Concentration (mg/kg)')
        melt['Full Name'] = melt['Element'].map(ELEMENT_FULL)
        melt['Color']     = melt['Element'].map(ELEMENT_COLORS)
        fig_box = px.box(
            melt, x='Full Name', y='Concentration (mg/kg)',
            color='Full Name',
            color_discrete_map={ELEMENT_FULL[e]: ELEMENT_COLORS[e] for e in ELEMENTS},
            points='all',
            title='Heavy Metal Concentrations Across All Samples',
        )
        fig_box.update_layout(
            showlegend=False, height=420,
            plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # ── Correlation heatmap ────────────────────────────────────────────────
        st.markdown('<div class="section-title">🔗 Element Correlation Matrix</div>', unsafe_allow_html=True)
        corr = df[ELEMENTS].rename(columns=ELEMENT_FULL).corr()
        fig_corr = px.imshow(
            corr, text_auto='.2f',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title='Pearson Correlation Between Heavy Metals',
        )
        fig_corr.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_corr, use_container_width=True)

        # ── Sample browser ─────────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔍 Sample Browser</div>', unsafe_allow_html=True)
        unique_df = df[ELEMENTS].drop_duplicates().copy()
        unique_df['Sample ID'] = df.loc[unique_df.index, 'Sample ID'].values
        selected_sample = st.selectbox("Select a composition group", unique_df['Sample ID'].tolist())

        if selected_sample:
            row        = df[df['Sample ID'] == selected_sample].iloc[0]
            same_group = df[(df[ELEMENTS] == row[ELEMENTS]).all(axis=1)]['Sample ID'].tolist()
            col_si1, col_si2 = st.columns([2, 1])
            with col_si1:
                imgs_to_show = [s for s in same_group if (IMG_DIR / f"{s}.jpg").exists()]
                if imgs_to_show:
                    cols = st.columns(min(3, len(imgs_to_show)))
                    for i, sid in enumerate(imgs_to_show[:9]):
                        with cols[i % 3]:
                            st.image(str(IMG_DIR / f"{sid}.jpg"), caption=sid, use_container_width=True)
            with col_si2:
                st.markdown("**Ground Truth Composition:**")
                for e in ELEMENTS:
                    v = float(row[e])
                    css, pill_cls, badge = contamination_class(e, v)
                    color = ELEMENT_COLORS.get(e, '#6366f1')
                    st.markdown(
                        f'<div class="elem-card {css}" style="border-left-color:{color};">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'  <span>'
                        f'    <span class="e-sym" style="color:{color};">{e}</span>'
                        f'    <span class="e-name"> &nbsp;{ELEMENT_FULL[e]}</span>'
                        f'  </span>'
                        f'  <span class="{pill_cls}">{badge}</span>'
                        f'</div>'
                        f'<div style="margin-top:0.2rem;">'
                        f'  <span class="e-val">{v:.3f}</span>'
                        f'  <span class="e-unit"> mg/kg</span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with st.expander("📋 View Raw Dataset Table"):
            st.dataframe(
                df[['Sample ID'] + ELEMENTS].rename(columns={e: f'{e} (mg/kg)' for e in ELEMENTS}),
                use_container_width=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🏋️ Model Training & Performance</div>', unsafe_allow_html=True)

    if not COMP_MODEL.exists():
        st.warning("Composition model not trained yet.")
    elif metrics:
        overall  = metrics.get('overall', {})
        per_elem = metrics.get('per_element', {})

        # ── Headline metrics ───────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        metric_style = (
            '<div class="glass-card" style="text-align:center;">'
            '<div style="font-size:1.7rem;font-weight:900;color:{color};">{val}</div>'
            '<div style="font-size:0.8rem;color:#64748b;font-weight:600;">{label}</div>'
            '</div>'
        )
        with c1:
            st.markdown(metric_style.format(
                color='#4f46e5', val=f"{overall.get('mean_r2', 0):.4f}", label='Avg R² Score'
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_style.format(
                color='#0891b2', val=f"{overall.get('mean_mae', 0):.3f}", label='Avg MAE (mg/kg)'
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_style.format(
                color='#7c3aed', val=f"{overall.get('mean_rmse', 0):.3f}", label='Avg RMSE (mg/kg)'
            ), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_style.format(
                color='#059669', val=str(overall.get('best_epoch', 'N/A')), label='Best Epoch'
            ), unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Heatmap ────────────────────────────────────────────────────────────
        if per_elem:
            st.plotly_chart(metrics_heatmap(per_elem), use_container_width=True)

            rows = [
                {
                    'Element': ELEMENT_FULL[e],
                    'Symbol': e,
                    'R² Score': round(per_elem[e].get('r2', 0), 4),
                    'MAE (mg/kg)': round(per_elem[e].get('mae', 0), 4),
                    'RMSE (mg/kg)': round(per_elem[e].get('rmse', 0), 4),
                }
                for e in ELEMENTS if e in per_elem
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Loss curve ────────────────────────────────────────────────────────
        if 'train_losses' in metrics and 'val_losses' in metrics:
            st.markdown('<div class="section-title">📉 Training Loss Curve</div>', unsafe_allow_html=True)
            epochs = list(range(1, len(metrics['train_losses']) + 1))
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=metrics['train_losses'],
                name='Train Loss', line=dict(color='#6366f1', width=2.5),
                fill='tozeroy', fillcolor='rgba(99,102,241,0.07)',
            ))
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=metrics['val_losses'],
                name='Val Loss', line=dict(color='#f59e0b', width=2.5, dash='dot'),
            ))
            best_ep = overall.get('best_epoch')
            if best_ep:
                fig_loss.add_vline(
                    x=best_ep, line_dash='dash', line_color='#10b981', line_width=2,
                    annotation_text=f'Best Epoch {best_ep}',
                    annotation_font_color='#10b981',
                )
            fig_loss.update_layout(
                xaxis_title='Epoch', yaxis_title='Huber Loss',
                plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)',
                height=360,
                legend=dict(font=dict(size=11)),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        # ── Training images from Colab ─────────────────────────────────────────
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            loss_img = ROOT / 'loss_curve.png'
            if loss_img.exists():
                st.image(str(loss_img), caption="Loss Curve (Colab Training)", use_container_width=True)
        with col_p2:
            scatter_img = ROOT / 'prediction_scatter.png'
            if scatter_img.exists():
                st.image(str(scatter_img), caption="Predicted vs Ground Truth", use_container_width=True)
    else:
        st.info("Place `training_metrics.json` in `models/` to see performance stats.")
