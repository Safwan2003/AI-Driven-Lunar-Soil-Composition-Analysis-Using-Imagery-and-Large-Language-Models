"""
Streamlit UI for SUPARCO Lunar Analysis System
AI-Driven Lunar Soil Composition Analysis & Terrain Classification
"""

import streamlit as st
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import sys
import traceback
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.pipeline import LunarAnalysisPipeline

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SUPARCO Lunar Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS Theme
# ──────────────────────────────────────────────────────────────────────────────
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Orbitron:wght@400;500;700&display=swap');

        .stApp {
            background: linear-gradient(160deg, #020210 0%, #050520 60%, #08102a 100%);
            color: #e0e6f5;
            font-family: 'Inter', sans-serif;
        }
        h1, h2, h3 { font-family: 'Orbitron', sans-serif; color: #ffffff; }
        h1 { font-size: 2.2rem; letter-spacing: 3px; font-weight: 700; border-left: 5px solid #00bfff; padding-left: 20px; }
        h2 { font-size: 1.3rem; color: #00bfff; text-transform: uppercase; letter-spacing: 1px; }
        h3 { font-size: 1.0rem; color: #7899bb; }

        /* Metric cards */
        .metric-card {
            background: rgba(0, 191, 255, 0.05);
            border: 1px solid rgba(0, 191, 255, 0.15);
            border-left: 3px solid #00bfff;
            border-radius: 4px;
            padding: 15px 20px;
            margin-bottom: 10px;
            transition: all 0.2s ease;
        }
        .metric-card:hover {
            background: rgba(0, 191, 255, 0.1);
            border-color: rgba(0, 191, 255, 0.4);
        }
        .metric-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.8rem;
            font-weight: 600;
            color: #00bfff;
        }
        .metric-label {
            font-size: 0.7rem;
            color: #7899bb;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-sub {
            font-size: 0.78rem;
            color: #556677;
            margin-top: 4px;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #02020a;
            border-right: 1px solid rgba(0,191,255,0.1);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid rgba(0,191,255,0.15); }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.04);
            border-radius: 6px 6px 0 0;
            color: #aac0dd;
            padding: 8px 20px;
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(0,191,255,0.15) !important;
            color: #00bfff !important;
            border-bottom: 2px solid #00bfff !important;
            font-weight: 600;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #005f8a 0%, #0099cc 100%);
            border: none;
            color: #fff;
            padding: 10px 26px;
            border-radius: 8px;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.82rem;
            letter-spacing: 1px;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #0077aa 0%, #00bfff 100%);
            box-shadow: 0 0 18px rgba(0,191,255,0.45);
        }

        /* Section dividers */
        .section-header {
            color: #00bfff;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.95rem;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin: 24px 0 12px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(0,191,255,0.2);
        }

        /* Status pills */
        .status-ok  { color: #00e676; font-weight: 600; }
        .status-warn{ color: #ffb300; font-weight: 600; }
        .status-err { color: #ff5252; font-weight: 600; }

        /* Report box */
        .report-box {
            background: rgba(0,10,30,0.7);
            border: 1px solid rgba(0,191,255,0.15);
            border-radius: 10px;
            padding: 24px;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            line-height: 1.7;
        }

        /* Confidence bar wrapper */
        .conf-bar { margin-bottom: 10px; }
        .conf-label { font-size: 0.82rem; color: #99b3cc; margin-bottom: 2px; }
    </style>
    """, unsafe_allow_html=True)

load_css()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline + LLM availability (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    try:
        return LunarAnalysisPipeline(use_heuristic_fallback=True)
    except Exception as e:
        st.error(f"Pipeline init failed: {e}")
        return None


@st.cache_resource
def check_llm_available():
    try:
        from src.llm.llm_reporter import LunarLLMReporter
        r = LunarLLMReporter()
        return r.is_available
    except Exception:
        return False


def display_metric(label, value, sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ''
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# App Header
# ──────────────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 7])
with col_logo:
    st.markdown("<div style='font-size:3.5rem;text-align:center;padding-top:8px'>🛰️</div>",
                unsafe_allow_html=True)
with col_title:
    st.title("SUPARCO LUNAR ANALYSIS SYSTEM")
    st.markdown(
        "<span style='color:#7899bb;font-size:0.95rem;letter-spacing:1px'>"
        "AI-DRIVEN SOIL COMPOSITION · TERRAIN CLASSIFICATION · LLM SCIENTIFIC REPORTING"
        "</span>",
        unsafe_allow_html=True
    )

st.markdown('<hr style="border-color:rgba(0,191,255,0.2)">', unsafe_allow_html=True)

# Load pipeline
pipeline = load_pipeline()
llm_ready = check_llm_available()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/SUPARCO_Logo.png/600px-SUPARCO_Logo.png",
            use_container_width=True
        )
    except Exception:
        st.markdown("### 🛰️ SUPARCO")

    st.markdown("---")
    st.markdown("#### Analysis Parameters")
    min_segment_area = st.slider(
        "Min Feature Size (px)",
        min_value=100, max_value=2000, value=500, step=100,
        help="Ignore segments smaller than this area"
    )
    use_vision_llm = st.checkbox(
        "Send Image to Gemini Vision",
        value=True,
        help="Enable visual analysis alongside numerical data"
    )
    mission_context = st.text_area(
        "Mission Context (optional)",
        placeholder="e.g. Chang'e 3 landing site, Mare Imbrium, target for ISRU demo",
        height=80
    )

    st.markdown("---")
    st.markdown("#### System Status")
    seg_status = "✔ Online" if pipeline else "✘ Offline"
    llm_status_text = "✔ Gemini Online" if llm_ready else "⚠ Fallback Mode"
    st.markdown(f"<span class='{'status-ok' if pipeline else 'status-err'}'>{seg_status} — Segmentation</span>",
                unsafe_allow_html=True)
    st.markdown(f"<span class='{'status-ok' if llm_ready else 'status-warn'}'>{llm_status_text} — LLM Reporter</span>",
                unsafe_allow_html=True)
    st.markdown(f"<span class='status-ok'>✔ Online — Composition Estimator</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<small style='color:#445566'>SUPARCO AI Research Division<br>Lunar Soil Analysis v2.0</small>",
                unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Main Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_upload, tab_results, tab_report = st.tabs([
    "🚀  DATA UPLOAD",
    "📊  TELEMETRY & ANALYSIS",
    "🤖  AI MISSION REPORT",
])

# ── TAB 1: Upload ──────────────────────────────────────────────────────────────
with tab_upload:
    section("// Upload Lunar Surface Imagery")

    uploaded_file = st.file_uploader(
        "Select PCAM / LROC Imagery",
        type=['png', 'jpg', 'jpeg'],
        help="Supports PNG/JPG. 16-bit images are normalised automatically."
    )

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        # Normalize 16-bit → 8-bit
        if image_np.dtype == np.uint16:
            image_np = (image_np / 256).astype(np.uint8)

        # Ensure RGB
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image_np)

        col_img, col_info = st.columns([3, 2])
        with col_img:
            st.image(image_np, caption=f"Input: {uploaded_file.name}", use_container_width=True)
        with col_info:
            section("// Image Metadata")
            st.code(
                f"File    : {uploaded_file.name}\n"
                f"Res     : {image_np.shape[1]} × {image_np.shape[0]} px\n"
                f"Channels: {image_np.shape[2]}\n"
                f"Bit Depth: 8-bit unsigned",
                language="yaml"
            )

            if st.button("⚡  INITIATE ANALYSIS SEQUENCE", type="primary"):
                if pipeline is None:
                    st.error("Pipeline offline. Check system logs.")
                else:
                    with st.spinner("Segmenting terrain... Classifying features... Estimating oxides..."):
                        import tempfile, os

                        fd, temp_path = tempfile.mkstemp(suffix='.png')
                        pil_image.save(temp_path)
                        try:
                            results = pipeline.analyze_image(
                                temp_path,
                                min_segment_area=min_segment_area
                            )
                            st.session_state['results'] = results
                            st.session_state['analyzed'] = True
                            st.session_state['llm_report'] = None  # Reset
                            st.success("✅ Analysis complete — navigate to Telemetry tab")
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                            st.code(traceback.format_exc())
                        finally:
                            os.close(fd)
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
    else:
        st.info("Upload a lunar surface image to begin analysis.")

# ── TAB 2: Results ─────────────────────────────────────────────────────────────
with tab_results:
    if st.session_state.get('analyzed'):
        results = st.session_state['results']
        stats = results['statistics']
        segments = results.get('segments', [])
        mode = results.get('mode', 'unknown')
        comp = stats['average_composition']

        # ── Top KPI row ──
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            display_metric("Detected Features", str(stats['total_segments']), f"Mode: {mode}")
        with m2:
            dom = max(stats['terrain_distribution'].items(),
                      key=lambda x: x[1]['percentage'])[0] if stats['terrain_distribution'] else "N/A"
            dom_pct = stats['terrain_distribution'].get(dom, {}).get('percentage', 0)
            display_metric("Primary Terrain", dom, f"{dom_pct:.1f}% coverage")
        with m3:
            display_metric("Avg TiO₂", f"{comp['TiO2']:.2f}%", "Titanium Dioxide")
        with m4:
            display_metric("Avg FeO", f"{comp['FeO']:.2f}%", "Iron Oxide")

        st.markdown("")

        # ── Visualizations ──
        section("// Visual Intelligence")
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**Terrain Segmentation Mask**")
            st.image(results['visualizations']['segmentation'], use_container_width=True)
            st.markdown("**Terrain Classification Map**")
            st.image(results['visualizations']['terrain_map'], use_container_width=True)
        with v2:
            st.markdown("**FeO Composition Heatmap** *(red=high, blue=low)*")
            st.image(results['visualizations']['composition_map'], use_container_width=True)

            section("// Elemental Breakdown (wt%)")
            oxide_cfg = [
                ("FeO — Iron Oxide",        comp.get('FeO', 0),  25.0, "#e63946"),
                ("MgO — Magnesium Oxide",   comp.get('MgO', 0),  15.0, "#f4a261"),
                ("TiO₂ — Titanium Dioxide", comp.get('TiO2', 0), 15.0, "#00bfff"),
                ("SiO₂ — Silicon Dioxide",  comp.get('SiO2', 0), 52.0, "#90e0ef"),
                ("Al₂O₃ — Aluminum Oxide",  comp.get('Al2O3', 0), 30.0, "#00e676"),
                ("CaO — Calcium Oxide",     comp.get('CaO', 0),   16.0, "#ffb300"),
            ]
            for label, val, max_val, color in oxide_cfg:
                pct = min(val / max_val, 1.0)
                st.markdown(
                    f"<div class='conf-bar'>"
                    f"<div class='conf-label'>{label}  —  <b style='color:{color}'>{val:.2f}%</b></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.progress(pct)

        # ── Terrain Distribution Table ──
        section("// Terrain Distribution")
        if stats['terrain_distribution']:
            df = pd.DataFrame([
                {
                    'Terrain Class': terrain,
                    'Segments': data['count'],
                    'Coverage (%)': f"{data['percentage']:.1f}",
                    'Area (px)': f"{data['area']:,}"
                }
                for terrain, data in stats['terrain_distribution'].items()
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Geological Quick Interpretation ──
        section("// Geological Classification")
        if pipeline:
            geo = pipeline.heuristic_estimator.classify_geologic_unit(comp)
        else:
            geo = "System Offline"

        st.markdown(f"**Preliminary classification:** {geo}")
        st.caption("Full interpretation available in the AI Mission Report tab.")

    else:
        st.info("No analysis data yet. Upload an image and run analysis first.")

# ── TAB 3: LLM Report ──────────────────────────────────────────────────────────
with tab_report:
    if st.session_state.get('analyzed'):
        results = st.session_state['results']

        section("// AI Scientific Mission Report")

        col_gen, col_status = st.columns([3, 1])
        with col_gen:
            gen_btn = st.button("🤖  GENERATE AI REPORT", type="primary")
        with col_status:
            if llm_ready:
                st.markdown("<span class='status-ok'>● Gemini Active</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='status-warn'>● Fallback Mode</span>", unsafe_allow_html=True)
                st.caption("Add GEMINI_API_KEY to .env for full LLM analysis")

        if gen_btn or st.session_state.get('llm_report'):
            if gen_btn or st.session_state.get('llm_report') is None:
                with st.spinner("Generating scientific report... (this may take 10–30 seconds)"):
                    try:
                        report_text = pipeline.generate_report(
                            results,
                            use_vision=use_vision_llm,
                            mission_context=mission_context
                        )
                        st.session_state['llm_report'] = report_text
                    except Exception as e:
                        st.error(f"Report generation failed: {e}")
                        st.session_state['llm_report'] = None

            if st.session_state.get('llm_report'):
                report_text = st.session_state['llm_report']

                # Render as markdown
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.markdown(report_text)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("")
                col_dl1, col_dl2 = st.columns(2)
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
                with col_dl1:
                    st.download_button(
                        label="💾  Export as Markdown",
                        data=report_text,
                        file_name=f"SUPARCO_Lunar_Report_{timestamp}.md",
                        mime="text/markdown",
                        type="primary"
                    )
                with col_dl2:
                    # Export JSON of full analysis
                    import json
                    stats = results['statistics']
                    export_data = {
                        'timestamp': timestamp,
                        'mode': results.get('mode'),
                        'statistics': {
                            'total_segments': stats['total_segments'],
                            'average_composition': stats['average_composition'],
                            'terrain_distribution': {
                                k: {'count': v['count'], 'percentage': v['percentage']}
                                for k, v in stats['terrain_distribution'].items()
                            }
                        },
                        'report': report_text
                    }
                    st.download_button(
                        label="📦  Export as JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"SUPARCO_Analysis_{timestamp}.json",
                        mime="application/json"
                    )
        else:
            st.info(
                "Click **GENERATE AI REPORT** to produce a full scientific report with:\n"
                "- Geological classification (Mare / Highland / KREEP)\n"
                "- Mineralogical interpretation (Ilmenite, Olivine, Pyroxene, Plagioclase)\n"
                "- ISRU resource potential assessment\n"
                "- Terrain safety evaluation\n"
                "- Mission recommendations"
            )

    else:
        st.info("Report module standing by. Upload and analyse an image first.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<hr style="border-color:rgba(0,191,255,0.1);margin-top:40px">', unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; font-size:0.75rem; color:#334455; padding-bottom:10px'>
    SUPARCO AI RESEARCH DIVISION &nbsp;·&nbsp;
    Lunar Soil Composition Analysis System v2.0 &nbsp;·&nbsp;
    SAM 2.1 · ResNet-18 · Gemini LLM
</div>
""", unsafe_allow_html=True)
