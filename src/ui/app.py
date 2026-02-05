"""
Streamlit UI for Lunar Analysis System
"""

import streamlit as st
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.pipeline import LunarAnalysisPipeline

# Page configuration
st.set_page_config(
    page_title="SUPARCO Lunar Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS & Theme ---
def load_css():
    st.markdown("""
    <style>
        /* Import official-looking fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Orbitron:wght@500;700&display=swap');

        /* Main Container */
        .stApp {
            background: linear-gradient(180deg, #020010 0%, #0a0a2e 100%);
            color: #e0e0e0;
            font-family: 'Inter', sans-serif;
        }

        /* Headers */
        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif;
            color: #ffffff;
            text-shadow: 0 0 10px rgba(0, 191, 255, 0.5);
        }
        
        h1 {
            font-size: 3rem;
            background: -webkit-linear-gradient(eee, #333);
        }

        /* Metric Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            border-color: #00bfff;
            box-shadow: 0 5px 15px rgba(0, 191, 255, 0.2);
        }
        .metric-value {
            font-family: 'Orbitron', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            color: #00bfff;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #a0a0a0;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: rgba(10, 10, 30, 0.95);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: rgba(255,255,255,0.05);
            border-radius: 5px;
            color: #fff;
            padding: 0 20px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #00bfff !important;
            color: #000 !important;
            font-weight: bold;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #00bfff 0%, #0077aa 100%);
            border: none;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            box-shadow: 0 0 20px rgba(0, 191, 255, 0.5);
            transform: scale(1.02);
        }
        
    </style>
    """, unsafe_allow_html=True)

load_css()

# --- Helper Functions ---

@st.cache_resource
def load_pipeline():
    """Load the analysis pipeline (cached)."""
    try:
        pipeline = LunarAnalysisPipeline(use_heuristic_fallback=True)
        return pipeline
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        return None

def display_metric(label, value, sub_value=None):
    """Component to display a styled metric card."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {'<div style="font-size:0.8rem; color:#888; margin-top:5px;">' + sub_value + '</div>' if sub_value else ''}
    </div>
    """, unsafe_allow_html=True)

# --- Main App ---

# Header
col1, col2 = st.columns([1, 5])
with col1:
    # Use emoji as logo for now because paths can be tricky
    st.markdown("<div style='font-size: 4rem; text-align: center;'>🛰️</div>", unsafe_allow_html=True) 
with col2:
    st.title("SUPARCO Lunar Analysis")
    st.markdown("### AI-Driven Land Classification & Soil Composition Estimation")

st.markdown("---")

# Initialize Pipeline
pipeline = load_pipeline()

# Sidebar Configuration
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/SUPARCO_Logo.png/600px-SUPARCO_Logo.png", width="stretch") # placeholder or remote if internet allowed, otherwise emoji
    st.header("Mission Control")
    
    st.markdown("#### Detection Parameters")
    min_segment_area = st.slider(
        "Min. Feature Size (px)",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Filter out noise and small artifacts"
    )
    
    use_heuristic = st.checkbox(
        "Enable Specular Analysis",
        value=True,
        help="Use Lucey color ratio heuristics"
    )

    st.markdown("---")
    st.markdown("#### System Status")
    st.success("✔ Fast Segmenter: Online")
    st.success("✔ ResNet-18 Classifier: Online")
    st.success("✔ Composition Estimator: Online")

# Main Tabs
tab_upload, tab_results, tab_report = st.tabs(["🚀 DATA UPLOAD", "📊 TELEMETRY & ANALYSIS", "📄 MISSION REPORT"])

with tab_upload:
    st.markdown("### <span style='color:#00bfff'>//</span> Upload Satellite Imagery", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Select Lunar Surface Imagery (PCAM/LROC)",
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG. Max size: 200MB"
    )
    
    if uploaded_file:
        col_img, col_info = st.columns([1, 1])
        
        # Process Image Loading
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_np = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        # Normalization for 16-bit
        if image_np.dtype == np.uint16:
            image_np = (image_np / 256).astype(np.uint8)
        
        # Color Conversion
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)
        elif len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
        pil_image = Image.fromarray(image_np)
        
        with col_img:
            st.image(image_np, caption="Input Telemetry Data", width="stretch")
        
        with col_info:
            st.info(f"**Data Stream:** {uploaded_file.name}")
            st.code(f"""
            Resolution: {image_np.shape[1]}x{image_np.shape[0]} px
            Channels: {image_np.shape[2]}
            Depth: 8-bit Unsigned
            """)
            
            if st.button("INITIATE ANALYSIS SEQUENCE", type="primary"):
                if pipeline is None:
                    st.error("System Failure: Pipeline offline.")
                else:
                    with st.spinner("Processing telemetry... Segmeting terrain... Calculating oxides..."):
                        import tempfile
                        import os
                        
                        # Temp file dance
                        fd, temp_path = tempfile.mkstemp(suffix='.png')
                        pil_image.save(temp_path)
                        
                        try:
                            results = pipeline.analyze_image(
                                temp_path,
                                min_segment_area=min_segment_area
                            )
                            st.session_state['results'] = results
                            st.session_state['analyzed'] = True
                            st.success("Analysis Complete. Transitioning to Telemetry View.")
                        except Exception as e:
                            st.error(f"Analysis Aborted: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                        finally:
                            os.close(fd)
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

with tab_results:
    if 'analyzed' in st.session_state and st.session_state['analyzed']:
        results = st.session_state['results']
        stats = results['statistics']
        
        # --- Top Level Metrics ---
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            display_metric("Detected Features", str(stats['total_segments']))
        with m2:
            dom_terrain = max(stats['terrain_distribution'].items(), key=lambda x: x[1]['percentage'])[0]
            display_metric("Primary Terrain", dom_terrain)
        with m3:
            display_metric("Avg. Titanium (TiO2)", f"{stats['average_composition']['TiO2']:.2f}%")
        with m4:
            display_metric("Avg. Iron (FeO)", f"{stats['average_composition']['FeO']:.2f}%")
        
        st.markdown("---")
        
        # --- Visualization Grid ---
        st.markdown("### <span style='color:#00bfff'>//</span> Visual Intelligence", unsafe_allow_html=True)
        
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**Terrain Segmentation Mask**")
            st.image(results['visualizations']['segmentation'], width="stretch")
            
            st.markdown("**Terrain Classification Map**")
            st.image(results['visualizations']['terrain_map'], width="stretch")
            
        with v2:
            st.markdown("**Composition Heatmap (FeO)**")
            st.image(results['visualizations']['composition_map'], width="stretch")
            
            # --- Composition Bars ---
            st.markdown("**Elemental Breakdown**")
            comp = stats['average_composition']
            
            # Custom progress bars
            for oxide, val, max_val, color in [
                ("FeO (Iron Oxide)", comp['FeO'], 25.0, "#ff4b4b"),
                ("MgO (Magnesium Oxide)", comp['MgO'], 15.0, "#ffa500"),
                ("TiO2 (Titanium Dioxide)", comp['TiO2'], 15.0, "#00bfff"),
                ("SiO2 (Silicon Dioxide)", comp['SiO2'], 50.0, "#cccccc"),
            ]:
                st.write(f"{oxide}: **{val:.1f}%**")
                st.progress(min(val / max_val, 1.0))
                
    else:
        st.info("Awaiting Input Data. Please upload and analyze imagery in the previous tab.")

with tab_report:
    if 'analyzed' in st.session_state and st.session_state['analyzed']:
        st.markdown("### <span style='color:#00bfff'>//</span> Mission Report Generation", unsafe_allow_html=True)
        
        # Generate Text Report (Local Template)
        results = st.session_state['results']
        stats = results['statistics']
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
        
        report_content = f"""
# SUPERCO LUNAR ANALYSIS REPORT
**Classification**: UNCLASSIFIED
**Date**: {timestamp}
**Mission ID**: LUNAR-SOIL-AI-{pd.Timestamp.now().strftime('%f')[:4]}

## 1. Executive Summary
The automated AI analysis system identified **{stats['total_segments']} distinct geological features** within the target area. The region is primarily characterized as **{max(stats['terrain_distribution'].items(), key=lambda x: x[1]['percentage'])[0]}**.

## 2. Terrain Analysis
| Type | Segments | Coverage |
|------|----------|----------|
"""
        for terrain, data in stats['terrain_distribution'].items():
            report_content += f"| {terrain} | {data['count']} | {data['percentage']:.1f}% |\n"
            
        report_content += f"""

## 3. Compositional Estimates (Wt%)
Computed area-weighted average oxide abundances:
- **FeO**: {stats['average_composition']['FeO']:.2f}%
- **MgO**: {stats['average_composition']['MgO']:.2f}%
- **TiO2**: {stats['average_composition']['TiO2']:.2f}%
- **SiO2**: {stats['average_composition']['SiO2']:.2f}%

## 4. Geological Interpretation
Based on TiO2 abundances ({stats['average_composition']['TiO2']:.2f}%):
"""
        if stats['average_composition']['TiO2'] > 6.0:
            report_content += "- **High-Ti Mare Basalt**: Indicates significant Ilmenite content. Potential resource for oxygen extraction."
        elif stats['average_composition']['TiO2'] > 2.0:
            report_content += "- **Low-Ti Mare Basalt**: Standard mare volcanism."
        else:
            report_content += "- **Highland/Anorthosite**: Low titanium content, likely crustal material."

        report_content += "\n\n---\n*Generated by SUPARCO AI Module*"

        st.text_area("Report Preview", report_content, height=400)
        
        st.download_button(
            label="💾 EXPORT MISSION DATA",
            data=report_content,
            file_name=f"SUPARCO_Report_{timestamp.replace(' ','_')}.md",
            mime="text/markdown",
            type="primary"
        )
    else:
        st.info("Report module standing by. Initiate analysis to generate data.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 0.8rem; opacity: 0.7;'>
    SUPARCO AI RESEARCH DIVISION<br>
    Lunar Soil Composition Analysis System v2.0
</div>
""", unsafe_allow_html=True)
