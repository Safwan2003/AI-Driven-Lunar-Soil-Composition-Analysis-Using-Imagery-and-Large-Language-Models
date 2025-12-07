import streamlit as st
from PIL import Image
import os
import torch
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.terrain_classifier import get_model

# Page Config
st.set_page_config(
    page_title="Lunar Soil Analysis AI",
    page_icon="🌕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Space" theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942544.png", width=100)
    st.title("Lunar Analyst Control")
    st.markdown("---")
    
    analysis_mode = st.selectbox(
        "Select Analysis Mode",
        ["Terrain Classification", "Soil Composition Estimation", "Full Scientific Report"]
    )
    
    st.markdown("### Model Config")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    
    st.info("System Ready. GPU Status: " + ("🟢 Active" if torch.cuda.is_available() else "⚪ CPU Only"))

# Main Content
st.title("🚀 AI-Driven Lunar Surface Analysis")
st.markdown(f"**Current Mode:** {analysis_mode}")

# File Uploader
uploaded_file = st.file_uploader("Upload Lunar Rover Image (RGB)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # 1. Display Image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Analysis Results")
        with st.spinner("Analyzing Terrain Features..."):
            # Placeholder for inference logic
            # model = get_model()
            # result = model(transform(image))
            
            # Simulate processing time
            import time
            time.sleep(1)
            
            st.success("Analysis Complete!")
            
            # Mock Data for demo
            if analysis_mode == "Terrain Classification":
                st.metric("Detected Terrain", "Regolith (85%)")
                st.metric("Obstacles", "Small Craters Detected")
            elif analysis_mode == "Soil Composition Estimation":
                st.bar_chart({"Fe": 12, "Ti": 4, "Si": 45, "Mg": 8})
    
    # LLM Report Section
    st.markdown("---")
    st.subheader("📝 Scientific Report (LLM Generated)")
    with st.expander("Show Detailed Report", expanded=True):
        st.write("""
        **Executive Summary:**
        The analyzed image indicates a region dominated by **Regolith** with sparse cratering. 
        
        **Composition Analysis:**
        Preliminary spectral analysis suggests a **Low-Titanium (Low-Ti)** basalt composition, typical of the lunar mare regions. High Silicon content observed.
        
        **Recommendations:**
        Safe for rover navigation. Potential site for sampling hydrated minerals in nearby shadowed regions.
        """)

else:
    st.warning("Please upload an image to begin analysis.")
    
    # Demo Image Gallery
    st.markdown("### Or select a demo image:")
    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            st.button(f"Demo Sample {i+1}")

