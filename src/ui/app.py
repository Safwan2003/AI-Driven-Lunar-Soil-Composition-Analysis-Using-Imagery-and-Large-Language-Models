"""
Enhanced Streamlit Application for Lunar Analysis
Multi-page app with LLM integration and complete pipeline
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import pandas as pd
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.terrain_classifier import LunarTerrainClassifier
from src.models.composition_estimator import CompositionEstimator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="Lunar Analysis AI - FYP",
    page_icon="🌑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BASE_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
TERRAIN_MODEL_PATH = MODEL_DIR / "lunar_terrain_classifier.pth"
COMP_MODEL_PATH = MODEL_DIR / "composition_estimator.pth"

TERRAIN_CLASSES = {0: "Regolith (Flat)", 1: "Crater", 2: "Boulder/Rock", 3: "Mixed"}
MOISTURE_CLASSES = {0: "None", 1: "Trace", 2: "Low", 3: "Medium", 4: "High"}

# Styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1c3a 100%);
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .report-box {
        border: 2px solid #667eea;
        padding: 20px;
        border-radius: 12px;
        background: rgba(26, 28, 58, 0.8);
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #667eea44;
    }
    h1, h2, h3 {
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load terrain and composition models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Terrain classifier
    terrain_model = LunarTerrainClassifier(num_classes=4)
    if TERRAIN_MODEL_PATH.exists():
        try:
            terrain_model.load_state_dict(torch.load(TERRAIN_MODEL_PATH, map_location=device))
            st.toast("✓ Terrain model loaded", icon="✅")
        except Exception as e:
            st.warning(f"Using untrained terrain model: {e}")
    else:
        st.warning("Terrain model not found - using untrained weights")
    
    # Composition estimator
    comp_model = CompositionEstimator(pretrained=True)
    if COMP_MODEL_PATH.exists():
        try:
            comp_model.load_state_dict(torch.load(COMP_MODEL_PATH, map_location=device))
            st.toast("✓ Composition model loaded", icon="✅")
        except Exception as e:
            st.warning(f"Using untrained composition model: {e}")
    else:
        st.info("Composition model not found - using pretrained ImageNet weights")
    
    terrain_model.to(device).eval()
    comp_model.to(device).eval()
    
    return terrain_model, comp_model, device

@st.cache_resource
def load_llm_client():
    """Load Gemini LLM client if API key is available."""
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key or api_key == 'your-api-key-here':
        return None
    
    try:
        from src.llm.gemini_client import GeminiClient
        client = GeminiClient(api_key=api_key)
        st.toast("✓ Gemini LLM connected", icon="🤖")
        return client
    except Exception as e:
        st.error(f"LLM initialization failed: {e}")
        return None

def process_image(image):
    """Preprocess image for model input."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def generate_fallback_report(terrain_class, confidence, composition):
    """Generate report when LLM is unavailable."""
    return f"""
### Analysis Report (Generated without LLM)

**Terrain Classification:** {terrain_class}  
**Confidence:** {confidence:.1%}

**Detected Composition:**
- Iron (Fe): {composition['fe']:.2f}%
- Magnesium (Mg): {composition['mg']:.2f}%
- Titanium (Ti): {composition['ti']:.2f}%
- Silicon (Si): {composition['si']:.2f}%
- Moisture: {composition['moisture']}

**Note:** For detailed LLM-powered analysis, please configure GEMINI_API_KEY in your .env file.
Get a free API key at: https://ai.google.dev
"""

# Sidebar Navigation
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/FullMoon2010.jpg/800px-FullMoon2010.jpg", use_column_width=True)
    st.title("🌑 Lunar Command Center")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔬 Live Analysis", "📊 Batch Processing", "💾 Dataset Explorer", "⚙️ System Status"],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.caption("SUPARCO Lunar FYP Project")
    st.caption(f"Device: {'🎮 CUDA' if torch.cuda.is_available() else '💻 CPU'}")

# PAGE: Home
if page == "🏠 Home":
    st.title("🌑 AI-Powered Lunar Surface Analysis")
    st.markdown("### Final Year Project | SUPARCO Collaboration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎯 Phase 1")
        st.markdown("""
        **Terrain Classification**
        - Crater Detection
        - Regolith Analysis
        - Boulder Identification
        """)
    
    with col2:
        st.markdown("#### 🧪 Phase 2")
        st.markdown("""
        **Composition Analysis**
        - Fe, Mg, Ti, Si Detection
        - Moisture Level
        - Mineral Mapping
        """)
    
    with col3:
        st.markdown("#### 🤖 LLM Integration")
        st.markdown("""
        **AI Reasoning**
        - Natural Language Reports
        - Scientific Analysis
        - Mission Recommendations
        """)
    
    st.divider()
    
    st.markdown("### 📁 Dataset Status")
    
    # Check dataset status
    pcam_dir = BASE_DIR / "data" / "pcam"
    labels_file = BASE_DIR / "labeled_data" / "annotations.csv"
    suparco_dir = BASE_DIR / "labeled_data" / "suparco" / "images"
    
    dcol1, dcol2, dcol3 = st.columns(3)
    
    with dcol1:
        if pcam_dir.exists():
            count = len(list(pcam_dir.glob("*.png")))
            st.metric("Raw Images (PCAM)", f"{count}")
        else:
            st.metric("Raw Images", "⚠️ Not Found")
    
    with dcol2:
        if labels_file.exists():
            df = pd.read_csv(labels_file)
            st.metric("Labeled Samples", f"{len(df)}")
        else:
            st.metric("Labeled Samples", "⚠️ Not Found")
    
    with dcol3:
        if suparco_dir.exists() and any(suparco_dir.iterdir()):
            count = len(list(suparco_dir.glob("*.png")))
            st.metric("SUPARCO Data", f"✓ {count} images")
        else:
            st.metric("SUPARCO Data", "📂 Ready (Empty)")
    
    st.info("💡 **Quick Start:** Go to '🔬 Live Analysis' to upload and analyze lunar images")

# PAGE: Live Analysis
elif page == "🔬 Live Analysis":
    st.title("🔬 Live Image Analysis")
    st.write("Upload a lunar rover image for AI-powered terrain and composition analysis")
    
    terrain_model, comp_model, device = load_models()
    llm_client = load_llm_client()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Lunar Image", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Rover View', use_column_width=True)
            
            if st.button("🚀 Analyze Surface", type="primary"):
                with st.spinner("Processing AI models..."):
                    input_tensor = process_image(image).to(device)
                    
                    # Terrain Classification
                    with torch.no_grad():
                        terrain_output = terrain_model(input_tensor)
                        terrain_probs = torch.nn.functional.softmax(terrain_output, dim=1)
                        terrain_conf, terrain_pred = torch.max(terrain_probs, 1)
                        
                        terrain_class = TERRAIN_CLASSES[terrain_pred.item()]
                        terrain_confidence = terrain_conf.item()
                    
                    # Composition Estimation
                    with torch.no_grad():
                        comp_output = comp_model(input_tensor)
                        
                        composition = {
                            'fe': comp_output['fe'].item(),
                            'mg': comp_output['mg'].item(),
                            'ti': comp_output['ti'].item(),
                            'si': comp_output['si'].item(),
                            'moisture': MOISTURE_CLASSES[torch.argmax(comp_output['moisture']).item()]
                        }
                    
                    # Store results in session state
                    st.session_state['analysis_complete'] = True
                    st.session_state['terrain_class'] = terrain_class
                    st.session_state['terrain_confidence'] = terrain_confidence
                    st.session_state['composition'] = composition
                    st.session_state['terrain_probs'] = terrain_probs.cpu().numpy()[0]
                
                st.success("✓ Analysis Complete")
                st.rerun()
    
    with col2:
        if st.session_state.get('analysis_complete'):
            st.subheader("📋 Results")
            
            # Terrain Classification
            st.markdown("#### Terrain Detection")
            st.metric(
                label="Classification",
                value=st.session_state['terrain_class'],
                delta=f"{st.session_state['terrain_confidence']:.1%} confidence"
            )
            
            # Composition
            st.markdown("#### Elemental Composition")
            comp = st.session_state['composition']
            
            ccol1, ccol2 = st.columns(2)
            with ccol1:
                st.metric("Iron (Fe)", f"{comp['fe']:.2f}%")
                st.metric("Magnesium (Mg)", f"{comp['mg']:.2f}%")
            with ccol2:
                st.metric("Titanium (Ti)", f"{comp['ti']:.2f}%")
                st.metric("Silicon (Si)", f"{comp['si']:.2f}%")
            
            st.metric("Moisture Level", comp['moisture'])
            
            # LLM Report
            st.markdown("#### 🤖 LLM Analysis Report")
            
            if llm_client:
                with st.spinner("Generating scientific analysis..."):
                    try:
                        # Save uploaded image temporarily
                        temp_path = BASE_DIR / "temp_analysis.png"
                        image.save(temp_path)
                        
                        report = llm_client.generate_terrain_report(
                            image_path=str(temp_path),
                            terrain_class=st.session_state['terrain_class'],
                            confidence=st.session_state['terrain_confidence'],
                            composition=comp
                        )
                        
                        with st.expander("📄 Full Scientific Report", expanded=True):
                            st.markdown(report['full_report'])
                        
                        # Cleanup
                        temp_path.unlink(missing_ok=True)
                        
                    except Exception as e:
                        st.error(f"LLM Error: {e}")
                        st.markdown(generate_fallback_report(
                            st.session_state['terrain_class'],
                            st.session_state['terrain_confidence'],
                            comp
                        ))
            else:
                st.warning("⚠️ LLM Not Configured")
                st.markdown(generate_fallback_report(
                    st.session_state['terrain_class'],
                    st.session_state['terrain_confidence'],
                    comp
                ))
                st.info("Configure GEMINI_API_KEY in .env to enable LLM reports")

# PAGE: Batch Processing
elif page == "📊 Batch Processing":
    st.title("📊 Batch Image Analysis")
    st.write("Analyze multiple images and generate aggregate reports")
    
    st.info("🚧 Coming Soon: Upload multiple images and get batch statistics")
    
    # Placeholder UI
    uploaded_files = st.file_uploader("Upload Multiple Images", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"Loaded {len(uploaded_files)} images")
        
        if st.button("Process Batch"):
            st.warning("Batch processing will be implemented in next version")

# PAGE: Dataset Explorer
elif page == "💾 Dataset Explorer":
    st.title("💾 Dataset Explorer")
    
    labels_file = BASE_DIR / "labeled_data" / "annotations.csv"
    
    if labels_file.exists():
        df = pd.read_csv(labels_file)
        
        st.markdown(f"### 📊 Dataset Statistics")
        st.write(f"Total Samples: **{len(df)}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Terrain Distribution")
            terrain_dist = df['terrain_class'].value_counts()
            st.bar_chart(terrain_dist)
        
        with col2:
            st.markdown("#### Composition Stats")
            st.write(df[['fe_percent', 'mg_percent', 'ti_percent', 'si_percent']].describe())
        
        st.markdown("### 🔍 Browse Samples")
        st.dataframe(df.head(50), use_container_width=True)
        
        # Sample Images
        st.markdown("### 🖼️ Sample Gallery")
        pcam_dir = BASE_DIR / "data" / "pcam"
        
        if pcam_dir.exists():
            sample_files = list(pcam_dir.glob("*.png"))[:12]
            
            cols = st.columns(4)
            for i, img_file in enumerate(sample_files):
                with cols[i % 4]:
                    try:
                        img = Image.open(img_file)
                        st.image(img, caption=img_file.stem[:20], use_column_width=True)
                    except:
                        pass
    else:
        st.error("❌ No labeled dataset found")
        st.info("Run `python src/data/label_importer.py` to generate labels")

# PAGE: System Status
elif page == "⚙️ System Status":
    st.title("⚙️ System Diagnostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🖥️ Environment")
        st.metric("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        st.metric("PyTorch", torch.__version__)
        st.metric("CUDA Available", "✓ Yes" if torch.cuda.is_available() else "✗ No")
        
        if torch.cuda.is_available():
            st.metric("GPU", torch.cuda.get_device_name(0))
    
    with col2:
        st.markdown("### 📦 Model Status")
        
        if TERRAIN_MODEL_PATH.exists():
            st.success("✓ Terrain Classifier Found")
        else:
            st.error("✗ Terrain Classifier Missing")
        
        if COMP_MODEL_PATH.exists():
            st.success("✓ Composition Estimator Found")
        else:
            st.warning("⚠ Composition Estimator Missing (will use pretrained)")
        
        st.markdown("### 🤖 LLM Configuration")
        api_key = os.getenv('GEMINI_API_KEY')
        
        if api_key and api_key != 'your-api-key-here':
            st.success("✓ Gemini API Key Configured")
        else:
            st.error("✗ Gemini API Key Not Set")
            st.code("1. Copy .env.example to .env\n2. Add your API key from ai.google.dev\n3. Restart app", language="bash")
    
    st.markdown("### 📁 Project Structure")
    st.code("""
Project/
├── data/              ← Raw images
├── labeled_data/      ← Annotations
│   └── suparco/      ← SUPARCO data folder
├── models/           ← Trained checkpoints
├── src/
│   ├── data/         ← Data tools
│   ├── models/       ← AI models
│   ├── llm/          ← LLM client
│   └── ui/           ← This app
└── .env              ← Configuration
    """, language="plaintext")
