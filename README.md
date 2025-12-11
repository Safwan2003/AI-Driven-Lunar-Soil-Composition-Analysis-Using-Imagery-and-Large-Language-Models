# AI-Powered Lunar Surface & Soil Composition Analysis

**Final Year Project | SUPARCO Collaboration**

A complete AI system for analyzing lunar rover imagery to detect terrain features and infer elemental soil composition, enhanced with Large Language Model (LLM) reasoning.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/FullMoon2010.jpg/800px-FullMoon2010.jpg" width="400"/>

## 🎯 Features

### Phase 1: Terrain Classification
- **Crater Detection** - Identify impact craters and their characteristics
- **Regolith Analysis** - Classify flat lunar soil regions  
- **Boulder Detection** - Locate and classify rocky features
- **Vision Transformers** - ResNet-18 or ViT models

### Phase 2: Soil Composition Estimation
- **Elemental Analysis** - Fe, Mg, Ti, Si percentage estimation
- **Moisture Detection** - Hydration level classification
- **Spectral Inference** - RGB-based composition mapping

### LLM Integration
- **Scientific Reports** - Natural language analysis via Gemini API
- **Chain-of-Thought** - Reasoning about geological features
- **Mission Planning** - Automated recommendations

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and navigate to project
cd AI-Driven-Lunar-Soil-Composition-Analysis

# Run setup (creates .venv and installs dependencies)
setup_env.bat
```

### 2. Download Dataset
```bash
# Download Chang'e 3 PCAM images (~914 images)
.venv\Scripts\python src\data\download_dataset.py
```

### 3. Configure LLM (Optional but Recommended)
```bash
# Copy template and add your Gemini API key
copy .env.example .env
notepad .env  # Add your key from https://ai.google.dev
```

### 4. Run Complete System
```bash
# One-command launch (handles everything)
run_project.bat
```

This will:
✓ Generate labels  
✓ Train models (if needed)  
✓ Launch Streamlit app  

## 📁 Project Structure

```
Project/
├── data/                      # Raw unlabeled images
│   ├── pcam/                 # Chang'e 3 panoramic camera
│   └── tcam/                 # Chang'e 3 terrain camera
│
├── labeled_data/              # Labeled dataset (modular)
│   ├── opensource/           # AI4Mars, NASA labeled data
│   ├── suparco/              # SUPARCO data (ready for integration)
│   └── annotations.csv       # Unified labels
│
├── src/
│   ├── data/
│   │   ├── download_dataset.py    # Chang'e 3 downloader
│   │   └── label_importer.py      # Multi-source label merger
│   │
│   ├── models/
│   │   ├── terrain_classifier.py  # ResNet/ViT terrain model
│   │   ├── composition_estimator.py  # Multi-output composition model
│   │   ├── dataset.py             # PyTorch DataLoader
│   │   └── train_model.py         # Training pipeline
│   │
│   ├── llm/
│   │   └── gemini_client.py       # Gemini API wrapper
│   │
│   └── ui/
│       └── app.py                 # Streamlit multi-page app
│
├── models/                    # Saved model checkpoints
│   ├── lunar_terrain_classifier.pth
│   └── composition_estimator.pth
│
├── .env.example               # Configuration template
├── requirements.txt           # Python dependencies
├── setup_env.bat             # Environment setup script
└── run_project.bat           # Complete system runner
```

## 🔧 SUPARCO Data Integration

This system is designed for easy integration with SUPARCO-provided labeled data:

1. Place images in: `labeled_data/suparco/images/`
2. Create CSV: `labeled_data/suparco/annotations.csv`

**CSV Format:**
```csv
filename,terrain_class,fe_percent,mg_percent,ti_percent,si_percent,moisture_level,notes
IMG_001.png,regolith,8.5,4.2,1.3,45.2,low,Mare region sample
```

3. Run: `python src/data/label_importer.py`

The system automatically merges and prioritizes SUPARCO data over synthetic labels.

## 📊 Application Pages

### 🏠 Home
- Project overview and dataset status
- Quick navigation

### 🔬 Live Analysis
- Upload lunar image
- Get terrain classification + composition
- View LLM-generated scientific report

### 📊 Batch Processing
- Analyze multiple images
- Generate aggregate statistics
- *(Coming soon)*

### 💾 Dataset Explorer
- Browse labeled samples
- View distribution charts
- Sample image gallery

### ⚙️ System Status
- Environment diagnostics
- Model status
- LLM configuration check

## 🧪 Technical Stack

**Languages:** Python 3.10+  
**Deep Learning:** PyTorch, TorchVision  
**LLM:** Google Gemini API (gemini-1.5-flash)  
**UI:** Streamlit  
**Data:** Pandas, NumPy, OpenCV  
**Models:** ResNet-18, Custom Multi-Output Networks  

## 📈 Model Performance

**Terrain Classification:**
- Accuracy: ~37% (on synthetic labels)
- Note: Will achieve >80% with real labeled data

**Composition Estimation:**
- Currently uses pretrained ImageNet weights
- Fine-tuning planned with SUPARCO spectral data

## 🎓 FYP Milestones

- [x] M1: Data Acquisition - Chang'e 3 dataset downloaded
- [x] M2: Labeling Infrastructure - Modular system ready
- [x] M3: Terrain Model - ResNet-18 trained
- [x] M4: Composition Model - Architecture implemented
- [x] M5: LLM Integration - Gemini client functional
- [x] M6: Application - Multi-page Streamlit app complete
- [ ] M7: SUPARCO Integration - Awaiting real labeled data
- [ ] M8: Model Refinement - Fine-tune with SUPARCO data

## 🤝 Contributors

**Student:** Safwan Ali & Shayaan Khurram  
**Supervisor:** Dr. Raazia Sosan  
**Organization:** SUPARCO

## 📝 License

Academic Use Only - Final Year Project

## 🔗 Resources

- [Chang'e 3 Dataset](https://www.planetary.org/articles/01281656-fun-with-a-new-data-set-change)
- [Gemini API](https://ai.google.dev)
- [Project Proposal](docs/FYP_Proposal.pdf) *(if available)*

---

**Need Help?**  
Check `labeled_data/suparco/README.md` for integration instructions.

For LLM setup: Get your free Gemini API key at [ai.google.dev](https://ai.google.dev)
