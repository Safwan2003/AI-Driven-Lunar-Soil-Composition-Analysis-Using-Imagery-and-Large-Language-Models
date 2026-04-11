# 🌙 SUPARCO Lunar Soil Composition Analysis System

**AI-Driven Terrain Classification & Composition Estimation from Chang'e 3 PCAM Imagery**

---

## 🎯 Project Overview

This system analyzes lunar surface images to estimate soil composition (FeO, TiO2, MgO, SiO2, Al2O3, CaO) using:
- **Calibrated Heuristic Color Ratio Method** (Anchored to CE-3 APXS ground truth)  
- **SAM 2.1 Segmentation** (Precision terrain analysis)
- **Deep Learning ResNet-18** (Terrain classification)
- **Gemini 2.0 LLM** (Scientific geological reporting)

**Status**: ✅ **Phase 4 Complete**, 🚀 **Ready for FYP Presentation**

---

## 🚀 Quick Start

### Run the Analysis System

```bash
# Launch the Mission Control UI
streamlit run src/ui/app.py

# 1. Open http://localhost:8501
# 2. Upload an image from data/pcam/
# 3. View automated composition analysis and terrain mapping
# 4. Generate the AI Scientific Mission Report
```

---

## 📊 System Features

### ✅ Fully Operational
- **457 PCAM Images**: Real data from Chang'e 3 Yutu Rover
- **High-Precision Segmentation**: SAM 2.1 for identifying craters, boulders, and regolith
- **6-Oxide Estimation**: Scientifically validated FeO, TiO2, MgO, SiO2, Al2O3, and CaO wt%
- **Geologic Classifier**: Identifies units like Highland Anorthosite, Mare Basalts, KREEP, etc.
- **AI Mission Reporter**: Deep interpretative reports powered by Gemini 2.0
- **SUPARCO Mission Control UI**: Interactive dashboard with real-time telemetry
- **Physics-Based Constraints**: Enforces oxide-sum conservation and geochemical anticorrelations

---

## 📁 Project Structure

```
├── data/
│   └── pcam/              # 457 downloaded lunar images
├── src/
│   ├── terrain/           # SAM 2.1 + ResNet-18 Classification
│   ├── composition/       # Calibrated Heuristic Estimators
│   ├── analysis/          # Unified Multi-Phase Pipeline
│   ├── llm/               # Gemini Scientific Reporter
│   └── ui/                # Streamlit Mission Control
├── docs/                  # Research reports and scientific basis
└── labeled_data/          # Generated terrain training sets
```

---

## 🧪 Scientific Approach

### Composition Estimation (Lucey + APXS)

Our pipeline uses a hybrid method anchored to **Chang'e-3 APXS in-situ data** (Nature Communications 2015):

1. **TiO2 (Titanium)**: Calibrated Blue/Red ratio (proxy for 415nm/750nm)
2. **FeO (Iron)**: Brightness-based albedo inversion with maturity correction
3. **MgO/SiO2/Al2O3/CaO**: Derived through physics-based geochemical trade-off models
4. **Maturity Correction**: Spatial texture standard deviation (OMAT proxy) adjusts for space weathering

---

## 📚 Mission Research

- **SUPARCO Context**: Designed for ICUBE-Qamar 1MP optical cameras.
- **Geochemical Logic**: Based on Lucey (2000) and Prettyman (2006).
- **Architecture**: Segment Anything Model 2.1 for foundational computer vision.

---

## 🎓 FYP Presentation Highlights

**Key Successes**:
1. ✅ **ICUBE-Q Alignment**: Processes 1MP RGB data exactly as Pakistan's lunar satellite requires.
2. ✅ **Scientific Rigor**: Anchored to real Moon-landing ground truth (APXS), not just visual heuristics.
3. ✅ **End-to-End Flow**: Image → Terrain Map → Chemical Profile → Scientific Report.
4. ✅ **Modern AI Stack**: SAM 2.1, ResNet-18, and Gemini 2.0 Flash.

---

## 👥 Team

**SUPARCO Lunar Exploration Program**  
Developed as a Final Year Project for Satellite Analysis

---

**Status**: Ready for final submission and deployment. 🎉
