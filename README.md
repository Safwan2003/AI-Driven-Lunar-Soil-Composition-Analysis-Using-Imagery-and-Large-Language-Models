# AI-Driven Lunar Soil Composition Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

## 🌖 Project Overview
This project is an AI-powered framework designed to analyze **Lunar Rover RGB Imagery**. It integrates Deep Learning (CNN/ResNet) for terrain classification and soil composition estimation with a **Large Language Model (LLM)** to generate scientific reports.

**Key Goals:**
- **Terrain Classification**: Detect Craters, Boulders, and Regolith.
- **Soil Analysis**: Estimate elemental composition (Fe, Ti, Si) from visual features.
- **Automated Reporting**: Generate natural language summaries of findings.

## 📂 Project Structure
```
.
├── app/
│   └── main.py              # Streamlit Frontend Application
├── data/                    # Dataset storage (created via notebook)
├── models/
│   └── terrain_classifier.py # PyTorch Model Definition
├── notebooks/
│   └── 01_Data_Acquisition_and_Training.ipynb # Colab Notebook for Data & Training
├── src/
│   └── llm_engine.py        # Logic for LLM Report Generation
├── requirements.txt         # Project Dependencies
└── README.md                # Project Documentation
```

## 🚀 Getting Started

### 1. Data Acquisition & Training (Google Colab)
Since the dataset is large (Chang'e 3 Mission Data), we use **Google Colab** for downloading and training.

1.  Open `notebooks/01_Data_Acquisition_and_Training.ipynb`.
2.  Upload it to [Google Colab](https://colab.research.google.com/) or run it locally if you have a GPU.
3.  Run all cells to:
    - Download PCAM/TCAM images from The Planetary Society.
    - Train the `ResNet-18` model.
    - Save the model weights to `models/lunar_terrain_model.pth`.

### 2. Running the Web Application (Local)
Once you have the model (or just to test the UI):

**Prerequisites:**
- Python 3.8+ installed.
- Virtual Environment set up.

**Installation:**
```bash
# 1. Create Virtual Env
python -m venv venv

# 2. Activate Env
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install Requirements
pip install -r requirements.txt
```

**Launch App:**
```bash
streamlit run app/main.py
```

## 🛠 Features
- **Interactive Dashboard**: Upload images and view real-time analysis.
- **Deep Learning Inference**: Uses ResNet-18 for robust terrain segmentation.
- **Scientific Reporting**: Mock LLM integration (extensible to OpenAI GPT-4V) provides context-aware summaries.

## 📚 Data Source
Data is sourced from the **Chang'e 3** mission via **The Planetary Society** mirrors.
- [PCAM Data Link](http://planetary.s3.amazonaws.com/data/change3/pcam.html)
- [TCAM Data Link](http://planetary.s3.amazonaws.com/data/change3/tcam.html)

## 📄 License
This project is for educational purposes (FYP).
