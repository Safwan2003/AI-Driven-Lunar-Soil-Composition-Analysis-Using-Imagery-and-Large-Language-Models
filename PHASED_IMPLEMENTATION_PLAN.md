# Phased Implementation Plan: Lunar Soil Composition Analysis

This document outlines the step-by-step roadmap for developing the **AI and LLM-Based Lunar Surface Analysis** system.

---

## 🗓️ Phase 1: Data Acquisition & Labeling (Milestone M1)
**Goal:** Establish the dataset foundation for training models.

- [x] **Project Setup**: Initialize directory structure and Python environment.
- [x] **Data Sourcing Logic**: Create scripts to scrape/download imagery from **Chang'e 3** repositories (The Planetary Society).
- [ ] **Data Downloading**: Execute the download script (via Colab) to populate `data/pcam` and `data/tcam`.
- [ ] **Labeling**:
    - *Auto-Labeling (Provisional)*: Use unsupervised clustering or pre-trained segmentation models to generate initial masks for Craters vs. Flat Regolith.
    - *Manual Correction*: Verify a subset of labels against ground truth maps.

**Deliverables:**
- `notebooks/01_Data_Acquisition_and_Training.ipynb` (Completed)
- Raw Dataset (~35GB potential, subset for FYP).

---

## 🧠 Phase 2: Model Development (Milestone M2)
**Goal:** Train Deep Learning models to "see" and "understand" the lunar surface.

- [x] **Architecture Design**: Define `LunarTerrainClassifier` using **ResNet-18** (PyTorch).
- [ ] **Training Pipeline**:
    - Implement Data Loaders with augmentation (RandomRotate, Flip, Normalize).
    - Train the Terrain Classifier on the PCAM dataset.
    - *Target Metric*: >85% Accuracy on validation set.
- [ ] **Composition Estimation (Extension)**:
    - Train a secondary regression head to predict Soil Composition (Fe, Ti abundant) based on color/spectral features.

**Deliverables:**
- Trained Model Weights: `models/lunar_terrain_model.pth`.
- Training Performance Graphs (Loss/Accuracy).

---

## 🤖 Phase 3: LLM Integration (Milestone M3)
**Goal:** Give the AI a "voice" to explain its findings scientifically.

- [x] **Module Setup**: Create `src/llm_engine.py` structure.
- [ ] **Prompt Engineering**:
    - Design prompts that take Model Outputs (e.g., "Crater detected", "High Ti") and convert them into a geologist-style report.
- [ ] **API Integration**:
    - Connect to **OpenAI API (GPT-4 Vision)** or a local LLaVA instance.
    - *Current State*: Mock implementation provided for UI testing.

**Deliverables:**
- Functional `LLMEngine` class.
- Sample generated reports.

---

## 💻 Phase 4: Application Development (Milestone M4)
**Goal:** Create a user-friendly interface for end-users.

- [x] **Frontend Setup**: Initialize **Streamlit** application (`app/main.py`).
- [x] **UI Layout**: Design Sidebar (Controls) and Main Area (Image/Results).
- [ ] **Real-time Inference**:
    - Load the trained `.pth` model in the app.
    - Pass user-uploaded images through the model.
    - Visualize the output (e.g., draw bounding boxes or heatmaps on the image).
- [ ] **Report Display**: Show the LLM-generated text alongside the visual results.

**Deliverables:**
- Fully functional Web App accessible via `streamlit run app/main.py`.

---

## ✅ Phase 5: Testing & Finalization
- [ ] **Integration Testing**: Verify the flow from Image Upload -> Model Inference -> LLM Report.
- [ ] **Documentation**: Finalize `README.md` and User Manual.
