# 🚀 Project Plan: AI-Driven Lunar Soil Composition Analysis

---

## 👥 Team Members

-   **Safwan**
-   **Shayaan**

---

## 🎯 Project Overview

This project aims to build an AI-driven pipeline for analyzing lunar soil (regolith) composition using satellite imagery and Large Language Models (LLMs). The system will classify regolith types, detect minerals, and generate explainable natural-language reports to assist in future lunar exploration and resource utilization.

---

## 🧑‍💻 Roles and Responsibilities

To ensure clear ownership and streamline development, the project is divided into the following domains:

### 👨‍🔬 Safwan: Lead - Data Engineering & Vision Modeling

-   **Responsibilities:**
    -   **Data Pipeline:** Manage the collection, preprocessing, and augmentation of lunar imagery from NASA, ISRO, and Clementine datasets.
    -   **Vision Models:** Develop, train, and evaluate the CNN/Vision Transformer models for regolith classification and feature extraction.
    -   **Model Optimization:** Fine-tune models for accuracy and performance.
-   **Primary Focus:** `src/data/`, `src/models/`, `datasets/`

### 👨‍🚀 Shayaan: Lead - LLM Integration & Visualization

-   **Responsibilities:**
    -   **LLM Integration:** Design and implement the pipeline to feed vision model outputs into an LLM (e.g., GPT) for interpretation.
    -   **Report Generation:** Develop the logic for generating natural-language reports based on the LLM's output.
    -   **Visualization:** Create interactive maps, dashboards, and mineral overlays to present the findings.
-   **Primary Focus:** `src/llm_integration/`, `src/visualization/`, `reports/`

---

## 🏛️ Project Architecture

The project is broken down into five main components:

1.  **Data Collection & Preprocessing (`src/data`)**
    -   Scripts to fetch and process data from various lunar missions.
    -   Handles denoising, normalization, and feature extraction.

2.  **Vision Models (`src/models`)**
    -   Contains the CNN and/or ViT architectures for image analysis.
    -   Training and evaluation scripts for the models.

3.  **LLM Integration (`src/llm_integration`)**
    -   Connects the vision model's output to an LLM.
    -   Generates prompts and parses LLM responses to create scientific explanations.

4.  **Visualization (`src/visualization`)**
    -   Generates visual outputs like composition maps and mineral overlays.
    -   Could include a web-based dashboard for interactive analysis.

5.  **Utilities & Configuration (`src/utils`, `config.yaml`)**
    -   Helper functions, metrics, and global configurations.

---

## 🗺️ Project Roadmap & Milestones

This project will be developed in five phases.

### Phase 1: Data Foundation (Estimated Time: 1-2 Weeks)

-   [ ] **Task 1.1 (Safwan):** Finalize data sources and download initial datasets.
-   [ ] **Task 1.2 (Safwan):** Develop and test preprocessing scripts (`src/data/preprocessing.py`).
-   [ ] **Task 1.3 (Shayaan):** Set up the basic project structure and environment.
-   **Milestone:** Be able to load and preprocess a sample lunar image.

### Phase 2: Core Vision Model (Estimated Time: 2-3 Weeks)

-   [ ] **Task 2.1 (Safwan):** Implement and train a baseline CNN model (`src/models/cnn.py`).
-   [ ] **Task 2.2 (Safwan):** Evaluate the model's performance on a test set.
-   [ ] **Task 2.3 (Shayaan):** Research and prototype LLM prompt strategies.
-   **Milestone:** A trained model capable of classifying major regolith types from an image.

### Phase 3: LLM-Powered Explanation (Estimated Time: 2 Weeks)

-   [ ] **Task 3.1 (Shayaan):** Connect the CNN output to the LLM API (`src/llm_integration/interpretation.py`).
-   [ ] **Task 3.2 (Shayaan):** Develop the report generation logic.
-   [ ] **Task 3.3 (Safwan):** Refine the vision model based on initial integration results.
-   **Milestone:** Generate a basic text report for a given lunar image.

### Phase 4: Visualization & User Interface (Estimated Time: 2 Weeks)

-   [ ] **Task 4.1 (Shayaan):** Create functions to plot composition maps and mineral overlays (`src/visualization/plotting.py`).
-   [ ] **Task 4.2 (Shayaan):** (Optional) Develop a simple dashboard to display results.
-   [ ] **Task 4.3 (Safwan):** Prepare the model for integration with the visualization tools.
-   **Milestone:** An image can be processed, analyzed, and visualized with a corresponding report.

### Phase 5: Testing, Validation, and Documentation (Estimated Time: 1-2 Weeks)

-   [ ] **Task 5.1 (Both):** Write comprehensive unit and integration tests (`tests/`).
-   [ ] **Task 5.2 (Both):** Validate model predictions against ground truth data (e.g., Apollo samples).
-   [ ] **Task 5.3 (Both):** Clean up the code, add documentation, and finalize the `README.md`.
-   **Milestone:** A finished, validated, and well-documented prototype.

---

## 🛠️ Tools & Technologies

-   **Programming Language:** Python 3.10+
-   **AI/ML:** TensorFlow, PyTorch, scikit-learn
-   **LLMs:** OpenAI GPT API (or similar)
-   **Data Handling:** NumPy, OpenCV
-   **Visualization:** Matplotlib, Plotly, Folium
-   **Project Management:** This file (`PROJECT_PLAN.md`) will be the source of truth for the project plan.
