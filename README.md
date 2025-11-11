# AI-Driven Lunar Soil Composition Analysis Using Imagery and Large Language Models

AI-driven analysis of lunar soil composition using satellite imagery and Large Language Models (LLMs). Combines vision models and LLM reasoning to classify regolith, map minerals, and generate explainable reports for lunar exploration and resource planning.

## 🌙 Overview

This project leverages state-of-the-art computer vision models and Large Language Models to analyze lunar surface imagery and provide detailed compositional analysis of lunar soil. The system combines:

- **Computer Vision**: Deep learning models for analyzing lunar surface imagery
- **LLM Integration**: Large Language Models for reasoning, interpretation, and report generation
- **Mineral Classification**: Automated detection and classification of lunar regolith composition
- **Explainable AI**: Human-readable reports and visualizations for scientific insights

## 🏗️ Project Structure

```
.
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   ├── models/                   # Computer vision models
│   ├── llm_integration/          # LLM integration and prompting
│   ├── visualization/            # Visualization utilities
│   └── utils/                    # Utility functions
├── notebooks/                    # Jupyter notebooks for experiments
├── datasets/                     # Dataset storage (not tracked in git)
├── reports/                      # Generated analysis reports
├── docs/                         # Documentation
├── tests/                        # Unit and integration tests
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation script
├── LICENSE                       # License information
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for model training)
- 8GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Safwan2003/AI-Driven-Lunar-Soil-Composition-Analysis-Using-Imagery-and-Large-Language-Models.git
cd AI-Driven-Lunar-Soil-Composition-Analysis-Using-Imagery-and-Large-Language-Models
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

### Configuration

Edit `config.yaml` to configure:
- Model paths and parameters
- LLM API keys and endpoints
- Data paths
- Output directories

## 📊 Usage

### Data Processing

```python
from src.data import load_lunar_imagery

# Load and preprocess lunar imagery
images = load_lunar_imagery('datasets/lunar_images/')
```

### Model Training

```python
from src.models import train_model

# Train a vision model
model = train_model(config_path='config.yaml')
```

### LLM Integration

```python
from src.llm_integration import analyze_composition

# Generate analysis report
report = analyze_composition(image_path='path/to/image.jpg')
```

### Visualization

```python
from src.visualization import plot_composition

# Visualize composition analysis
plot_composition(results, save_path='reports/analysis.png')
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

## 📓 Notebooks

Explore the `notebooks/` directory for:
- Data exploration and visualization
- Model training experiments
- LLM prompt engineering
- End-to-end analysis pipelines

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Research Applications

This project supports:
- Lunar resource mapping and planning
- Automated mineral classification
- Scientific report generation
- Mission planning and site selection
- Educational demonstrations of AI in space exploration

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{lunar_soil_analysis,
  title={AI-Driven Lunar Soil Composition Analysis Using Imagery and LLMs},
  author={Your Name},
  year={2024},
  url={https://github.com/Safwan2003/AI-Driven-Lunar-Soil-Composition-Analysis-Using-Imagery-and-Large-Language-Models}
}
```

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

## 🙏 Acknowledgments

- NASA for lunar imagery datasets
- The open-source AI/ML community
- Contributors and researchers in planetary science
