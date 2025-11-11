# Getting Started Guide

## Introduction

This guide will help you get started with the AI-Driven Lunar Soil Composition Analysis project.

## Quick Start

### 1. Installation

Follow the installation steps in the main [README.md](../README.md).

### 2. Configuration

Edit `config.yaml` to set up your environment:

```yaml
# Set your LLM API key environment variable
llm:
  api_key_env: "OPENAI_API_KEY"
```

Set the environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Prepare Data

Place your lunar imagery in the `datasets/` directory:

```
datasets/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

### 4. Run Analysis

```python
from src.data import load_lunar_imagery
from src.models import LunarVisionModel
from src.llm_integration import LLMAnalyzer
from src.utils import load_config

# Load configuration
config = load_config('config.yaml')

# Load data
images = load_lunar_imagery('datasets/train')

# Initialize model
model = LunarVisionModel(config['model'])

# Analyze with LLM
analyzer = LLMAnalyzer(config['llm'])
results = analyzer.analyze_composition('path/to/image.jpg')
```

## Next Steps

- Explore the [Jupyter notebooks](../notebooks/) for examples
- Read the [API documentation](api_reference.md)
- Check out example [workflows](workflows.md)
