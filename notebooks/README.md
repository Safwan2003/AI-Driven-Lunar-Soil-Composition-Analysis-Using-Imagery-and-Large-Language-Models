# Notebooks

This directory contains Jupyter notebooks for exploration, experimentation, and visualization.

## Getting Started

Start Jupyter:
```bash
jupyter notebook
```

Or use JupyterLab:
```bash
jupyter lab
```

## Notebook Organization

Recommended naming convention:
- `01_data_exploration.ipynb` - Data loading and exploration
- `02_preprocessing.ipynb` - Data preprocessing pipeline
- `03_model_training.ipynb` - Model training and evaluation
- `04_llm_integration.ipynb` - LLM integration experiments
- `05_visualization.ipynb` - Results visualization
- `06_end_to_end_pipeline.ipynb` - Complete workflow

## Best Practices

1. **Clear Outputs**: Clear notebook outputs before committing to git
2. **Documentation**: Add markdown cells explaining your approach
3. **Reproducibility**: Set random seeds for reproducible results
4. **Modular Code**: Move reusable code to `src/` modules
5. **Version Control**: Use descriptive commit messages for notebook changes

## Tips

- Use `%load_ext autoreload` and `%autoreload 2` to auto-reload modules
- Keep notebooks focused on a single task or experiment
- Export production-ready code to Python scripts in `src/`
