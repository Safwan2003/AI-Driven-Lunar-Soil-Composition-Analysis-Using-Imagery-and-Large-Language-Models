# Datasets

This directory contains lunar imagery datasets for training and evaluation.

## Structure

```
datasets/
├── raw/              # Raw, unprocessed lunar imagery
├── processed/        # Preprocessed and augmented data
├── vector_db/        # Vector database for embeddings
└── README.md         # This file
```

## Data Sources

Add your lunar soil imagery datasets here. Supported formats:
- JPEG/JPG
- PNG
- TIFF/TIF
- HDF5
- NumPy arrays

## Dataset Organization

Organize your data as follows:

```
raw/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

## Notes

- Large datasets are excluded from git (see `.gitignore`)
- Maintain data privacy and licensing requirements
- Document data sources and preprocessing steps
