# Reports

This directory contains generated analysis reports and figures.

## Structure

```
reports/
├── figures/          # Generated plots and visualizations
├── analysis/         # Analysis reports
└── README.md         # This file
```

## Report Types

- **Markdown Reports**: Human-readable analysis summaries
- **HTML Reports**: Interactive web-based reports
- **PDF Reports**: Formatted documents for sharing
- **Figures**: Charts, plots, and visualizations

## Generating Reports

Use the `ReportGenerator` class:

```python
from visualization import ReportGenerator

generator = ReportGenerator()
generator.create_report(
    predictions=predictions,
    interpretations=interpretations,
    output_path="reports/analysis/lunar_analysis.md"
)
```

## Notes

- Reports are git-ignored by default (see `.gitignore`)
- Keep reports organized by date or experiment
- Archive important reports separately
