# SUPARCO Dataset Integration

## Purpose
This folder is reserved for **SUPARCO-provided labeled lunar imagery** and annotations.

## Expected Format

### Directory Structure
```
suparco/
├── images/
│   ├── IMG_001.png
│   ├── IMG_002.png
│   └── ...
└── annotations.csv
```

### Annotations CSV Format
The `annotations.csv` file should follow this schema:

```csv
filename,terrain_class,fe_percent,mg_percent,ti_percent,si_percent,moisture_level,notes
IMG_001.png,regolith,8.5,4.2,1.3,45.2,low,Mare region sample
IMG_002.png,crater,12.1,3.8,2.1,42.0,none,Fresh impact crater
IMG_003.png,boulder,6.2,5.1,0.9,48.5,trace,Highland rock
```

**Columns:**
- `filename`: Image filename (must exist in `images/` folder)
- `terrain_class`: One of: `regolith`, `crater`, `boulder`, `mixed`
- `fe_percent`: Iron (Fe) percentage (0-100, or NA)
- `mg_percent`: Magnesium (Mg) percentage (0-100, or NA)
- `ti_percent`: Titanium (Ti) percentage (0-100, or NA)
- `si_percent`: Silicon (Si) percentage (0-100, or NA)
- `moisture_level`: One of: `none`, `trace`, `low`, `medium`, `high`, `NA`
- `notes`: Optional descriptive notes

## Integration
Once you add data to this folder:
1. Place images in `suparco/images/`
2. Create/update `suparco/annotations.csv`
3. Restart the application or run: `python src/data/label_importer.py`

The system will automatically detect and use SUPARCO data for training.

## Contact
For labeling assistance or format questions, contact the FYP supervisor.
