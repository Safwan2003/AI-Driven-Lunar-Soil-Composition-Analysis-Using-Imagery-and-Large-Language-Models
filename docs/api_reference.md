# API Reference

## Data Module (`src.data`)

### `load_lunar_imagery(data_dir, image_size)`

Load lunar imagery from a directory.

**Parameters:**
- `data_dir` (str): Path to directory containing images
- `image_size` (tuple, optional): Tuple (height, width) to resize images

**Returns:**
- `np.ndarray`: Array of loaded images

### `preprocess_image(image)`

Preprocess lunar imagery for model input.

**Parameters:**
- `image` (np.ndarray): Raw image array

**Returns:**
- `np.ndarray`: Preprocessed image array

## Models Module (`src.models`)

### `LunarVisionModel`

Vision model for analyzing lunar soil composition from imagery.

**Methods:**
- `__init__(config)`: Initialize the model with configuration
- `forward(x)`: Forward pass through the model

### `train_model(config_path)`

Train a vision model for lunar soil analysis.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `LunarVisionModel`: Trained model

## LLM Integration Module (`src.llm_integration`)

### `LLMAnalyzer`

Integrates Large Language Models for reasoning and report generation.

**Methods:**

#### `__init__(config)`
Initialize the LLM analyzer.

**Parameters:**
- `config` (dict): LLM configuration dictionary

#### `analyze_composition(image_path, vision_results)`
Analyze lunar soil composition using LLM reasoning.

**Parameters:**
- `image_path` (str): Path to lunar image
- `vision_results` (dict, optional): Optional results from vision model

**Returns:**
- `dict`: Analysis results with interpretation

#### `generate_report(analysis_results)`
Generate a comprehensive report from analysis results.

**Parameters:**
- `analysis_results` (dict): Results from composition analysis

**Returns:**
- `str`: Formatted report text

## Visualization Module (`src.visualization`)

### `plot_composition(results, save_path)`

Plot lunar soil composition analysis results.

**Parameters:**
- `results` (dict): Analysis results dictionary
- `save_path` (str, optional): Optional path to save the plot

### `visualize_predictions(image, predictions, save_path)`

Visualize model predictions overlaid on the original image.

**Parameters:**
- `image` (np.ndarray): Original lunar image
- `predictions` (dict): Model predictions
- `save_path` (str, optional): Optional path to save the visualization

## Utils Module (`src.utils`)

### `load_config(config_path)`

Load configuration from YAML file.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `dict`: Configuration dictionary

### `save_config(config, config_path)`

Save configuration to YAML file.

**Parameters:**
- `config` (dict): Configuration dictionary
- `config_path` (str): Path to save configuration

### `get_env_variable(var_name, default)`

Get environment variable with optional default.

**Parameters:**
- `var_name` (str): Name of environment variable
- `default` (any): Default value if not found

**Returns:**
- `any`: Environment variable value or default
