"""
Configuration management utilities.
"""

import yaml
from typing import Dict, Any
import os


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_env_variable(var_name: str, default: Any = None) -> Any:
    """
    Get environment variable with optional default.
    
    Args:
        var_name: Name of environment variable
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    return os.environ.get(var_name, default)
