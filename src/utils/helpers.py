# src/utils/helpers.py

"""
This script contains helper functions and utility classes used across the project.

Functions:
- load_config: Loads the main configuration file.
- save_model: Saves a trained model to disk.
- calculate_metrics: Computes evaluation metrics for the model.
"""

import yaml
import torch

def load_config(config_path='config.yaml'):
    """
    Loads the YAML configuration file.
    
    Args:
        config_path (str): The path to the config file.
        
    Returns:
        A dictionary with the configuration settings.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_model(model, filepath):
    """
    Saves a PyTorch model to a file.
    
    Args:
        model: The PyTorch model to save.
        filepath (str): The path where the model will be saved.
    """
    print(f"Saving model to {filepath}...")
    torch.save(model.state_dict(), filepath)

def calculate_metrics(y_true, y_pred):
    """
    Calculates accuracy, precision, and recall.
    
    Args:
        y_true: The ground truth labels.
        y_pred: The predicted labels.
        
    Returns:
        A dictionary containing the calculated metrics.
    """
    # Placeholder for metric calculation logic
    accuracy = (y_true == y_pred).mean()
    return {"accuracy": accuracy}
