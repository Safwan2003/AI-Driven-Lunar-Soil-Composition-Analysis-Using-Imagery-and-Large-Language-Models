"""
Tests for configuration utilities.
"""

import pytest
import yaml
import tempfile
import os
from src.utils.config import load_config, save_config, get_env_variable


def test_load_config():
    """Test loading configuration from YAML file."""
    # Create a temporary config file
    config_data = {
        'project': {'name': 'test_project'},
        'paths': {'data_dir': 'test_data'}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        loaded_config = load_config(temp_path)
        assert loaded_config['project']['name'] == 'test_project'
        assert loaded_config['paths']['data_dir'] == 'test_data'
    finally:
        os.unlink(temp_path)


def test_save_config():
    """Test saving configuration to YAML file."""
    config_data = {
        'test_key': 'test_value',
        'nested': {'key': 'value'}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        save_config(config_data, temp_path)
        
        # Load and verify
        with open(temp_path, 'r') as f:
            loaded = yaml.safe_load(f)
        
        assert loaded['test_key'] == 'test_value'
        assert loaded['nested']['key'] == 'value'
    finally:
        os.unlink(temp_path)


def test_get_env_variable():
    """Test getting environment variables."""
    # Set a test environment variable
    os.environ['TEST_VAR'] = 'test_value'
    
    # Test retrieving existing variable
    assert get_env_variable('TEST_VAR') == 'test_value'
    
    # Test retrieving non-existing variable with default
    assert get_env_variable('NON_EXISTENT_VAR', 'default') == 'default'
    
    # Test retrieving non-existing variable without default
    assert get_env_variable('NON_EXISTENT_VAR') is None
    
    # Clean up
    del os.environ['TEST_VAR']
