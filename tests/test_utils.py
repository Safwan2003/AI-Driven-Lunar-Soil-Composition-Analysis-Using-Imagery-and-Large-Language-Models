"""
Test Utils Module
=================

Tests for utility functions.
"""

import pytest
import tempfile
from pathlib import Path

from src.utils import load_config, setup_logger


def test_load_config():
    """Test configuration loading."""
    # Test with default config
    config = load_config("config.yaml")
    assert config is not None
    assert "project" in config
    assert "data" in config
    assert "models" in config


def test_load_config_file_not_found():
    """Test config loading with non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_setup_logger():
    """Test logger setup."""
    logger = setup_logger(name="test_logger", level="INFO")
    assert logger is not None
    assert logger.name == "test_logger"


def test_setup_logger_with_file():
    """Test logger setup with file output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = setup_logger(
            name="test_logger_file",
            level="DEBUG",
            log_file=str(log_file)
        )
        
        logger.info("Test message")
        assert log_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
