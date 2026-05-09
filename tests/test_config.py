"""Tests for configuration loader."""

import pytest
from src.utils.config_loader import ConfigLoader


def test_config_loader_initialization():
    """Test that config loader initializes successfully."""
    config = ConfigLoader()
    assert config.config is not None


def test_config_get():
    """Test getting configuration values."""
    config = ConfigLoader()
    
    # Test existing key
    model = config.get("llm.model")
    assert model is not None
    
    # Test default value
    nonexistent = config.get("nonexistent.key", "default_value")
    assert nonexistent == "default_value"


def test_config_sections():
    """Test that required configuration sections exist."""
    config = ConfigLoader()
    
    required_sections = ["dataset", "embeddings", "llm", "rag_configs", "evaluation"]
    for section in required_sections:
        assert section in config.config
