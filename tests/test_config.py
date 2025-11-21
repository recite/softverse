"""Test configuration module."""

from softverse.config import get_config


def test_config_loads_default():
    """Test that config loads default settings."""
    config = get_config()
    assert config.get("processing.parallel_workers", 0) > 0


def test_config_get_with_default():
    """Test getting config values with defaults."""
    config = get_config()

    # Test existing value
    assert isinstance(config.get("processing.parallel_workers"), int)

    # Test non-existing value with default
    assert config.get("non.existing.key", "default") == "default"


def test_config_api_token_methods():
    """Test API token retrieval methods."""
    config = get_config()

    # Should return None for non-existent tokens
    token = config.get_api_token("nonexistent")
    assert token is None


def test_config_properties():
    """Test configuration property accessors."""
    config = get_config()

    # Test that properties return dictionaries
    assert isinstance(config.dataverse_config, dict)
    assert isinstance(config.zenodo_config, dict)
    assert isinstance(config.icpsr_config, dict)
    assert isinstance(config.processing_config, dict)
    assert isinstance(config.output_config, dict)
