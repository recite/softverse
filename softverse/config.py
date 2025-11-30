"""Configuration management for Softverse."""

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Configuration manager for Softverse."""

    def __init__(self, config_path: str | None = None):
        """Initialize configuration.

        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            # Default to config/settings.yaml relative to project root
            project_root = Path(__file__).parent.parent
            default_config = project_root / "config" / "settings.yaml"
            self.config_path = default_config
        else:
            self.config_path = Path(config_path)
        self._config = self._load_config()
        self._setup_directories()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _setup_directories(self) -> None:
        """Create necessary directories based on configuration."""
        dirs_to_create = [
            self.get("output.base_dir"),
            self.get("data_sources.dataverse.output_dir"),
            self.get("data_sources.zenodo.output_dir"),
            self.get("data_sources.icpsr.output_dir"),
            self.get("incremental.checkpoints_dir"),
            os.path.dirname(self.get("logging.file")),
        ]

        for dir_path in dirs_to_create:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'data_sources.dataverse.enabled')
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.

        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_api_token(self, service: str) -> str | None:
        """Get API token from environment variable.

        Args:
            service: Service name (e.g., 'dataverse', 'zenodo', 'osf')

        Returns:
            API token or None
        """
        # Special case for OSF which uses different env var name
        if service.lower() == "osf":
            env_var = "OSF_API_TOKEN"
        else:
            env_var = f"{service.upper()}_TOKEN"

        token = os.getenv(env_var)

        if not token:
            # Try loading from token file
            token_file = f"{service}_token.txt"
            if os.path.exists(token_file):
                with open(token_file) as f:
                    token = f.read().strip()

        return token

    @property
    def dataverse_config(self) -> dict[str, Any]:
        """Get Dataverse configuration."""
        return self.get("data_sources.dataverse", {})

    @property
    def zenodo_config(self) -> dict[str, Any]:
        """Get Zenodo configuration."""
        return self.get("data_sources.zenodo", {})

    @property
    def icpsr_config(self) -> dict[str, Any]:
        """Get ICPSR configuration."""
        return self.get("data_sources.icpsr", {})

    @property
    def osf_config(self) -> dict[str, Any]:
        """Get OSF configuration."""
        return self.get("data_sources.osf", {})

    @property
    def researchbox_config(self) -> dict[str, Any]:
        """Get ResearchBox configuration."""
        return self.get("data_sources.researchbox", {})

    @property
    def processing_config(self) -> dict[str, Any]:
        """Get processing configuration."""
        return self.get("processing", {})

    @property
    def output_config(self) -> dict[str, Any]:
        """Get output configuration."""
        return self.get("output", {})


# Global configuration instance
_config: Config | None = None


def get_config(config_path: str | None = None) -> Config:
    """Get global configuration instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration instance
    """
    global _config
    if _config is None or config_path is not None:
        _config = Config(config_path)
    return _config


def reload_config(config_path: str | None = None) -> Config:
    """Reload configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        New configuration instance
    """
    global _config
    _config = Config(config_path)
    return _config
