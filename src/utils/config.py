"""
Configuration management for citation prediction project.

This module handles loading and accessing configuration parameters
from YAML configuration files.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Configuration manager for the project."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration YAML file.
                        If None, uses default config/config.yaml
        """
        if config_path is None:
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Dictionary containing configuration parameters.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (supports nested keys with dot notation,
                 e.g., 'data.raw_dir')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """
        Get configuration value using dictionary notation.

        Args:
            key: Configuration key

        Returns:
            Configuration value
        """
        return self.get(key)

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(config_path='{self.config_path}')"


# Global configuration instance
config = Config()
