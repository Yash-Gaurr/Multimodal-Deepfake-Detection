"""
Configuration loader and validator for Multimodal Deepfake Detection.
"""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Loads configuration settings from YAML file.
    
    Args:
        config_path: Optional path to config file. Defaults to config/config.yaml.
        
    Returns:
        Dictionary containing configuration parameters.
    """
    if config_path is None:
        root = get_project_root()
        config_path = os.path.join(root, "config", "config.yaml")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config


class ConfigDict(dict):
    """Dot-notation accessible dictionary for convenient attribute access."""
    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        except KeyError:
            raise AttributeError(f"No configuration attribute named '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def get_config(config_path: str = None) -> ConfigDict:
    """Returns configuration as a dot-notation accessible object."""
    raw_config = load_config(config_path)
    return ConfigDict(raw_config)
