"""
Configuration management for Algua trading platform.
"""

# For backward compatibility with the old configuration system
from typing import Any
from configs.settings import get_settings as _get_new_settings


def get_settings():
    """Get application settings (backward compatibility)."""
    return _get_new_settings()


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value (backward compatibility)."""
    settings = get_settings()
    return getattr(settings, key, default) 