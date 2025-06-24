"""
Settings factory and configuration management.
"""

import os
from typing import Type, Union
from functools import lru_cache

from .environments.base import BaseConfig
from .environments.development import DevelopmentConfig
from .environments.production import ProductionConfig
from .environments.testing import TestingConfig


def get_config_class() -> Type[BaseConfig]:
    """Get configuration class based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig,
        "test": TestingConfig,  # Alias for testing
    }
    
    return config_map.get(env, DevelopmentConfig)


@lru_cache()
def get_settings() -> BaseConfig:
    """Get cached settings instance."""
    config_class = get_config_class()
    return config_class()


def validate_settings() -> bool:
    """Validate settings and return True if valid."""
    settings = get_settings()
    
    # Check for missing secrets
    missing_secrets = settings.validate_required_secrets()
    if missing_secrets:
        print(f"Missing required environment variables: {', '.join(missing_secrets)}")
        return False
    
    # Additional production validation
    if isinstance(settings, ProductionConfig):
        production_issues = settings.validate_production_requirements()
        if production_issues:
            print(f"Production validation issues: {', '.join(production_issues)}")
            return False
    
    return True


# Convenience function for backward compatibility
def get_config() -> BaseConfig:
    """Alias for get_settings()."""
    return get_settings()