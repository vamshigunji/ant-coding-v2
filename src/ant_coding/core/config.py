"""
Configuration and environment loading utilities.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass

def get_env(key: str, default: str = None) -> str:
    """
    Get an environment variable or raise ConfigError if it's missing and no default is provided.
    
    Args:
        key: The environment variable key.
        default: The default value to return if the key is missing.
        
    Returns:
        The value of the environment variable.
        
    Raises:
        ConfigError: If the key is missing and no default is provided.
    """
    value = os.getenv(key, default)
    if value is None:
        raise ConfigError(f"Missing required environment variable: {key}")
    return value
