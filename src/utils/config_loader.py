"""Configuration loader with environment variable support.

Industry best practice: Centralized configuration management with validation.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from loguru import logger


class ConfigLoader:
    """Load and validate configuration from YAML and environment variables."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_env()
        self._load_yaml()
        self._validate()

    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}. Using environment variables only.")

    def _load_yaml(self) -> None:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")

    def _validate(self) -> None:
        """Validate required configuration and API keys."""
        # Validate API keys from environment
        required_keys = ["OPENAI_API_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            logger.warning(
                f"Missing required API keys: {missing_keys}. "
                "Set them in .env file or environment variables."
            )
        
        # Validate configuration structure
        required_sections = ["dataset", "embeddings", "llm", "rag_configs", "evaluation"]
        missing_sections = [sec for sec in required_sections if sec not in self._config]
        
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {missing_sections}")
        
        logger.info("Configuration validation passed")

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'llm.model')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config = ConfigLoader()
            >>> model = config.get('llm.model')
            >>> model
            'gpt-3.5-turbo'
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config

    def get_api_key(self, service: str) -> str:
        """Get API key from environment variables.
        
        Args:
            service: Service name (openai, cohere, huggingface)
            
        Returns:
            API key
            
        Raises:
            ValueError: If API key not found
        """
        key_map = {
            "openai": "OPENAI_API_KEY",
            "cohere": "COHERE_API_KEY",
            "huggingface": "HF_TOKEN",
        }
        
        env_var = key_map.get(service.lower())
        if not env_var:
            raise ValueError(f"Unknown service: {service}")
        
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"API key for {service} not found. "
                f"Set {env_var} in .env file or environment variables."
            )
        
        return api_key


# Global configuration instance
_config_instance = None


def get_config(config_path: str = "configs/config.yaml") -> ConfigLoader:
    """Get global configuration instance (singleton pattern).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance
