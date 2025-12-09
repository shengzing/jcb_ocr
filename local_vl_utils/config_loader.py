"""Environment configuration loader for VLM clients.

This module provides centralized configuration management for all VLM client types.
All model configurations (model_name, server_url, api_key) are loaded from .env file
at process startup time.

Usage:
    from local_vl_utils.config_loader import get_client_config

    # Get configuration for a specific client type
    config = get_client_config("qwen")
    print(config.model_name)  # wen/Qwen3-VL-8B-Instruct
    print(config.server_url)  # https://api.siliconflow.cn
    print(config.api_key)     # sk-xxx...
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


@dataclass
class ClientConfig:
    """Configuration for a VLM client."""

    model_name: Optional[str] = None
    server_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 300
    max_retries: int = 3
    temperature: float = 0.0
    top_p: float = 0.01

    def get_headers(self) -> Optional[Dict[str, str]]:
        """Get HTTP headers with API key if available."""
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return None


class ConfigLoader:
    """Central configuration loader for all VLM clients."""

    # Environment variable name mapping
    ENV_MAPPING = {
        "qwen": {
            "model_name": "QWEN_MODEL_NAME",
            "server_url": "QWEN_SERVER_URL",
            "api_key": "QWEN_API_KEY",
        },
        "deepseek": {
            "model_name": "DEEPSEEK_MODEL_NAME",
            "server_url": "DEEPSEEK_SERVER_URL",
            "api_key": "DEEPSEEK_API_KEY",
        },
        "hunyuan": {
            "model_name": "HUNYUAN_MODEL_NAME",
            "server_url": "HUNYUAN_SERVER_URL",
            "api_key": "HUNYUAN_API_KEY",
        },
        "mineru": {
            "model_name": "MINERU_MODEL_NAME",
            "server_url": "MINERU_SERVER_URL",
            "api_key": "MINERU_API_KEY",
        },
    }

    # Common environment variables
    COMMON_ENV_VARS = {
        "timeout": "DEFAULT_TIMEOUT",
        "max_retries": "DEFAULT_MAX_RETRIES",
        "temperature": "DEFAULT_TEMPERATURE",
        "top_p": "DEFAULT_TOP_P",
    }

    _env_loaded = False

    @classmethod
    def load_env_file(cls, env_file: Optional[Path] = None) -> None:
        """Load environment variables from .env file.

        Args:
            env_file: Path to .env file. If None, looks for .env in parent directory.
        """
        if cls._env_loaded:
            return

        if env_file is None:
            # Look for .env in the vlm_client directory
            env_file = Path(__file__).parent.parent / ".env"

        if not env_file.exists():
            logger.warning(f".env file not found at {env_file}, using environment variables only")
            cls._env_loaded = True
            return

        logger.info(f"Loading configuration from {env_file}")

        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Only set if not already in environment
                    if key and not os.getenv(key):
                        os.environ[key] = value

        cls._env_loaded = True
        logger.debug("Environment configuration loaded successfully")

    @classmethod
    def get_client_config(
        cls,
        client_type: str,
        model_name: Optional[str] = None,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> ClientConfig:
        """Get configuration for a specific client type.

        Args:
            client_type: Client type ("qwen", "deepseek", "hunyuan")
            model_name: Override model name (optional)
            server_url: Override server URL (optional)
            api_key: Override API key (optional)
            **kwargs: Additional config overrides (timeout, max_retries, etc.)

        Returns:
            ClientConfig with all settings loaded from environment or overrides

        Raises:
            ValueError: If client_type is unknown
        """
        # Ensure .env file is loaded
        cls.load_env_file()

        if client_type not in cls.ENV_MAPPING:
            available = list(cls.ENV_MAPPING.keys())
            raise ValueError(f"Unknown client type '{client_type}'. Available: {available}")

        # Get client-specific environment variable names
        env_vars = cls.ENV_MAPPING[client_type]

        # Load configuration with priority: parameter > env var > default
        config = ClientConfig(
            model_name=model_name or os.getenv(env_vars["model_name"]),
            server_url=server_url or os.getenv(env_vars["server_url"]),
            api_key=api_key or os.getenv(env_vars["api_key"]),
        )

        # Load common configuration
        for attr, env_var in cls.COMMON_ENV_VARS.items():
            if attr in kwargs:
                # Use provided override
                setattr(config, attr, kwargs[attr])
            else:
                # Try to load from environment
                env_value = os.getenv(env_var)
                if env_value:
                    # Convert to appropriate type
                    if attr in ("timeout", "max_retries"):
                        setattr(config, attr, int(env_value))
                    elif attr in ("temperature", "top_p"):
                        setattr(config, attr, float(env_value))

        # Log configuration (hide API key)
        logger.debug(
            f"Loaded config for {client_type}: "
            f"model={config.model_name}, "
            f"url={config.server_url}, "
            f"api_key={'***' if config.api_key else 'None'}"
        )

        return config

    @classmethod
    def get_default_client_type(cls) -> str:
        """Get default client type from environment.

        Returns:
            Default client type ("qwen" if not specified)
        """
        cls.load_env_file()
        return os.getenv("DEFAULT_CLIENT_TYPE", "qwen")


# Convenience functions
def get_client_config(
    client_type: str,
    model_name: Optional[str] = None,
    server_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> ClientConfig:
    """Get configuration for a specific client type.

    This is a convenience wrapper around ConfigLoader.get_client_config().

    Args:
        client_type: Client type ("qwen", "deepseek", "hunyuan")
        model_name: Override model name (optional)
        server_url: Override server URL (optional)
        api_key: Override API key (optional)
        **kwargs: Additional config overrides

    Returns:
        ClientConfig with all settings
    """
    return ConfigLoader.get_client_config(
        client_type,
        model_name=model_name,
        server_url=server_url,
        api_key=api_key,
        **kwargs
    )


def load_env_file(env_file: Optional[Path] = None) -> None:
    """Load environment variables from .env file.

    Args:
        env_file: Path to .env file (optional)
    """
    ConfigLoader.load_env_file(env_file)


def get_default_client_type() -> str:
    """Get default client type from environment."""
    return ConfigLoader.get_default_client_type()


__all__ = [
    "ClientConfig",
    "ConfigLoader",
    "get_client_config",
    "load_env_file",
    "get_default_client_type",
]
