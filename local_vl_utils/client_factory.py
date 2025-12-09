"""OCR Client Factory and Registration System.

This module provides a unified factory pattern for creating different OCR clients
(Qwen, DeepSeek, Hunyuan, etc.) with automatic registration and discovery.

Usage:
    # Create clients by name
    client = create_ocr_client("qwen", backend="http-client", ...)
    client = create_ocr_client("deepseek", backend="http-client", ...)
    client = create_ocr_client("hunyuan", backend="http-client", ...)

    # List available clients
    available = list_available_clients()
    print(available)  # ["qwen", "deepseek", "hunyuan"]

    # Get client info
    info = get_client_info("qwen")
    print(info.description)  # "Qwen-VL OCR client with multi-mode support"
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from loguru import logger

from .base_client import BaseOCRClient

@dataclass
class ClientInfo:
    """Information about a registered OCR client."""

    name: str
    description: str
    client_class: Type[BaseOCRClient]
    supported_backends: List[str]
    default_prompts: Dict[str, str]
    model_capabilities: Dict[str, Any]


class OCRClientRegistry:
    """Registry for OCR client implementations."""

    def __init__(self):
        self._clients: Dict[str, ClientInfo] = {}
        self._factories: Dict[str, Callable[..., BaseOCRClient]] = {}

    def register_client(
        self,
        name: str,
        description: str,
        client_class: Type[BaseOCRClient],
        factory: Callable[..., BaseOCRClient],
        supported_backends: List[str],
        default_prompts: Dict[str, str],
        model_capabilities: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new OCR client.

        Args:
            name: Unique name for the client (e.g., "qwen", "deepseek")
            description: Human-readable description
            client_class: The client class
            factory: Factory function that creates client instances
            supported_backends: List of supported backend types
            default_prompts: Default prompts for different task types
            model_capabilities: Model-specific capabilities info
        """
        if name in self._clients:
            logger.warning(f"Overwriting existing client registration: {name}")

        self._clients[name] = ClientInfo(
            name=name,
            description=description,
            client_class=client_class,
            supported_backends=supported_backends,
            default_prompts=default_prompts,
            model_capabilities=model_capabilities or {},
        )
        self._factories[name] = factory

        logger.debug(f"Registered OCR client: {name} -> {client_class.__name__}")

    def create_client(self, name: str, **kwargs) -> BaseOCRClient:
        """Create a client instance by name.

        Args:
            name: Client name (e.g., "qwen", "deepseek", "hunyuan")
            **kwargs: Client-specific initialization parameters

        Returns:
            Initialized client instance

        Raises:
            ValueError: If client name is not registered
        """
        if name not in self._factories:
            available = list(self._clients.keys())
            raise ValueError(f"Unknown client '{name}'. Available clients: {available}")

        factory = self._factories[name]
        try:
            logger.info(f"Creating OCR client: {name} with kwargs: {kwargs}")
            client = factory(**kwargs)
            logger.info(f"Created OCR client: {name} -> {type(client).__name__}")
            return client
        except Exception as e:
            logger.exception(f"Failed to create client '{name}': {e}")
            raise

    def list_clients(self) -> List[str]:
        """Get list of available client names."""
        return list(self._clients.keys())

    def get_client_info(self, name: str) -> ClientInfo:
        """Get information about a registered client."""
        if name not in self._clients:
            raise ValueError(f"Unknown client '{name}'")
        return self._clients[name]

    def get_all_client_info(self) -> Dict[str, ClientInfo]:
        """Get information about all registered clients."""
        return self._clients.copy()


# Global registry instance
_registry = OCRClientRegistry()


# Factory functions for each client type
def _create_client_with_config(client_type: str, client_class_name: str, module_name: str, **kwargs) -> BaseOCRClient:
    """Universal factory function with auto-configuration."""
    try:
        from .config_loader import get_client_config

        # Dynamic import of the client class from local_vl_utils package
        full_module_name = f"local_vl_utils{module_name}"
        module = importlib.import_module(full_module_name)
        client_class = getattr(module, client_class_name)

        # Load configuration from environment with parameter overrides
        config = get_client_config(client_type, **kwargs)

        # Convert config to client parameters
        client_kwargs = {}
        if config.model_name:
            client_kwargs["model_name"] = config.model_name
        if config.server_url:
            client_kwargs["server_url"] = config.server_url
        # Always set server_headers if we have an API key
        headers = config.get_headers()
        if headers:
            client_kwargs["server_headers"] = headers

        # Parameters that are backend-specific and should be filtered out
        # Note: 'backend' is NOT filtered - it's required by QwenClient
        backend_specific_params = {
            "model",        # Transformers backend only
            "processor",    # Transformers backend only
            "vllm_llm",     # vLLM engine backend only
            "vllm_async_llm",  # vLLM async backend only
            "batch_size",   # Local inference only
        }

        # Add any additional kwargs, but filter out backend-specific and None values
        for key, value in kwargs.items():
            if key in ("model_name", "server_url", "api_key"):
                # Skip these as they're already handled above
                continue
            elif key == "server_headers" and value is not None:
                # Only override server_headers if explicitly provided and not None
                client_kwargs[key] = value
            elif key not in backend_specific_params:
                # Add all other parameters that are not backend-specific
                client_kwargs[key] = value

        logger.debug(f"Creating {client_type} client with config: {client_kwargs}")
        return client_class(**client_kwargs)
    except ImportError as e:
        raise ImportError(f"Failed to import {client_class_name}: {e}") from e


def _create_qwen_client(**kwargs) -> BaseOCRClient:
    """Factory function for Qwen client with auto-configuration."""
    return _create_client_with_config("qwen", "QwenClient", ".qwen_client", **kwargs)


def _create_deepseek_client(**kwargs) -> BaseOCRClient:
    """Factory function for DeepSeek client with auto-configuration."""
    return _create_client_with_config("deepseek", "DeepSeekClient", ".deepseek_client", **kwargs)


def _create_hunyuan_client(**kwargs) -> BaseOCRClient:
    """Factory function for Hunyuan client with auto-configuration."""
    return _create_client_with_config("hunyuan", "HunyuanClient", ".hunyuan_client", **kwargs)


def _create_mineru_client(**kwargs) -> BaseOCRClient:
    """Factory function for MinerU client with auto-configuration."""
    return _create_client_with_config("mineru", "MinerUClient", ".mineru_client", **kwargs)


def _register_builtin_clients() -> None:
    """Register all built-in OCR clients."""

    # Register Qwen client
    try:
        from .qwen_client import QwenClient, DEFAULT_PROMPTS, DEFAULT_SAMPLING_PARAMS

        _registry.register_client(
            name="qwen",
            description="Qwen-VL OCR client with multi-mode support (Layout/Text/Markdown)",
            client_class=QwenClient,
            factory=_create_qwen_client,
            supported_backends=["http-client", "transformers", "vllm-engine", "vllm-async-engine"],
            default_prompts=DEFAULT_PROMPTS,
            model_capabilities={
                "layout_detection": True,
                "text_recognition": True,
                "table_extraction": True,
                "equation_extraction": True,
                "multi_mode": True,
                "coordinate_formats": ["absolute", "relative"],
                "output_formats": ["json", "markdown"],
                "rotation_support": True,
            },
        )
        logger.debug("Registered Qwen client")
    except ImportError as e:
        logger.warning(f"Failed to register Qwen client: {e}")

    # Register DeepSeek client
    try:
        from .deepseek_client import DeepSeekClient, DEFAULT_DEEPSEEK_PROMPTS, DEFAULT_DEEPSEEK_SAMPLING_PARAMS

        _registry.register_client(
            name="deepseek",
            description="DeepSeek-OCR client with JSON output format",
            client_class=DeepSeekClient,
            factory=_create_deepseek_client,
            supported_backends=["http-client", "transformers", "vllm-engine", "vllm-async-engine"],
            default_prompts=DEFAULT_DEEPSEEK_PROMPTS,
            model_capabilities={
                "layout_detection": True,
                "text_recognition": True,
                "table_extraction": True,
                "equation_extraction": True,
                "multi_mode": False,
                "coordinate_formats": ["relative"],
                "output_formats": ["json"],
                "rotation_support": True,
            },
        )
        logger.debug("Registered DeepSeek client")
    except ImportError as e:
        logger.warning(f"Failed to register DeepSeek client: {e}")

    # Register Hunyuan client
    try:
        from .hunyuan_client import HunyuanClient, DEFAULT_HUNYUAN_PROMPTS, DEFAULT_HUNYUAN_SAMPLING_PARAMS

        _registry.register_client(
            name="hunyuan",
            description="Hunyuan-OCR client with XML output format and HTML table support",
            client_class=HunyuanClient,
            factory=_create_hunyuan_client,
            supported_backends=["http-client", "transformers", "vllm-engine", "vllm-async-engine"],
            default_prompts=DEFAULT_HUNYUAN_PROMPTS,
            model_capabilities={
                "layout_detection": True,
                "text_recognition": True,
                "table_extraction": True,
                "equation_extraction": True,
                "multi_mode": False,
                "coordinate_formats": ["absolute"],
                "output_formats": ["xml", "html"],
                "rotation_support": False,
                "html_tables": True,
            },
        )
        logger.debug("Registered Hunyuan client")
    except ImportError as e:
        logger.warning(f"Failed to register Hunyuan client: {e}")

    # Register MinerU client
    try:
        from .mineru_client import MinerUClient, MINERU_DEFAULT_PROMPTS, DEFAULT_MINERU_SAMPLING_PARAMS

        _registry.register_client(
            name="mineru",
            description="MinerU VLM client using legacy prompt set",
            client_class=MinerUClient,
            factory=_create_mineru_client,
            supported_backends=["http-client", "transformers", "vllm-engine", "vllm-async-engine"],
            default_prompts=MINERU_DEFAULT_PROMPTS,
            model_capabilities={
                "layout_detection": True,
                "text_recognition": True,
                "table_extraction": True,
                "equation_extraction": True,
                "multi_mode": False,
                "coordinate_formats": ["relative"],
                "output_formats": ["json"],
                "rotation_support": True,
            },
        )
        logger.debug("Registered MinerU client")
    except ImportError as e:
        logger.warning(f"Failed to register MinerU client: {e}")


# Public API functions
def create_ocr_client(client_type: str, **kwargs) -> BaseOCRClient:
    """Create an OCR client by type name.

    Args:
        client_type: Type of client to create ("qwen", "deepseek", "hunyuan")
        **kwargs: Client-specific initialization parameters

    Returns:
        Initialized OCR client instance

    Example:
        client = create_ocr_client(
            "qwen",
            backend="http-client",
            server_url="http://localhost:8000",
            model_name="Qwen2-VL-72B-Instruct"
        )
    """
    return _registry.create_client(client_type, **kwargs)


def list_available_clients() -> List[str]:
    """Get list of available OCR client names.

    Returns:
        List of client names (e.g., ["qwen", "deepseek", "hunyuan"])
    """
    return _registry.list_clients()


def get_client_info(client_type: str) -> ClientInfo:
    """Get detailed information about a client.

    Args:
        client_type: Client name

    Returns:
        ClientInfo object with details about the client
    """
    return _registry.get_client_info(client_type)


def get_all_client_info() -> Dict[str, ClientInfo]:
    """Get information about all available clients.

    Returns:
        Dictionary mapping client names to ClientInfo objects
    """
    return _registry.get_all_client_info()


def register_custom_client(
    name: str,
    description: str,
    client_class: Type[BaseOCRClient],
    factory: Callable[..., BaseOCRClient],
    supported_backends: List[str],
    default_prompts: Dict[str, str],
    model_capabilities: Optional[Dict[str, Any]] = None,
) -> None:
    """Register a custom OCR client.

    This allows users to add their own client implementations.

    Args:
        name: Unique name for the client
        description: Human-readable description
        client_class: The client class (must inherit from BaseOCRClient)
        factory: Factory function that creates client instances
        supported_backends: List of supported backend types
        default_prompts: Default prompts for different task types
        model_capabilities: Model-specific capabilities info
    """
    _registry.register_client(
        name, description, client_class, factory, supported_backends, default_prompts, model_capabilities
    )


# Auto-register built-in clients on module import
_register_builtin_clients()


__all__ = [
    # Main factory function
    "create_ocr_client",

    # Info functions
    "list_available_clients",
    "get_client_info",
    "get_all_client_info",

    # Registration
    "register_custom_client",

    # Classes
    "ClientInfo",
    "OCRClientRegistry",
]
