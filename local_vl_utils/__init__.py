import importlib
from typing import TYPE_CHECKING

from .version import __version__


__lazy_attrs__ = {
    # Legacy Qwen client
    "QwenClient": (".qwen_client", "QwenClient"),
    "QwenSamplingParams": (".qwen_client", "QwenSamplingParams"),
    "QwenLogitsProcessor": (".logits_processor.vllm_v1_no_repeat_ngram", "VllmV1NoRepeatNGramLogitsProcessor"),

    # Multi-client support
    "create_ocr_client": (".client_factory", "create_ocr_client"),
    "list_available_clients": (".client_factory", "list_available_clients"),
    "get_client_info": (".client_factory", "get_client_info"),
    "get_all_client_info": (".client_factory", "get_all_client_info"),

    # Configuration management
    "get_client_config": (".config_loader", "get_client_config"),
    "load_env_file": (".config_loader", "load_env_file"),
    "get_default_client_type": (".config_loader", "get_default_client_type"),
    "ClientConfig": (".config_loader", "ClientConfig"),
}

if TYPE_CHECKING:
    # Rename for future compatibility
    from .logits_processor.vllm_v1_no_repeat_ngram import (
        VllmV1NoRepeatNGramLogitsProcessor as QwenLogitsProcessor,
    )
    from .qwen_client import QwenClient, QwenSamplingParams
    from .client_factory import (
        create_ocr_client,
        list_available_clients,
        get_client_info,
        get_all_client_info,
    )
    from .config_loader import (
        get_client_config,
        load_env_file,
        get_default_client_type,
        ClientConfig,
    )


def __getattr__(name: str):
    if name in __lazy_attrs__:
        module_name, attr_name = __lazy_attrs__[name]
        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Legacy Qwen
    "QwenClient",
    "QwenSamplingParams",
    "QwenLogitsProcessor",

    # Multi-client factory
    "create_ocr_client",
    "list_available_clients",
    "get_client_info",
    "get_all_client_info",

    # Configuration management
    "get_client_config",
    "load_env_file",
    "get_default_client_type",
    "ClientConfig",

    # Version
    "__version__",
]
