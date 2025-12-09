"""Unified JSON format converter for different OCR model outputs.

This module provides conversion functions to ensure all models output
consistent JSON format, regardless of their internal representation.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from .structs import ContentBlock


def blocks_to_json(blocks: list[ContentBlock], pretty: bool = False) -> str:
    """Convert ContentBlock list to JSON string.

    Args:
        blocks: List of ContentBlock objects
        pretty: If True, format JSON with indentation

    Returns:
        JSON string representation
    """
    # ContentBlock already inherits from dict, so can be directly serialized
    json_data = [dict(block) for block in blocks]

    if pretty:
        return json.dumps(json_data, ensure_ascii=False, indent=2)
    return json.dumps(json_data, ensure_ascii=False)


def blocks_to_dict_list(blocks: list[ContentBlock]) -> list[dict]:
    """Convert ContentBlock list to list of dictionaries.

    Args:
        blocks: List of ContentBlock objects

    Returns:
        List of dictionaries with standardized format
    """
    return [
        {
            "type": block.type,
            "bbox": block.bbox,
            "angle": block.angle,
            "content": block.content,
            **({"image_type": block.image_type} if block.image_type else {})
        }
        for block in blocks
    ]


def normalize_layout_output(raw_output: str, model_type: str) -> dict:
    """Normalize raw model output to standard JSON format.

    This function wraps the raw output and metadata into a consistent structure
    for easier consumption by downstream systems.

    Args:
        raw_output: Raw string output from the model
        model_type: Type of model ("qwen", "deepseek", "hunyuan", "mineru")

    Returns:
        Dictionary with structure:
        {
            "model_type": str,
            "raw_output": str,
            "format": str (json/xml/legacy)
        }
    """
    # Detect output format
    format_type = "unknown"
    if raw_output.strip().startswith(("{", "[")):
        format_type = "json"
    elif "<|box_start|>" in raw_output:
        format_type = "legacy_box"
    elif "<|ref|>" in raw_output and "<|det|>" in raw_output:
        format_type = "grounding"
    elif "<ref>" in raw_output and "<quad>" in raw_output:
        format_type = "xml_quad"

    return {
        "model_type": model_type,
        "format": format_type,
        "raw_output": raw_output
    }


def blocks_to_standard_json(
    blocks: list[ContentBlock],
    model_type: str,
    raw_output: str | None = None,
    include_metadata: bool = True
) -> dict:
    """Convert blocks to standardized JSON output with optional metadata.

    Args:
        blocks: List of ContentBlock objects
        model_type: Type of model ("qwen", "deepseek", "hunyuan", "mineru")
        raw_output: Optional raw model output for reference
        include_metadata: Whether to include metadata in output

    Returns:
        Standardized dictionary with structure:
        {
            "blocks": [...],
            "metadata": {
                "model_type": str,
                "block_count": int,
                "raw_output": str (optional)
            }
        }
    """
    result = {
        "blocks": blocks_to_dict_list(blocks)
    }

    if include_metadata:
        metadata = {
            "model_type": model_type,
            "block_count": len(blocks)
        }
        if raw_output is not None:
            metadata["raw_output"] = raw_output
        result["metadata"] = metadata

    return result


def parse_json_output(json_str: str) -> list[ContentBlock]:
    """Parse JSON string back to ContentBlock list.

    Args:
        json_str: JSON string representation

    Returns:
        List of ContentBlock objects

    Raises:
        ValueError: If JSON is invalid or doesn't match expected format
    """
    try:
        data = json.loads(json_str)

        # Handle both raw list and wrapped format
        if isinstance(data, dict) and "blocks" in data:
            blocks_data = data["blocks"]
        elif isinstance(data, list):
            blocks_data = data
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")

        blocks = []
        for block_data in blocks_data:
            if not isinstance(block_data, dict):
                logger.warning(f"Skipping non-dict block: {block_data}")
                continue

            try:
                block = ContentBlock(
                    type=block_data["type"],
                    bbox=block_data["bbox"],
                    angle=block_data.get("angle"),
                    content=block_data.get("content"),
                    image_type=block_data.get("image_type")
                )
                blocks.append(block)
            except (KeyError, AssertionError) as e:
                logger.warning(f"Invalid block data: {block_data}, error: {e}")
                continue

        return blocks

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")


def convert_legacy_to_json(legacy_output: str, model_type: str) -> str:
    """Convert legacy format output to JSON string.

    This is a convenience function that combines parsing and conversion.

    Args:
        legacy_output: Legacy format string (box_start/ref_start or XML)
        model_type: Type of model to parse with

    Returns:
        JSON string representation

    Note:
        This requires importing the specific client to parse,
        so it's recommended to use parse_layout_output directly
        from the helper instead.
    """
    # Import here to avoid circular dependency
    if model_type == "qwen":
        from .qwen_client import QwenClientHelper
        helper = QwenClientHelper(
            backend="http-client",
            prompts={},
            sampling_params={},
            layout_image_size=(1036, 1036),
            min_image_edge=28,
            max_image_edge_ratio=50,
            handle_equation_block=True,
            abandon_list=False,
            abandon_paratext=False,
            debug=False
        )
    elif model_type == "deepseek":
        from .deepseek_client import DeepSeekClientHelper
        helper = DeepSeekClientHelper(
            backend="http-client",
            prompts={},
            sampling_params={},
            layout_image_size=(1036, 1036),
            min_image_edge=28,
            max_image_edge_ratio=50,
            handle_equation_block=True,
            abandon_list=False,
            abandon_paratext=False,
            debug=False
        )
    elif model_type == "hunyuan":
        from .hunyuan_client import HunyuanClientHelper
        helper = HunyuanClientHelper(
            backend="http-client",
            prompts={},
            sampling_params={},
            layout_image_size=(1036, 1036),
            min_image_edge=28,
            max_image_edge_ratio=50,
            handle_equation_block=True,
            abandon_list=False,
            abandon_paratext=False,
            debug=False
        )
    elif model_type == "mineru":
        from .mineru_client import MinerUClientHelper
        helper = MinerUClientHelper(
            backend="http-client",
            prompts={},
            sampling_params={},
            layout_image_size=(1036, 1036),
            min_image_edge=28,
            max_image_edge_ratio=50,
            handle_equation_block=True,
            abandon_list=False,
            abandon_paratext=False,
            debug=False
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    blocks = helper.parse_layout_output(legacy_output)
    return blocks_to_json(blocks)


# Convenience functions for specific use cases
def ensure_json_output(output: Any, model_type: str | None = None) -> str:
    """Ensure output is in JSON format, converting if necessary.

    Args:
        output: Can be list[ContentBlock], dict, or string
        model_type: Required if output is a string to parse

    Returns:
        JSON string

    Raises:
        ValueError: If conversion is not possible
    """
    if isinstance(output, str):
        # Try to parse as JSON first
        try:
            json.loads(output)
            return output  # Already valid JSON
        except json.JSONDecodeError:
            # Not JSON, need to convert
            if model_type is None:
                raise ValueError("model_type required to convert non-JSON string")
            return convert_legacy_to_json(output, model_type)

    elif isinstance(output, list):
        # Assume it's list of ContentBlock
        return blocks_to_json(output)

    elif isinstance(output, dict):
        # Already a dict, just serialize
        return json.dumps(output, ensure_ascii=False)

    else:
        raise ValueError(f"Cannot convert type {type(output)} to JSON")
