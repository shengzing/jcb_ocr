"""Example: Unified JSON Output for Different OCR Models

This script demonstrates how to get consistent JSON output from
different OCR models (Qwen, DeepSeek, Hunyuan, MinerU).
"""

from PIL import Image
from vlm_client.local_vl_utils.qwen_client import QwenClient
from vlm_client.local_vl_utils.deepseek_client import DeepSeekClient
from vlm_client.local_vl_utils.hunyuan_client import HunyuanClient
from vlm_client.local_vl_utils.mineru_client import MinerUClient
from vlm_client.local_vl_utils.format_converter import (
    blocks_to_json,
    blocks_to_standard_json,
    parse_json_output,
    ensure_json_output
)
import json


def example_1_basic_json_output():
    """Example 1: Get JSON output from layout detection"""
    print("=" * 60)
    print("Example 1: Basic JSON Output")
    print("=" * 60)

    # Initialize client (using Qwen as example)
    client = QwenClient(
        backend="http-client",
        server_url="http://localhost:8000/v1/chat/completions",
        model_name="Qwen2-VL-7B-Instruct"
    )

    # Load image
    image = Image.open("test_image.png")

    # Method 1: Use layout_detect_as_json (convenience method)
    json_result = client.layout_detect_as_json(image)
    print("\nMethod 1 - Using layout_detect_as_json():")
    print(json_result)

    # Method 2: Manual conversion
    blocks = client.layout_detect(image)
    json_result = blocks_to_json(blocks, pretty=True)
    print("\nMethod 2 - Manual conversion:")
    print(json_result)


def example_2_standard_format_with_metadata():
    """Example 2: Get standardized JSON with metadata"""
    print("\n" + "=" * 60)
    print("Example 2: Standardized JSON with Metadata")
    print("=" * 60)

    client = DeepSeekClient(
        backend="http-client",
        server_url="http://localhost:8001/v1/chat/completions",
        model_name="DeepSeek-VL2"
    )

    image = Image.open("test_image.png")
    blocks = client.layout_detect(image)

    # Get standardized format with metadata
    result = blocks_to_standard_json(
        blocks,
        model_type="deepseek",
        include_metadata=True
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Access metadata
    print(f"\nModel Type: {result['metadata']['model_type']}")
    print(f"Block Count: {result['metadata']['block_count']}")


def example_3_compare_multiple_models():
    """Example 3: Compare outputs from different models in unified format"""
    print("\n" + "=" * 60)
    print("Example 3: Compare Multiple Models")
    print("=" * 60)

    # Initialize multiple clients
    clients = {
        "qwen": QwenClient(
            backend="http-client",
            server_url="http://localhost:8000/v1/chat/completions",
        ),
        "deepseek": DeepSeekClient(
            backend="http-client",
            server_url="http://localhost:8001/v1/chat/completions",
        ),
        "hunyuan": HunyuanClient(
            backend="http-client",
            server_url="http://localhost:8002/v1/chat/completions",
        ),
        "mineru": MinerUClient(
            backend="http-client",
            server_url="http://localhost:8003/v1/chat/completions",
        ),
    }

    image = Image.open("test_image.png")

    # Get results from all models in unified JSON format
    results = {}
    for name, client in clients.items():
        blocks = client.layout_detect(image)
        results[name] = blocks_to_standard_json(blocks, name, include_metadata=True)

    # Compare results
    print("\nComparison of block counts:")
    for name, result in results.items():
        print(f"  {name}: {result['metadata']['block_count']} blocks")

    # Save all results to file
    with open("comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nResults saved to comparison_results.json")


def example_4_mixed_layout_and_extract():
    """Example 4: Use one model for layout, another for extraction"""
    print("\n" + "=" * 60)
    print("Example 4: Mixed Models (Layout + Extraction)")
    print("=" * 60)

    # Use Qwen for layout detection
    qwen_client = QwenClient(
        backend="http-client",
        server_url="http://localhost:8000/v1/chat/completions",
    )

    # Use DeepSeek for content extraction
    deepseek_client = DeepSeekClient(
        backend="http-client",
        server_url="http://localhost:8001/v1/chat/completions",
    )

    image = Image.open("test_image.png")

    # Step 1: Layout detection with Qwen
    print("\nStep 1: Layout detection with Qwen...")
    blocks = qwen_client.layout_detect(image)
    layout_json = blocks_to_standard_json(blocks, "qwen", include_metadata=True)
    print(f"Detected {len(blocks)} blocks")

    # Step 2: Content extraction with DeepSeek
    print("\nStep 2: Content extraction with DeepSeek...")
    block_images, prompts, params, indices = deepseek_client.helper.prepare_for_extract(image, blocks)
    outputs = deepseek_client.client.batch_predict(block_images, prompts, params)

    # Fill content
    for idx, output in zip(indices, outputs):
        blocks[idx].content = deepseek_client._extract_text(output)

    # Post-process
    blocks = deepseek_client.helper.post_process(blocks)

    # Get final result in JSON
    final_json = blocks_to_standard_json(
        blocks,
        model_type="mixed_qwen_layout_deepseek_extract",
        include_metadata=True
    )

    print(json.dumps(final_json, ensure_ascii=False, indent=2))


def example_5_parse_json_back():
    """Example 5: Parse JSON back to ContentBlock objects"""
    print("\n" + "=" * 60)
    print("Example 5: Parse JSON Back to ContentBlock")
    print("=" * 60)

    # Sample JSON data
    json_data = """
    [
        {
            "type": "text",
            "bbox": [0.1, 0.1, 0.9, 0.3],
            "angle": null,
            "content": "Sample text content"
        },
        {
            "type": "table",
            "bbox": [0.1, 0.4, 0.9, 0.8],
            "angle": 0,
            "content": "| Header | Data |\\n|--------|------|"
        }
    ]
    """

    # Parse JSON back to ContentBlock objects
    blocks = parse_json_output(json_data)

    print(f"\nParsed {len(blocks)} blocks:")
    for i, block in enumerate(blocks):
        print(f"  Block {i+1}: type={block.type}, bbox={block.bbox}")

    # Can now use these blocks normally
    json_output = blocks_to_json(blocks, pretty=True)
    print("\nRe-serialized JSON:")
    print(json_output)


def example_6_ensure_json_output():
    """Example 6: Automatically ensure JSON output from any format"""
    print("\n" + "=" * 60)
    print("Example 6: Auto-convert to JSON")
    print("=" * 60)

    # Test with different input types
    test_inputs = [
        # Already JSON
        '[{"type": "text", "bbox": [0.0, 0.0, 1.0, 1.0], "angle": null, "content": null}]',

        # ContentBlock list
        # blocks,  # from previous example

        # Legacy format (requires model_type)
        # "<|box_start|>100 200 300 400<|box_end|><|ref_start|>text<|ref_end|>",
    ]

    for i, input_data in enumerate(test_inputs):
        print(f"\nTest {i+1}:")
        try:
            result = ensure_json_output(input_data)
            print(f"  Success: {result[:100]}...")
        except ValueError as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    # Run examples
    # Note: You'll need to adjust server URLs and image paths for your setup

    print("Unified JSON Output Examples")
    print("=" * 60)
    print("\nThese examples demonstrate how to get consistent JSON output")
    print("from different OCR models.\n")

    # Uncomment the examples you want to run:

    # example_1_basic_json_output()
    # example_2_standard_format_with_metadata()
    # example_3_compare_multiple_models()
    # example_4_mixed_layout_and_extract()
    # example_5_parse_json_back()
    # example_6_ensure_json_output()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
