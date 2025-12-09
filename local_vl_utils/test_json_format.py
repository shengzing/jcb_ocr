"""Test script for unified JSON output functionality.

Run this to verify that the JSON conversion works correctly.
"""

import json
from vlm_client.local_vl_utils.structs import ContentBlock
from vlm_client.local_vl_utils.format_converter import (
    blocks_to_json,
    blocks_to_dict_list,
    blocks_to_standard_json,
    parse_json_output,
    ensure_json_output,
)


def test_basic_conversion():
    """Test basic ContentBlock to JSON conversion"""
    print("=" * 60)
    print("Test 1: Basic Conversion")
    print("=" * 60)

    # Create sample blocks
    blocks = [
        ContentBlock("text", [0.1, 0.1, 0.9, 0.3], content="Sample text"),
        ContentBlock("title", [0.1, 0.05, 0.9, 0.1], angle=0),
        ContentBlock("table", [0.1, 0.4, 0.9, 0.8], content="| A | B |"),
    ]

    # Convert to JSON
    json_str = blocks_to_json(blocks, pretty=True)
    print("\nJSON Output:")
    print(json_str)

    # Verify it's valid JSON
    data = json.loads(json_str)
    assert len(data) == 3
    assert data[0]["type"] == "text"
    assert data[0]["content"] == "Sample text"
    print("\n✓ Basic conversion passed!")


def test_standard_format():
    """Test standard format with metadata"""
    print("\n" + "=" * 60)
    print("Test 2: Standard Format with Metadata")
    print("=" * 60)

    blocks = [
        ContentBlock("text", [0.0, 0.0, 1.0, 0.5]),
        ContentBlock("image", [0.0, 0.5, 1.0, 1.0]),
    ]

    result = blocks_to_standard_json(
        blocks,
        model_type="qwen",
        include_metadata=True
    )

    print("\nStandard Format:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Verify structure
    assert "blocks" in result
    assert "metadata" in result
    assert result["metadata"]["model_type"] == "qwen"
    assert result["metadata"]["block_count"] == 2
    print("\n✓ Standard format passed!")


def test_parse_json_back():
    """Test parsing JSON back to ContentBlock"""
    print("\n" + "=" * 60)
    print("Test 3: Parse JSON Back to ContentBlock")
    print("=" * 60)

    json_str = """
    [
        {
            "type": "text",
            "bbox": [0.1, 0.2, 0.8, 0.5],
            "angle": null,
            "content": "Test content"
        },
        {
            "type": "title",
            "bbox": [0.0, 0.0, 1.0, 0.1],
            "angle": 0,
            "content": null
        }
    ]
    """

    blocks = parse_json_output(json_str)

    print(f"\nParsed {len(blocks)} blocks:")
    for i, block in enumerate(blocks):
        print(f"  Block {i+1}: type={block.type}, bbox={block.bbox}, content={block.content}")

    # Verify
    assert len(blocks) == 2
    assert blocks[0].type == "text"
    assert blocks[0].content == "Test content"
    assert blocks[1].type == "title"
    assert blocks[1].angle == 0
    print("\n✓ Parse JSON back passed!")


def test_round_trip():
    """Test round-trip: ContentBlock -> JSON -> ContentBlock"""
    print("\n" + "=" * 60)
    print("Test 4: Round-trip Conversion")
    print("=" * 60)

    # Original blocks
    original_blocks = [
        ContentBlock("text", [0.1, 0.2, 0.9, 0.8], angle=90, content="Rotated text"),
        ContentBlock("seal", [0.8, 0.8, 1.0, 1.0]),
    ]

    # Convert to JSON
    json_str = blocks_to_json(original_blocks)
    print("\nOriginal blocks -> JSON:")
    print(json_str)

    # Parse back
    parsed_blocks = parse_json_output(json_str)
    print(f"\nJSON -> Parsed {len(parsed_blocks)} blocks")

    # Verify they match
    for orig, parsed in zip(original_blocks, parsed_blocks):
        assert orig.type == parsed.type
        assert orig.bbox == parsed.bbox
        assert orig.angle == parsed.angle
        assert orig.content == parsed.content
        print(f"  ✓ Block {orig.type} matches")

    print("\n✓ Round-trip conversion passed!")


def test_ensure_json_output():
    """Test automatic JSON conversion"""
    print("\n" + "=" * 60)
    print("Test 5: Ensure JSON Output")
    print("=" * 60)

    # Test 1: Already JSON
    json_input = '[{"type": "text", "bbox": [0.0, 0.0, 1.0, 1.0], "angle": null, "content": null}]'
    result = ensure_json_output(json_input)
    assert json.loads(result)  # Verify it's valid JSON
    print("✓ Already JSON: passed")

    # Test 2: ContentBlock list
    blocks = [ContentBlock("text", [0.0, 0.0, 1.0, 1.0])]
    result = ensure_json_output(blocks)
    assert json.loads(result)  # Verify it's valid JSON
    print("✓ ContentBlock list: passed")

    # Test 3: Dict
    dict_input = {"blocks": [{"type": "text", "bbox": [0.0, 0.0, 1.0, 1.0]}]}
    result = ensure_json_output(dict_input)
    assert json.loads(result)  # Verify it's valid JSON
    print("✓ Dict input: passed")

    print("\n✓ Ensure JSON output passed!")


def test_dict_list_conversion():
    """Test blocks_to_dict_list"""
    print("\n" + "=" * 60)
    print("Test 6: Dict List Conversion")
    print("=" * 60)

    blocks = [
        ContentBlock("text", [0.1, 0.2, 0.9, 0.8], content="Text 1"),
        ContentBlock("text", [0.1, 0.8, 0.9, 0.95], content="Text 2", image_type="blurry"),
    ]

    dict_list = blocks_to_dict_list(blocks)

    print("\nDict List:")
    print(json.dumps(dict_list, ensure_ascii=False, indent=2))

    # Verify
    assert len(dict_list) == 2
    assert dict_list[0]["content"] == "Text 1"
    assert dict_list[1]["image_type"] == "blurry"
    assert "image_type" not in dict_list[0]
    print("\n✓ Dict list conversion passed!")


def test_all_block_types():
    """Test all supported block types"""
    print("\n" + "=" * 60)
    print("Test 7: All Block Types")
    print("=" * 60)

    from vlm_client.local_vl_utils.structs import BLOCK_TYPES

    blocks = []
    y = 0.0
    for block_type in sorted(BLOCK_TYPES):
        if y >= 1.0:
            break
        blocks.append(ContentBlock(block_type, [0.0, y, 1.0, min(y + 0.1, 1.0)]))
        y += 0.1

    json_str = blocks_to_json(blocks, pretty=False)
    parsed_blocks = parse_json_output(json_str)

    print(f"\nTested {len(blocks)} block types")
    assert len(parsed_blocks) == len(blocks)
    print("✓ All block types conversion passed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print(" " * 15 + "JSON FORMAT CONVERTER TESTS")
    print("=" * 70 + "\n")

    tests = [
        test_basic_conversion,
        test_standard_format,
        test_parse_json_back,
        test_round_trip,
        test_ensure_json_output,
        test_dict_list_conversion,
        test_all_block_types,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
