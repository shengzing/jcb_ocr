"""DeepSeek-OCR client implementation.

DeepSeek-OCR output format (similar to Qwen3-VL):
- Layout Detection: JSON array with bbox and label
- Text Recognition: Nested list with type/bbox/content

Key differences from Qwen:
- Prompt templates optimized for DeepSeek
- Different sampling parameters
- Slightly different bbox format conventions
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from concurrent.futures import Executor
from typing import Any, Literal, Sequence

from loguru import logger
from PIL import Image

from .base_client import BaseOCRClient, BaseOCRClientHelper
from .post_process import post_process
from .prompt_library import DEEPSEEK_DEFAULT_PROMPTS
from .structs import BLOCK_TYPES, ContentBlock
from .vlm_client import DEFAULT_SYSTEM_PROMPT, SamplingParams, new_vlm_client
from .vlm_client.utils import gather_tasks, get_png_bytes, get_rgb_image

# DeepSeek-specific constants
_layout_re = r"^\\s*\\{\\s*\\\"bbox(_2d)?\\\"\\s*:\\s*\\[\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*,\\s*(-?\\d+)\\s*\\]\\s*,\\s*\\\"(label|type)\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"(?:,\\s*[^}]*)?\\s*\\}\\s*,?\\s*(.*)$"

# DeepSeek official grounding format: <|ref|>content<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
# The regex previously double-escaped bracket literals which produced an invalid pattern.
_ref_det_re = (
    r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[([0-9]+),\s*([0-9]+),\s*([0-9]+),\s*([0-9]+)\]\]<\|/det\|>"
)

_TYPE_ALIASES = {
    "stamp": "seal",
    "seal_stamp": "seal",
    "stamp_seal": "seal",
    "handwriting": "handwritten",
    "hand_written": "handwritten",
}



def _parse_primary_json_segment(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for open_char, close_char in (("[", "]"), ("{", "}")):
        start = stripped.find(open_char)
        end = stripped.rfind(close_char)
        if start == -1 or end == -1 or end <= start:
            continue
        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _extract_text_from_structured_output(output: str) -> str:
    stripped = output.strip()
    if not stripped:
        return ""

    parsed = _parse_primary_json_segment(stripped)
    contents: list[str] = []

    def _collect(item: Any) -> None:
        if isinstance(item, dict):
            content = item.get("content")
            if content is None:
                return
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            contents.append(content)
        elif isinstance(item, list):
            for sub in item:
                _collect(sub)

    if parsed is not None:
        _collect(parsed)
        if contents:
            return "\n".join(contents)
        return ""

    return output

ANGLE_MAPPING: dict[str, Literal[0, 90, 180, 270]] = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}


class DeepSeekSamplingParams(SamplingParams):
    def __init__(
        self,
        temperature: float | None = 0.0,
        top_p: float | None = 0.01,
        top_k: int | None = 1,
        presence_penalty: float | None = 0.0,
        frequency_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        no_repeat_ngram_size: int | None = 100,
        max_new_tokens: int | None = None,
    ):
        super().__init__(
            temperature,
            top_p,
            top_k,
            presence_penalty,
            frequency_penalty,
            repetition_penalty,
            no_repeat_ngram_size,
            max_new_tokens,
        )


# DeepSeek-specific prompts (optimized for official format)
# Reference: https://huggingface.co/deepseek-ai/DeepSeek-OCR
# DEFAULT_DEEPSEEK_PROMPTS: dict[str, str] = {
#     "table": (
#         "<image>\\n"
#         "<|grounding|>Convert this table to clean Markdown format, preserving header rows, merged cells "
#         "(repeat text when necessary), numeric precision, and alignment."
#         + _structured_prompt_suffix("table", "the Markdown table string")
#     ),
#     "equation": (
#         "<image>\\n"
#         "<|grounding|>Transcribe the mathematical expression exactly in LaTeX. Keep fractions, superscripts, "
#         "subscripts, Greek letters and special symbols. Do not paraphrase or simplify."
#         + _structured_prompt_suffix("equation", "the LaTeX expression")
#     ),
#     "title": (
#         "<image>\\n"
#         "<|grounding|>Transcribe this heading exactly, keeping numbering, punctuation, emoji and capitalization."
#         + _structured_prompt_suffix("title", "the heading text")
#     ),
#     "list": (
#         "<image>\\n"
#         "<|grounding|>Extract the list in plain text. Preserve original bullet/number markers and nesting by "
#         "indenting with two spaces per level. Keep checkbox or dash prefixes."
#         + _structured_prompt_suffix("list", "the full list text with markers")
#     ),
#     "image": (
#         "<image>\\n"
#         "<|grounding|>Describe the image content clearly and concisely. Include key visual elements, objects, people, "
#         "text visible in the image, charts, diagrams, or any important details. Keep the description factual."
#         + _structured_prompt_suffix("image", "the image description")
#     ),
#     "handwritten": (
#         "<image>\\n"
#         "<|grounding|>This region contains handwritten text. Output your best transcription in plain text and "
#         "keep line breaks. If a character is unreadable, use [?] as placeholder."
#         + _structured_prompt_suffix("handwritten", "the handwritten transcription")
#     ),
#     "seal": (
#         "<image>\\n"
#         "<|grounding|>Read the characters on the seal/stamp exactly. Output only the deciphered text "
#         "(usually short phrases, institution names, or dates). No extra commentary."
#         + _structured_prompt_suffix("seal", "the deciphered seal text")
#     ),
#     "[default]": (
#         "<image>\\n"
#         "<|grounding|>Free OCR. Transcribe the selected region verbatim. Preserve punctuation, math symbols, "
#         "inline tables and line breaks. Keep question-and-answer cues such as 'Q:' and 'A:' exactly as printed."
#         + _structured_prompt_suffix(None, "the verbatim transcription")
#     ),
#     "[layout]": (
#         "<image>\\n"
#         "<|grounding|>Detect document layout and return structured output. Output format: "
#         "<|ref|>TYPE<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|> where TYPE is one of: text, title, list, table, "
#         "image, handwritten, seal. Coordinates should be normalized 0-999 range. Always use literal label "
#         "'seal' for stamps/seals. Keep items in reading order (top-to-bottom, left-to-right). "
#         "If rotation detected, append <|rotate_right|>, <|rotate_left|>, or <|rotate_down|> to the label."
#     ),
#     "[find]": (
#         "<image>\\n"
#         "<|grounding|>Locate and extract all visible elements with their precise coordinates. "
#         "Output format: <|ref|>content<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|> for each detected element. "
#         "Cover ALL text blocks, headings, lists, tables, images, handwritten content, and seals. "
#         "Use normalized coordinates (0-999 range)."
#     ),
# }

DEFAULT_DEEPSEEK_PROMPTS: dict[str, str] = DEEPSEEK_DEFAULT_PROMPTS

DEFAULT_DEEPSEEK_SAMPLING_PARAMS: dict[str, SamplingParams] = {
    "table": DeepSeekSamplingParams(presence_penalty=1.0, frequency_penalty=0.005),
    "equation": DeepSeekSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[find]": DeepSeekSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[default]": DeepSeekSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[layout]": DeepSeekSamplingParams(),
}


def _convert_bbox(bbox: Sequence[int] | Sequence[str]) -> list[float] | None:
    """Convert bounding box coordinates from integers (0-1000) to floats (0.0-1.0).

    Args:
        bbox: Sequence of 4 coordinates [x1, y1, x2, y2], range 0-1000

    Returns:
        Normalized bounding box [x1, y1, x2, y2], range 0.0-1.0, or None if invalid
    """
    bbox = tuple(map(int, bbox))
    if any(coord < 0 or coord > 1000 for coord in bbox):
        logger.debug(f"无效的边界框坐标(超出范围): {bbox}")
        return None
    x1, y1, x2, y2 = bbox
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    if x1 == x2 or y1 == y2:
        logger.debug(f"无效的边界框(宽度或高度为零): {bbox}")
        return None
    return list(map(lambda num: num / 1000.0, (x1, y1, x2, y2)))


def _parse_angle(tail: str) -> Literal[None, 0, 90, 180, 270]:
    """Parse rotation angle from layout detection output tail."""
    for token, angle in ANGLE_MAPPING.items():
        if token in tail:
            logger.debug(f"找到旋转标记 '{token}' -> 角度 {angle}")
            return angle
    logger.debug(f"未找到旋转标记: {tail}")
    return None


def _normalize_block_type(label: str | None) -> str:
    """Normalize model output type text to BLOCK_TYPES format."""
    if label is None:
        return ""
    text = str(label)
    for token in ANGLE_MAPPING:
        text = text.replace(token, "")
    text = text.strip().lower()
    text = re.sub(r"[\\s\\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    text = _TYPE_ALIASES.get(text, text)
    return text


def _find_angle_from_sources(*sources: object) -> Literal[None, 0, 90, 180, 270]:
    """Parse rotation angle from multiple possible fields."""
    for source in sources:
        if source is None:
            continue
        if isinstance(source, (int, float)):
            value = int(source)
            if value in ANGLE_MAPPING.values():
                logger.debug(f"直接使用数值旋转角度: {value}")
                return value  # type: ignore[return-value]
            continue
        text = str(source)
        angle = _parse_angle(text)
        if angle is not None:
            return angle
        try:
            value = int(text)
        except ValueError:
            continue
        if value in ANGLE_MAPPING.values():
            logger.debug(f"从文本中解析到数值旋转角度: {value}")
            return value  # type: ignore[return-value]
    return None


def _record_to_block(
    record: dict,
    source_desc: str,
    extra_angle_sources: tuple[object, ...] = (),
) -> ContentBlock | None:
    """Convert a generic record to a ContentBlock."""
    bbox_raw = record.get("bbox_2d") or record.get("bbox") or record.get("bbox2d")
    if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
        logger.warning(f"{source_desc}缺少合法bbox: {record}")
        return None

    bbox = _convert_bbox(bbox_raw)
    if bbox is None:
        logger.warning(f"{source_desc}边界框无效: {bbox_raw}")
        return None

    type_candidates = (
        record.get("label"),
        record.get("type"),
        record.get("category"),
        record.get("block_type"),
        record.get("layout_type"),
    )
    raw_label = next((candidate for candidate in type_candidates if candidate), None)
    ref_type = _normalize_block_type(raw_label)

    content_text = record.get("text_content") or record.get("content") or record.get("text")

    if (not ref_type or ref_type not in BLOCK_TYPES) and content_text:
        fallback_type = "text"
        logger.debug(f"{source_desc}: 使用默认类型{fallback_type!r}，原始类型: {raw_label}")
        ref_type = fallback_type

    if not ref_type or ref_type not in BLOCK_TYPES:
        logger.warning(f"{source_desc}块类型{raw_label!r}未知: {record}")
        return None

    angle = _find_angle_from_sources(
        record.get("angle"),
        record.get("rotation"),
        record.get("orientation"),
        raw_label,
        *extra_angle_sources,
    )
    if angle is None:
        logger.debug(f"{source_desc}: 未找到旋转角度")

    block = ContentBlock(ref_type, bbox, angle=angle, content=content_text)
    content_state = "有" if block.content else "无"
    logger.debug(
        f"{source_desc}: 类型={block.type}, 边界框={block.bbox}, 角度={block.angle}, 文本={content_state}"
    )
    return block


class DeepSeekClientHelper(BaseOCRClientHelper):
    """Helper class for DeepSeek-OCR model-specific logic."""

    # resize_by_need() and prepare_for_layout() are inherited from BaseOCRClientHelper

    def parse_layout_output(self, output: str) -> list[ContentBlock]:
        """Parse DeepSeek layout detection model output to ContentBlock objects.

        Supports multiple formats:
        1. JSON format: {"bbox_2d": [x1,y1,x2,y2], "label": "TYPE"}
        2. Grounding format: <|ref|>TYPE<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
        """
        logger.debug(f'DeepSeek解析内容：{output}')
        blocks: list[ContentBlock] = []
        stripped_output = output.strip()

        def _append_block_from_record(record: dict, origin: str, extra_sources: tuple[object, ...] = ()) -> None:
            block = _record_to_block(record, origin, extra_sources)
            if block is not None:
                blocks.append(block)

        # First, try to parse <|ref|><|det|> grounding format (official DeepSeek-OCR format)
        ref_det_matches = re.findall(_ref_det_re, stripped_output, re.DOTALL)
        if ref_det_matches:
            logger.info(f"检测到DeepSeek grounding格式输出，共{len(ref_det_matches)}个匹配")
            for match in ref_det_matches:
                content, x1, y1, x2, y2 = match
                # Infer block type from content
                block_type = _normalize_block_type(content) or "text"

                # Create record
                record = {
                    "bbox_2d": [int(x1), int(y1), int(x2), int(y2)],
                    "label": block_type,
                    "content": content.strip()
                }
                _append_block_from_record(record, f"grounding格式")

            if blocks:
                logger.info(f"从grounding格式解析出{len(blocks)}个内容块")
                return blocks

        # Try JSON parsing
        parsed_output = _parse_primary_json_segment(stripped_output)

        json_records: list | None = None
        if isinstance(parsed_output, dict):
            json_records = [parsed_output]
        elif isinstance(parsed_output, list):
            json_records = parsed_output

        if json_records is not None:
            total_records = len(json_records)
            for idx, record in enumerate(json_records, 1):
                if not isinstance(record, dict):
                    logger.warning(f"JSON块{idx}不是对象: {record}")
                    continue
                _append_block_from_record(record, f"JSON块{idx}")

            logger.info(f"从JSON输出解析出{len(blocks)}个内容块(共{total_records}条)")
            return blocks

        # Parse line by line
        lines = output.split("\\n")
        for line_num, raw_line in enumerate(lines, 1):
            line = raw_line.strip()
            if not line or line in {"[", "]", ","}:
                continue

            try:
                json_record = json.loads(line.rstrip(","))
            except json.JSONDecodeError:
                json_record = None

            if isinstance(json_record, dict):
                _append_block_from_record(json_record, f"第{line_num}行JSON")
                continue

            match = re.match(_layout_re, line)
            if match:
                _, x1, y1, x2, y2, _, raw_label, tail = match.groups()
                record = {"bbox_2d": [x1, y1, x2, y2], "label": raw_label}
                _append_block_from_record(record, f"第{line_num}行结构化", extra_sources=(tail,))
                continue

            logger.warning(f"第{line_num}行不匹配布局格式: {line}")

        logger.info(f"从{len(lines)}行中解析出{len(blocks)}个内容块")
        return blocks

    def prepare_for_extract(
        self,
        image: Image.Image,
        blocks: list[ContentBlock],
    ) -> tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]:
        """Prepare content blocks for text extraction by cropping and processing regions."""
        image = get_rgb_image(image)
        width, height = image.size
        block_images: list[Image.Image | bytes] = []
        prompts: list[str] = []
        sampling_params: list[SamplingParams | None] = []
        indices: list[int] = []

        skipped_types = []
        for idx, block in enumerate(blocks):
            # Skip only image blocks (not list, to ensure all 7 elements are extracted)
            # equation_block is also skipped as it's usually handled differently
            if block.type in ("image", "equation_block"):
                skipped_types.append(block.type)
                continue

            # Crop block region from image
            x1, y1, x2, y2 = block.bbox
            scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            block_image = image.crop(scaled_bbox)

            # Rotate if needed
            if block.angle in [90, 180, 270]:
                block_image = block_image.rotate(block.angle, expand=True)
                logger.debug(f"块{idx}: 旋转{block.angle}度")

            # Resize to meet constraints
            block_image = self.resize_by_need(block_image)

            # Convert to bytes for HTTP backend
            if self.backend == "http-client":
                block_image = get_png_bytes(block_image)

            block_images.append(block_image)

            # Get prompt and sampling params for block type
            prompt = self.prompts.get(block.type) or self.prompts["[default]"]
            prompts.append(prompt)
            params = self.sampling_params.get(block.type) or self.sampling_params.get("[default]")
            sampling_params.append(params)
            indices.append(idx)

        logger.info(f"从{len(blocks)}个总块中准备了{len(block_images)}个块用于提取")
        if skipped_types:
            logger.debug(f"跳过的块类型: {set(skipped_types)}")

        return block_images, prompts, sampling_params, indices

    def post_process(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        """Post-process extracted content blocks (clean text, merge blocks, etc)."""
        original_count = len(blocks)
        blocks = post_process(
            blocks,
            handle_equation_block=self.handle_equation_block,
            abandon_list=self.abandon_list,
            abandon_paratext=self.abandon_paratext,
            debug=self.debug,
        )
        logger.info(f"后处理: {original_count}个块 -> {len(blocks)}个块")
        return blocks


class DeepSeekClient(BaseOCRClient):
    """DeepSeek-OCR client implementation."""

    def __init__(
        self,
        backend: Literal[
            "http-client",
            "transformers",
            "mlx-engine",
            "lmdeploy-engine",
            "vllm-engine",
            "vllm-async-engine",
        ],
        model_name: str | None = None,
        server_url: str | None = None,
        server_headers: dict[str, str] | None = None,
        model=None,
        processor=None,
        vllm_llm=None,
        vllm_async_llm=None,
        lmdeploy_engine=None,
        model_path: str | None = None,
        prompts: dict[str, str] = DEFAULT_DEEPSEEK_PROMPTS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: dict[str, SamplingParams] = DEFAULT_DEEPSEEK_SAMPLING_PARAMS,
        layout_image_size: tuple[int, int] = (1036, 1036),
        min_image_edge: int = 28,
        max_image_edge_ratio: float = 50,
        handle_equation_block: bool = True,
        abandon_list: bool = False,
        abandon_paratext: bool = False,
        incremental_priority: bool = False,
        max_concurrency: int = 100,
        executor: Executor | None = None,
        batch_size: int = 0,
        http_timeout: int = 600,
        use_tqdm: bool = True,
        debug: bool = False,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5,
    ) -> None:
        logger.info("=" * 60)
        logger.info("DeepSeekClient.__init__ called")
        logger.info(f"  backend: {backend}")
        logger.info(f"  model_name: {model_name}")
        logger.info(f"  server_url: {server_url}")

        # Create helper
        helper = DeepSeekClientHelper(
            backend=backend,
            prompts=prompts,
            sampling_params=sampling_params,
            layout_image_size=layout_image_size,
            min_image_edge=min_image_edge,
            max_image_edge_ratio=max_image_edge_ratio,
            handle_equation_block=handle_equation_block,
            abandon_list=abandon_list,
            abandon_paratext=abandon_paratext,
            debug=debug,
        )

        super().__init__(backend, model_name, helper)

        # Create VLM client
        logger.info("Creating VlmClient for DeepSeek...")
        self.client = new_vlm_client(
            backend=backend,
            model_name=model_name,
            server_url=server_url,
            server_headers=server_headers,
            model=model,
            processor=processor,
            lmdeploy_engine=lmdeploy_engine,
            vllm_llm=vllm_llm,
            vllm_async_llm=vllm_async_llm,
            system_prompt=system_prompt,
            allow_truncated_content=True,
            max_concurrency=max_concurrency,
            batch_size=batch_size,
            http_timeout=http_timeout,
            use_tqdm=use_tqdm,
            debug=debug,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
        )

        self.prompts = prompts
        self.sampling_params = sampling_params
        self.incremental_priority = incremental_priority
        self.max_concurrency = max_concurrency
        self.executor = executor
        self.use_tqdm = use_tqdm
        self.debug = debug
        self.client_name = "deepseek"  # For JSON format conversion

        logger.info(f"DeepSeekClient initialized successfully")
        logger.info("=" * 60)

    def layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
    ) -> list[ContentBlock]:
        """Detect layout elements in a single image."""
        logger.debug(f"DeepSeek检测布局，图像尺寸: {image.size}")
        layout_image = self.helper.prepare_for_layout(image)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        output = self.client.predict(layout_image, prompt, params, priority)
        blocks = self.helper.parse_layout_output(output)
        logger.info(f"DeepSeek布局检测完成: 找到{len(blocks)}个块")
        return blocks

    def two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
    ) -> list[ContentBlock]:
        """Two-step document extraction for DeepSeek."""
        logger.info("开始DeepSeek两步提取(同步)")
        blocks = self.layout_detect(image, priority)
        logger.debug(f"第1步完成: 检测到{len(blocks)}个块")

        block_images, prompts, params, indices = self.helper.prepare_for_extract(image, blocks)
        logger.debug(f"第2步开始: 从{len(block_images)}个块中提取内容")

        outputs = self.client.batch_predict(block_images, prompts, params, priority)
        for idx, output in zip(indices, outputs):
            blocks[idx].content = _extract_text_from_structured_output(output)

        blocks = self.helper.post_process(blocks)
        logger.info(f"DeepSeek两步提取完成: {len(blocks)}个包含内容的块")
        return blocks

    def _extract_text(self, output: str) -> str:
        """Extract text content from structured model output."""
        return _extract_text_from_structured_output(output)

    def batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
    ) -> list[list[ContentBlock]]:
        """Batch layout detection for multiple images."""
        logger.info(f"批量布局检测开始，共{len(images)}个图像")
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        layout_images = self.helper.batch_prepare_for_layout(self.executor, images)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        outputs = self.client.batch_predict(layout_images, [prompt] * len(images), [params] * len(images), priority)
        blocks_list = self.helper.batch_parse_layout_output(self.executor, outputs)
        logger.info(f"批量布局检测完成: 共{sum(len(b) for b in blocks_list)}个块")
        return blocks_list

    async def aio_batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
    ) -> list[list[ContentBlock]]:
        """Async batch layout detection for multiple images."""
        logger.info(f"异步批量布局检测开始，共{len(images)}个图像")
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        layout_images = self.helper.batch_prepare_for_layout(self.executor, images)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")

        # Use async predict if available
        if hasattr(self.client, 'aio_batch_predict'):
            outputs = await self.client.aio_batch_predict(layout_images, [prompt] * len(images), [params] * len(images), priority)
        else:
            # Fallback to sync version
            outputs = self.client.batch_predict(layout_images, [prompt] * len(images), [params] * len(images), priority)

        blocks_list = self.helper.batch_parse_layout_output(self.executor, outputs)
        logger.info(f"异步批量布局检测完成: 共{sum(len(b) for b in blocks_list)}个块")
        return blocks_list

    # batch_two_step_extract and aio_batch_two_step_extract are now inherited from BaseOCRClient (unified stepping mode)
