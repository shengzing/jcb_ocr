"""Hunyuan-OCR client implementation.

Hunyuan-OCR output format (XML-based):
- Layout Detection: <ref>content</ref><quad>(x1,y1),(x2,y2)</quad>
- Table Output: HTML table format
- Text Recognition: XML tags with quad coordinates

Key differences from Qwen/DeepSeek:
- XML parsing instead of JSON
- Absolute pixel coordinates in quad format
- HTML table output for structured data
"""

from __future__ import annotations

import asyncio
import math
import re
import xml.etree.ElementTree as ET
from concurrent.futures import Executor
from typing import Literal, Sequence

from loguru import logger
from PIL import Image

from .base_client import BaseOCRClient, BaseOCRClientHelper
from .post_process import post_process
from .prompt_library import HUNYUAN_DEFAULT_PROMPTS
from .structs import BLOCK_TYPES, ContentBlock
from .vlm_client import DEFAULT_SYSTEM_PROMPT, SamplingParams, new_vlm_client
from .vlm_client.utils import gather_tasks, get_png_bytes, get_rgb_image

# Hunyuan-specific constants and patterns
_hunyuan_ref_quad_re = r"<ref>([^<]*)</ref><quad>\\(([0-9]+),([0-9]+)\\),\\(([0-9]+),([0-9]+)\\)</quad>"
_hunyuan_table_re = r"<table[^>]*>.*?</table>"


class HunyuanSamplingParams(SamplingParams):
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


# # Hunyuan-specific prompts (Chinese natural language)
# DEFAULT_HUNYUAN_PROMPTS: dict[str, str] = {
#     "table": "\n提取表格内容，请返回完整的 HTML 表格结构：",
#     "equation": "\n识别该区域的公式，输出准确的 LaTeX：",
#     "title": "\n识别标题内容，保持原有的编号、标点、格式：",
#     "list": "\n识别列表内容，保持原有的项目符号或编号，嵌套用两个空格缩进：",
#     "image": "\n描述图片内容，清晰、简洁地说明关键视觉元素、物体、人物、图中文字、图表或重要细节：",
#     "handwritten": "\n识别手写内容，保持原始换行，难以辨认的字符用[?]占位：",
#     "seal": "\n识别印章或签章上的文字，只输出精确的文字内容：",
#     "[default]": "\n识别文本内容并保持原有的行文顺序：",
#     "[layout]": (
#         "\n检测文档布局，按照阅读顺序标注每个区域的类型和位置。"
#         "类型必须从{text, title, list, table, image, handwritten, seal}中选择，手写内容必须标注为handwritten，印章/签章为seal。"
#     ),
# }

DEFAULT_HUNYUAN_SAMPLING_PARAMS: dict[str, SamplingParams] = {
    "table": HunyuanSamplingParams(presence_penalty=1.0, frequency_penalty=0.005, temperature=0.1),
    "equation": HunyuanSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "title": HunyuanSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "list": HunyuanSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "image": HunyuanSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "handwritten": HunyuanSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "seal": HunyuanSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[default]": HunyuanSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[layout]": HunyuanSamplingParams(temperature=0.0),
}

DEFAULT_HUNYUAN_PROMPTS: dict[str, str] = HUNYUAN_DEFAULT_PROMPTS



def _convert_hunyuan_bbox(x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int) -> list[float] | None:
    """Convert Hunyuan absolute pixel coordinates to normalized (0.0-1.0) format.

    Args:
        x1, y1, x2, y2: Absolute pixel coordinates
        img_width, img_height: Image dimensions for normalization

    Returns:
        Normalized bounding box [x1, y1, x2, y2] range 0.0-1.0, or None if invalid
    """
    if img_width <= 0 or img_height <= 0:
        logger.warning(f"无效的图像尺寸: {img_width}x{img_height}")
        return None

    # Normalize to 0.0-1.0 range
    norm_x1 = x1 / img_width
    norm_y1 = y1 / img_height
    norm_x2 = x2 / img_width
    norm_y2 = y2 / img_height

    # Ensure proper order and bounds
    norm_x1, norm_x2 = sorted([norm_x1, norm_x2])
    norm_y1, norm_y2 = sorted([norm_y1, norm_y2])

    # Clamp to valid range
    norm_x1 = max(0.0, min(1.0, norm_x1))
    norm_y1 = max(0.0, min(1.0, norm_y1))
    norm_x2 = max(0.0, min(1.0, norm_x2))
    norm_y2 = max(0.0, min(1.0, norm_y2))

    if norm_x1 >= norm_x2 or norm_y1 >= norm_y2:
        logger.debug(f"无效的边界框(宽度或高度为零): ({x1},{y1}) -> ({x2},{y2})")
        return None

    return [norm_x1, norm_y1, norm_x2, norm_y2]


def _infer_block_type_from_content(content: str) -> str:
    """Infer block type from content for Hunyuan output.

    Args:
        content: Text content from <ref> tag

    Returns:
        Inferred block type
    """
    content_lower = content.lower().strip()

    # Seal patterns
    if any(keyword in content_lower for keyword in ["印章", "公章", "盖章", "盖印", "钤章", "签章", "seal", "stamp"]):
        return "seal"

    # Handwritten patterns
    if any(keyword in content_lower for keyword in ["手写", "手寫", "手迹", "手跡", "笔迹", "筆跡", "手写体", "handwritten"]):
        return "handwritten"

    # Table patterns
    if any(keyword in content_lower for keyword in ["表", "table", "行", "列", "row", "column"]):
        return "table"

    # Title patterns
    if (
        len(content_lower) < 50
        and any(keyword in content_lower for keyword in ["章", "节", "标题", "title", "chapter", "section"])
        or content.isupper()  # All caps might be title
        or re.match(r"^[\\d\\.]+\\s", content)  # Starts with numbers (1.1, 2.3, etc.)
    ):
        return "title"

    # Equation patterns
    if any(keyword in content for keyword in ["=", "∑", "∫", "√", "π", "α", "β", "γ"]) or re.search(r"\\$.*\\$", content):
        return "equation"

    # List patterns
    if content_lower.startswith(("•", "◦", "▪", "-", "*")) or re.match(r"^[\\d]+[\\.)]\\s", content):
        return "list"

    # Default to text
    return "text"


def _normalize_hunyuan_block_type(raw_type: str | None) -> str:
    """Normalize Hunyuan block type to BLOCK_TYPES format."""
    if not raw_type:
        return "text"

    type_mapping = {
        "文本": "text",
        "标题": "title",
        "表格": "table",
        "图片": "image",
        "图像": "image",
        "公式": "equation",
        "列表": "list",
        "代码": "code",
        "页眉": "header",
        "页脚": "footer",
        "页码": "page_number",
        "手写": "handwritten",
        "手寫": "handwritten",
        "handwritten": "handwritten",
        "印章": "seal",
        "公章": "seal",
        "盖章": "seal",
        "签章": "seal",
        "seal": "seal",
        "stamp": "seal",
    }

    raw_lower = raw_type.lower().strip()
    for chinese, english in type_mapping.items():
        if chinese in raw_type or english in raw_lower:
            return english

    # If no match found, try to infer from name
    if any(keyword in raw_lower for keyword in ["text", "content", "paragraph"]):
        return "text"
    elif any(keyword in raw_lower for keyword in ["title", "heading", "header"]):
        return "title"
    elif any(keyword in raw_lower for keyword in ["table", "grid"]):
        return "table"
    elif any(keyword in raw_lower for keyword in ["image", "picture", "figure"]):
        return "image"

    return "text"


class HunyuanClientHelper(BaseOCRClientHelper):
    """Helper class for Hunyuan-OCR model-specific logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_image_size = None  # Store current image size for coordinate conversion

    # resize_by_need() is inherited from BaseOCRClientHelper

    def prepare_for_layout(self, image: Image.Image) -> Image.Image | bytes:
        """Prepare image for layout detection by resizing and converting format."""
        original_size = image.size
        self.current_image_size = original_size  # Store for coordinate conversion
        image = get_rgb_image(image)
        image = image.resize(self.layout_image_size, Image.Resampling.BICUBIC)
        logger.debug(f"准备布局图像: {original_size} -> {self.layout_image_size}")

        if self.backend == "http-client":
            return get_png_bytes(image)
        return image

    def parse_layout_output(self, output: str) -> list[ContentBlock]:
        """Parse Hunyuan layout detection model output to ContentBlock objects.

        Hunyuan format: <ref>content</ref><quad>(x1,y1),(x2,y2)</quad>
        """
        logger.debug(f'Hunyuan解析内容：{output}')
        blocks: list[ContentBlock] = []

        # First try to find <ref><quad> patterns
        ref_quad_matches = re.findall(_hunyuan_ref_quad_re, output, re.DOTALL)
        for match in ref_quad_matches:
            content, x1_str, y1_str, x2_str, y2_str = match
            try:
                x1, y1, x2, y2 = int(x1_str), int(y1_str), int(x2_str), int(y2_str)

                # Convert to normalized coordinates
                if self.current_image_size:
                    img_width, img_height = self.current_image_size
                    bbox = _convert_hunyuan_bbox(x1, y1, x2, y2, img_width, img_height)
                else:
                    # Fallback: assume coordinates are already normalized percentage * 1000
                    bbox = [x1/1000.0, y1/1000.0, x2/1000.0, y2/1000.0]

                if bbox is None:
                    continue

                # Infer block type from content
                block_type = _infer_block_type_from_content(content)

                block = ContentBlock(block_type, bbox, content=content.strip())
                blocks.append(block)
                logger.debug(f"解析到块: 类型={block.type}, 内容='{content[:30]}...', bbox={bbox}")

            except (ValueError, IndexError) as e:
                logger.warning(f"解析坐标失败: {match}, 错误: {e}")
                continue

        # Also try to parse HTML tables
        table_matches = re.findall(_hunyuan_table_re, output, re.DOTALL | re.IGNORECASE)
        for table_html in table_matches:
            # For tables, we don't have explicit coordinates, so create a full-page block
            bbox = [0.0, 0.0, 1.0, 1.0]
            block = ContentBlock("table", bbox, content=table_html.strip())
            blocks.append(block)
            logger.debug(f"解析到表格块: 长度={len(table_html)} 字符")

        # If no structured patterns found, treat as plain text
        if not blocks:
            logger.info("未找到结构化模式，将输出视为纯文本")
            if output.strip():
                bbox = [0.0, 0.0, 1.0, 1.0]
                block = ContentBlock("text", bbox, content=output.strip())
                blocks.append(block)

        logger.info(f"从Hunyuan输出解析出{len(blocks)}个内容块")
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
            # Skip only image blocks (to ensure all 7 elements including list are extracted)
            if block.type in ("image",):
                skipped_types.append(block.type)
                continue

            # Crop block region from image
            x1, y1, x2, y2 = block.bbox
            scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            block_image = image.crop(scaled_bbox)

            # Rotate if needed (Hunyuan doesn't typically provide angle, but check anyway)
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

        # Hunyuan-specific post-processing
        processed_blocks = []
        for block in blocks:
            if block.content:
                # Clean up HTML content for tables
                if block.type == "table" and "<table" in block.content:
                    # Keep HTML content as-is for tables
                    processed_blocks.append(block)
                else:
                    # Clean text content
                    content = block.content
                    # Remove XML tags from non-table content
                    content = re.sub(r"<[^>]+>", "", content)
                    # Clean whitespace
                    content = re.sub(r"\\s+", " ", content).strip()
                    if content:  # Only keep non-empty content
                        block.content = content
                        processed_blocks.append(block)

        # Apply standard post-processing
        processed_blocks = post_process(
            processed_blocks,
            handle_equation_block=self.handle_equation_block,
            abandon_list=self.abandon_list,
            abandon_paratext=self.abandon_paratext,
            debug=self.debug,
        )

        logger.info(f"后处理: {original_count}个块 -> {len(processed_blocks)}个块")
        return processed_blocks


class HunyuanClient(BaseOCRClient):
    """Hunyuan-OCR client implementation."""

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
        prompts: dict[str, str] = DEFAULT_HUNYUAN_PROMPTS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: dict[str, SamplingParams] = DEFAULT_HUNYUAN_SAMPLING_PARAMS,
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
        logger.info("HunyuanClient.__init__ called")
        logger.info(f"  backend: {backend}")
        logger.info(f"  model_name: {model_name}")
        logger.info(f"  server_url: {server_url}")

        # Create helper
        helper = HunyuanClientHelper(
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
        logger.info("Creating VlmClient for Hunyuan...")
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
        self.client_name = "hunyuan"  # For JSON format conversion

        logger.info(f"HunyuanClient initialized successfully")
        logger.info("=" * 60)

    def layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
    ) -> list[ContentBlock]:
        """Detect layout elements in a single image."""
        logger.debug(f"Hunyuan检测布局，图像尺寸: {image.size}")
        layout_image = self.helper.prepare_for_layout(image)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        output = self.client.predict(layout_image, prompt, params, priority)
        blocks = self.helper.parse_layout_output(output)
        logger.info(f"Hunyuan布局检测完成: 找到{len(blocks)}个块")
        return blocks

    def two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
    ) -> list[ContentBlock]:
        """Two-step document extraction for Hunyuan."""
        logger.info("开始Hunyuan两步提取(同步)")
        blocks = self.layout_detect(image, priority)
        logger.debug(f"第1步完成: 检测到{len(blocks)}个块")

        block_images, prompts, params, indices = self.helper.prepare_for_extract(image, blocks)
        logger.debug(f"第2步开始: 从{len(block_images)}个块中提取内容")

        if block_images:
            outputs = self.client.batch_predict(block_images, prompts, params, priority)
            for idx, output in zip(indices, outputs):
                blocks[idx].content = output

        blocks = self.helper.post_process(blocks)
        logger.info(f"Hunyuan两步提取完成: {len(blocks)}个包含内容的块")
        return blocks

    def _extract_text(self, output: str) -> str:
        """Extract text content from model output.

        Hunyuan directly uses the output without structured extraction.
        """
        return output

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
