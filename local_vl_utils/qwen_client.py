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
from .prompt_library import QWEN_DEFAULT_PROMPTS
from .structs import BLOCK_TYPES, ContentBlock
from .vlm_client import DEFAULT_SYSTEM_PROMPT, SamplingParams, new_vlm_client
from .vlm_client.utils import gather_tasks, get_png_bytes, get_rgb_image

_layout_re = r"^\s*\{\s*\"bbox_2d\"\s*:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*,\s*\"label\"\s*:\s*\"([^\"]+)\"(?:,\s*[^}]*)?\s*\}\s*,?\s*(.*)$"
_legacy_layout_re = r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$"
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


class QwenSamplingParams(SamplingParams):
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


DEFAULT_PROMPTS: dict[str, str] = QWEN_DEFAULT_PROMPTS

DEFAULT_SAMPLING_PARAMS: dict[str, SamplingParams] = {
    "table": QwenSamplingParams(presence_penalty=1.0, frequency_penalty=0.005),
    "equation": QwenSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[default]": QwenSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[layout]": QwenSamplingParams(),
}

ANGLE_MAPPING: dict[str, Literal[0, 90, 180, 270]] = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}


def _convert_bbox(bbox: Sequence[int] | Sequence[str]) -> list[float] | None:
    """
    将边界框坐标从整数(0-1000)转换为浮点数(0.0-1.0)。

    参数:
        bbox: 包含4个坐标的序列 [x1, y1, x2, y2]，范围0-1000

    返回:
        归一化后的边界框 [x1, y1, x2, y2]，范围0.0-1.0，如果无效则返回None
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
    """
    从布局检测输出中解析旋转角度。

    参数:
        tail: 布局检测输出的尾部字符串

    返回:
        旋转角度 (0, 90, 180, 270)，如果未找到旋转标记则返回None
    """
    for token, angle in ANGLE_MAPPING.items():
        if token in tail:
            logger.debug(f"找到旋转标记 '{token}' -> 角度 {angle}")
            return angle
    logger.debug(f"未找到旋转标记: {tail}")
    return None


def _normalize_block_type(label: str | None) -> str:
    """将模型输出的类型文本标准化为BLOCK_TYPES中的形式。"""
    if label is None:
        return ""
    text = str(label)
    for token in ANGLE_MAPPING:
        text = text.replace(token, "")
    text = text.strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    text = _TYPE_ALIASES.get(text, text)
    return text


def _find_angle_from_sources(*sources: object) -> Literal[None, 0, 90, 180, 270]:
    """从多个可能的字段中解析旋转角度。"""
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


class QwenClientHelper(BaseOCRClientHelper):
    """Helper class for Qwen-OCR model-specific logic."""

    def __init__(
        self,
        backend: str,
        prompts: dict[str, str],
        sampling_params: dict[str, SamplingParams],
        layout_image_size: tuple[int, int],
        min_image_edge: int,
        max_image_edge_ratio: float,
        handle_equation_block: bool,
        abandon_list: bool,
        abandon_paratext: bool,
        debug: bool,
    ) -> None:
        super().__init__(
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

    # resize_by_need() and prepare_for_layout() are inherited from BaseOCRClientHelper

    def parse_layout_output(self, output: str) -> list[ContentBlock]:
        """
        将布局检测模型输出解析为ContentBlock对象。

        参数:
            output: 布局检测的原始字符串输出

        返回:
            解析后的ContentBlock对象列表
        """
        logger.debug(f'解析内容：{output}')
        blocks: list[ContentBlock] = []
        stripped_output = output.strip()

        def _append_block_from_record(record: dict, origin: str, extra_sources: tuple[object, ...] = ()) -> None:
            block = _record_to_block(record, origin, extra_sources)
            if block is not None:
                blocks.append(block)

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

        lines = output.split("\n")

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
                x1, y1, x2, y2, raw_label, tail = match.groups()
                record = {"bbox_2d": [x1, y1, x2, y2], "label": raw_label}
                _append_block_from_record(record, f"第{line_num}行结构化", extra_sources=(tail,))
                continue

            match = re.match(_legacy_layout_re, line)
            if match:
                x1, y1, x2, y2, ref_type_raw, tail = match.groups()
                record = {"bbox_2d": [x1, y1, x2, y2], "label": ref_type_raw}
                _append_block_from_record(record, f"第{line_num}行legacy", extra_sources=(tail,))
                continue

            logger.warning(f"第{line_num}行不匹配布局格式: {line}")

        logger.info(f"从{len(lines)}行中解析出{len(blocks)}个内容块")
        return blocks

    def prepare_for_extract(
        self,
        image: Image.Image,
        blocks: list[ContentBlock],
    ) -> tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]:
        """
        通过裁剪和处理区域为文本提取准备内容块。

        参数:
            image: 原始PIL图像
            blocks: 来自布局检测的ContentBlock对象列表

        返回:
            元组(block_images, prompts, sampling_params, indices):
                - block_images: 每个块的裁剪和处理后的图像
                - prompts: 每个块类型的提取提示词
                - sampling_params: 每个块类型的采样参数
                - indices: 要提取的块的原始索引
        """
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

            # 从图像中裁剪块区域
            x1, y1, x2, y2 = block.bbox
            scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            block_image = image.crop(scaled_bbox)

            # 如果需要则旋转
            if block.angle in [90, 180, 270]:
                block_image = block_image.rotate(block.angle, expand=True)
                logger.debug(f"块{idx}: 旋转{block.angle}度")

            # 调整大小以满足约束
            block_image = self.resize_by_need(block_image)

            # 对于HTTP后端转换为字节
            if self.backend == "http-client":
                block_image = get_png_bytes(block_image)

            block_images.append(block_image)

            # 获取块类型的提示词和采样参数
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
        """
        后处理提取的内容块(清理文本、合并块等)。

        参数:
            blocks: 包含提取内容的ContentBlock对象列表

        返回:
            后处理后的ContentBlock对象列表
        """
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

    def batch_prepare_for_layout(
        self,
        executor: Executor | None,
        images: list[Image.Image],
    ) -> list[Image.Image | bytes]:
        """
        批量准备多个图像用于布局检测。

        参数:
            executor: 可选的线程/进程池执行器用于并行处理
            images: 输入的PIL图像列表

        返回:
            准备好的图像列表(PIL图像或PNG字节)
        """
        logger.debug(f"批量准备{len(images)}个图像用于布局检测")
        if executor is None:
            return [self.prepare_for_layout(im) for im in images]
        return list(executor.map(self.prepare_for_layout, images))

    def batch_parse_layout_output(
        self,
        executor: Executor | None,
        outputs: list[str],
    ) -> list[list[ContentBlock]]:
        """
        批量解析多个布局检测输出。

        参数:
            executor: 可选的线程/进程池执行器用于并行处理
            outputs: 来自布局检测的原始字符串输出列表

        返回:
            ContentBlock列表的列表，每个输出对应一个列表
        """
        logger.debug(f"批量解析{len(outputs)}个布局输出")
        if executor is None:
            return [self.parse_layout_output(output) for output in outputs]
        return list(executor.map(self.parse_layout_output, outputs))

    def batch_prepare_for_extract(
        self,
        executor: Executor | None,
        images: list[Image.Image],
        blocks_list: list[list[ContentBlock]],
    ) -> list[tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]]:
        """
        批量准备从多个图像中提取内容块。

        参数:
            executor: 可选的线程/进程池执行器用于并行处理
            images: 原始PIL图像列表
            blocks_list: ContentBlock列表的列表，每个图像对应一个列表

        返回:
            元组列表(block_images, prompts, sampling_params, indices)，每个图像对应一个元组
        """
        logger.debug(f"批量准备{len(images)}个图像用于内容提取")
        if executor is None:
            return [self.prepare_for_extract(im, bls) for im, bls in zip(images, blocks_list)]
        return list(executor.map(self.prepare_for_extract, images, blocks_list))

    def batch_post_process(
        self,
        executor: Executor | None,
        blocks_list: list[list[ContentBlock]],
    ) -> list[list[ContentBlock]]:
        """
        批量后处理多个图像的内容块。

        参数:
            executor: 可选的线程/进程池执行器用于并行处理
            blocks_list: 要后处理的ContentBlock列表的列表

        返回:
            后处理后的ContentBlock列表的列表
        """
        logger.debug(f"批量后处理{len(blocks_list)}个块列表")
        if executor is None:
            return [self.post_process(blocks) for blocks in blocks_list]
        return list(executor.map(self.post_process, blocks_list))

    async def aio_prepare_for_layout(
        self,
        executor: Executor | None,
        image: Image.Image,
    ) -> Image.Image | bytes:
        """
        prepare_for_layout的异步版本。

        参数:
            executor: 可选的线程/进程池执行器
            image: 输入的PIL图像

        返回:
            准备好的图像(PIL图像或PNG字节)
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.prepare_for_layout, image)

    async def aio_parse_layout_output(
        self,
        executor: Executor | None,
        output: str,
    ) -> list[ContentBlock]:
        """
        parse_layout_output的异步版本。

        参数:
            executor: 可选的线程/进程池执行器
            output: 布局检测的原始字符串输出

        返回:
            解析后的ContentBlock对象列表
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.parse_layout_output, output)

    async def aio_prepare_for_extract(
        self,
        executor: Executor | None,
        image: Image.Image,
        blocks: list[ContentBlock],
    ) -> tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]:
        """
        prepare_for_extract的异步版本。

        参数:
            executor: 可选的线程/进程池执行器
            image: 原始PIL图像
            blocks: ContentBlock对象列表

        返回:
            元组(block_images, prompts, sampling_params, indices)
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.prepare_for_extract, image, blocks)

    async def aio_post_process(
        self,
        executor: Executor | None,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """
        post_process的异步版本。

        参数:
            executor: 可选的线程/进程池执行器
            blocks: 要后处理的ContentBlock对象列表

        返回:
            后处理后的ContentBlock对象列表
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.post_process, blocks)


class QwenClient(BaseOCRClient):
    """Qwen-based OCR client with two-step extraction (layout detection + content extraction)."""

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
        model=None,  # transformers model
        processor=None,  # transformers processor
        vllm_llm=None,  # vllm.LLM model
        vllm_async_llm=None,  # vllm.v1.engine.async_llm.AsyncLLM instance
        lmdeploy_engine=None,  # lmdeploy.serve.vl_async_engine.VLAsyncEngine instance
        model_path: str | None = None,
        prompts: dict[str, str] = DEFAULT_PROMPTS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: dict[str, SamplingParams] = DEFAULT_SAMPLING_PARAMS,
        layout_image_size: tuple[int, int] = (1036, 1036),
        min_image_edge: int = 28,
        max_image_edge_ratio: float = 50,
        handle_equation_block: bool = True,
        abandon_list: bool = False,
        abandon_paratext: bool = False,
        incremental_priority: bool = False,
        max_concurrency: int = 100,
        executor: Executor | None = None,
        batch_size: int = 0,  # for transformers and vllm-engine
        http_timeout: int = 600,  # for http-client backend only
        use_tqdm: bool = True,
        debug: bool = False,
        max_retries: int = 3,  # for http-client backend only
        retry_backoff_factor: float = 0.5,  # for http-client backend only
    ) -> None:
        from loguru import logger
        # logger.info("=" * 60)
        # logger.info("QwenClient.__init__ called")
        # logger.info(f"  backend: {backend}")
        # logger.info(f"  model_name: {model_name}")
        # logger.info(f"  server_url: {server_url}")
        # logger.info(f"  has server_headers: {server_headers is not None}")
        # logger.info(f"  model_path: {model_path}")
        # logger.info(f"  batch_size: {batch_size}")
        # logger.info(f"  http_timeout: {http_timeout}")
        # logger.info(f"  max_retries: {max_retries}")

        # Backend-specific model initialization (before creating helper)
        if backend == "transformers":
            if model is None or processor is None:
                if not model_path:
                    raise ValueError("model_path must be provided when model or processor is None.")

                try:
                    from transformers import (
                        AutoProcessor,
                        Qwen2VLForConditionalGeneration,
                    )
                    from transformers import __version__ as transformers_version
                except ImportError:
                    raise ImportError("Please install transformers to use the transformers backend.")

                if model is None:
                    dtype_key = "torch_dtype"
                    ver_parts = transformers_version.split(".")
                    if len(ver_parts) >= 2 and int(ver_parts[0]) >= 4 and int(ver_parts[1]) >= 56:
                        dtype_key = "dtype"
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        device_map="auto",
                        **{dtype_key: "auto"},  # type: ignore
                    )
                if processor is None:
                    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        elif backend == "mlx-engine":
            if model is None or processor is None:
                if not model_path:
                    raise ValueError("model_path must be provided when model or processor is None.")

                try:
                    from mlx_vlm import load as mlx_load
                except ImportError:
                    raise ImportError("Please install mlx-vlm to use the mlx-engine backend.")
                model, processor = mlx_load(model_path)

        elif backend == "lmdeploy-engine":
            if lmdeploy_engine is None:
                if not model_path:
                    raise ValueError("model_path must be provided when lmdeploy_engine is None.")

                try:
                    from lmdeploy.serve.vl_async_engine import VLAsyncEngine
                except ImportError:
                    raise ImportError("Please install lmdeploy to use the lmdeploy-engine backend.")

                lmdeploy_engine = VLAsyncEngine(
                    model_path,
                )

        elif backend == "vllm-engine":
            if vllm_llm is None:
                if not model_path:
                    raise ValueError("model_path must be provided when vllm_llm is None.")

                try:
                    import vllm
                except ImportError:
                    raise ImportError("Please install vllm to use the vllm-engine backend.")

                vllm_llm = vllm.LLM(model_path)

        elif backend == "vllm-async-engine":
            if vllm_async_llm is None:
                if not model_path:
                    raise ValueError("model_path must be provided when vllm_async_llm is None.")

                try:
                    from vllm.engine.arg_utils import AsyncEngineArgs
                    from vllm.v1.engine.async_llm import AsyncLLM
                except ImportError:
                    raise ImportError("Please install vllm to use the vllm-async-engine backend.")

                vllm_async_llm = AsyncLLM.from_engine_args(AsyncEngineArgs(model_path))

        logger.info("Creating QwenClientHelper...")
        helper = QwenClientHelper(
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
        logger.info("QwenClientHelper created successfully")

        # Initialize base class
        super().__init__(
            backend=backend,
            model_name=model_name,
            helper=helper,
        )

        # Create VlmClient after super().__init__() to avoid being overwritten by base class
        logger.info("Creating VlmClient...")
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
            allow_truncated_content=True,  # Allow truncated content for Qwen
            max_concurrency=max_concurrency,
            batch_size=batch_size,
            http_timeout=http_timeout,
            use_tqdm=use_tqdm,
            debug=debug,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
        )
        logger.info("VlmClient created successfully")

        # QwenClient-specific attributes
        self.prompts = prompts
        self.sampling_params = sampling_params
        self.incremental_priority = incremental_priority
        self.max_concurrency = max_concurrency
        self.executor = executor
        self.use_tqdm = use_tqdm
        self.debug = debug
        self.client_name = "qwen"  # For JSON format conversion

        logger.info(f"QwenClient initialized successfully")
        logger.info("=" * 60)

    def _extract_text(self, output: str) -> str:
        """Extract text content from structured model output.

        QwenClient uses structured JSON format for content extraction.

        Args:
            output: Raw model output string

        Returns:
            Extracted text content
        """
        return _extract_text_from_structured_output(output)

    def layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
    ) -> list[ContentBlock]:
        """
        检测单个图像中的布局元素(文本块、表格、图像等)。

        参数:
            image: 输入的PIL图像
            priority: 请求调度的可选优先级

        返回:
            检测到的ContentBlock对象列表，包含边界框和类型
        """
        logger.debug(f"检测布局，图像尺寸: {image.size}")
        layout_image = self.helper.prepare_for_layout(image)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        output = self.client.predict(layout_image, prompt, params, priority)
        blocks = self.helper.parse_layout_output(output)
        logger.info(f"布局检测完成: 找到{len(blocks)}个块")
        return blocks

    def batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
    ) -> list[list[ContentBlock]]:
        """
        检测多个图像中的布局元素(批处理)。

        参数:
            images: 输入的PIL图像列表
            priority: 请求调度的可选优先级

        返回:
            ContentBlock列表的列表，每个图像对应一个列表
        """
        logger.info(f"批量布局检测开始，共{len(images)}个图像")
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        layout_images = self.helper.batch_prepare_for_layout(self.executor, images)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        outputs = self.client.batch_predict(layout_images, prompt, params, priority)
        blocks_list = self.helper.batch_parse_layout_output(self.executor, outputs)
        logger.info(f"批量布局检测完成: 共{sum(len(b) for b in blocks_list)}个块")
        return blocks_list

    async def aio_layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        """
        layout_detect的异步版本。

        参数:
            image: 输入的PIL图像
            priority: 请求调度的可选优先级
            semaphore: 并发控制的可选信号量

        返回:
            检测到的ContentBlock对象列表
        """
        logger.debug(f"异步检测布局，图像尺寸: {image.size}")
        layout_image = await self.helper.aio_prepare_for_layout(self.executor, image)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        if semaphore is None:
            output = await self.client.aio_predict(layout_image, prompt, params, priority)
        else:
            async with semaphore:
                output = await self.client.aio_predict(layout_image, prompt, params, priority)
        blocks = await self.helper.aio_parse_layout_output(self.executor, output)
        logger.info(f"异步布局检测完成: 找到{len(blocks)}个块")
        return blocks

    async def aio_batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        """
        多个图像的异步批量布局检测。

        参数:
            images: 输入的PIL图像列表
            priority: 请求调度的可选优先级
            semaphore: 并发控制的可选信号量

        返回:
            ContentBlock列表的列表，每个图像对应一个列表
        """
        logger.info(f"异步批量布局检测开始，共{len(images)}个图像")
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        layout_images = await gather_tasks(
            tasks=[self.helper.aio_prepare_for_layout(self.executor, im) for im in images],
            use_tqdm=self.use_tqdm,
            tqdm_desc="布局准备",
        )
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        outputs = await self.client.aio_batch_predict(
            layout_images,
            [prompt] * len(images),
            [params] * len(images),
            priority,
            semaphore=semaphore,
            use_tqdm=self.use_tqdm,
            tqdm_desc="布局检测",
        )
        blocks_list = await gather_tasks(
            tasks=[self.helper.aio_parse_layout_output(self.executor, out) for out in outputs],
            use_tqdm=self.use_tqdm,
            tqdm_desc="布局输出解析",
        )
        logger.info(f"异步批量布局检测完成: 共{sum(len(b) for b in blocks_list)}个块")
        return blocks_list

    def content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
    ) -> str | None:
        blocks = [ContentBlock(type, [0.0, 0.0, 1.0, 1.0])]
        block_images, prompts, params, _ = self.helper.prepare_for_extract(image, blocks)
        if not (block_images and prompts and params):
            return None
        output = self.client.predict(block_images[0], prompts[0], params[0], priority)
        blocks[0].content = _extract_text_from_structured_output(output)
        blocks = self.helper.post_process(blocks)
        return blocks[0].content if blocks else None

    def batch_content_extract(
        self,
        images: list[Image.Image],
        types: Sequence[str] | str = "text",
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str | None]:
        if isinstance(types, str):
            types = [types] * len(images)
        if len(types) != len(images):
            raise Exception("Length of types must match length of images")
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        blocks_list = [[ContentBlock(type, [0.0, 0.0, 1.0, 1.0])] for type in types]
        all_images: list[Image.Image | bytes] = []
        all_prompts: list[str] = []
        all_params: list[SamplingParams | None] = []
        all_indices: list[tuple[int, int]] = []
        prepared_inputs = self.helper.batch_prepare_for_extract(self.executor, images, blocks_list)
        for img_idx, (block_images, prompts, params, indices) in enumerate(prepared_inputs):
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_params.extend(params)
            all_indices.extend([(img_idx, idx) for idx in indices])
        outputs = self.client.batch_predict(all_images, all_prompts, all_params, priority)
        for (img_idx, idx), output in zip(all_indices, outputs):
            blocks_list[img_idx][idx].content = _extract_text_from_structured_output(output)
        blocks_list = self.helper.batch_post_process(self.executor, blocks_list)
        return [blocks[0].content if blocks else None for blocks in blocks_list]

    async def aio_content_extract(
        self,
        image: Image.Image,
        type: str = "text",
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> str | None:
        blocks = [ContentBlock(type, [0.0, 0.0, 1.0, 1.0])]
        block_images, prompts, params, _ = await self.helper.aio_prepare_for_extract(self.executor, image, blocks)
        if not (block_images and prompts and params):
            return None
        if semaphore is None:
            output = await self.client.aio_predict(block_images[0], prompts[0], params[0], priority)
        else:
            async with semaphore:
                output = await self.client.aio_predict(block_images[0], prompts[0], params[0], priority)
        blocks[0].content = _extract_text_from_structured_output(output)
        blocks = await self.helper.aio_post_process(self.executor, blocks)
        return blocks[0].content if blocks else None

    async def aio_batch_content_extract(
        self,
        images: list[Image.Image],
        types: Sequence[str] | str = "text",
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[str | None]:
        if isinstance(types, str):
            types = [types] * len(images)
        if len(types) != len(images):
            raise Exception("Length of types must match length of images")
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        blocks_list = [[ContentBlock(type, [0.0, 0.0, 1.0, 1.0])] for type in types]
        all_images: list[Image.Image | bytes] = []
        all_prompts: list[str] = []
        all_params: list[SamplingParams | None] = []
        all_indices: list[tuple[int, int]] = []
        prepared_inputs = await gather_tasks(
            tasks=[self.helper.aio_prepare_for_extract(self.executor, *args) for args in zip(images, blocks_list)],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Extract Preparation",
        )
        for img_idx, (block_images, prompts, params, indices) in enumerate(prepared_inputs):
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_params.extend(params)
            all_indices.extend([(img_idx, idx) for idx in indices])
        outputs = await self.client.aio_batch_predict(
            all_images,
            all_prompts,
            all_params,
            priority,
            semaphore=semaphore,
            use_tqdm=self.use_tqdm,
            tqdm_desc="Extraction",
        )
        for (img_idx, idx), output in zip(all_indices, outputs):
            blocks_list[img_idx][idx].content = _extract_text_from_structured_output(output)
        blocks_list = await gather_tasks(
            tasks=[self.helper.aio_post_process(self.executor, blocks) for blocks in blocks_list],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Post Processing",
        )
        return [blocks[0].content if blocks else None for blocks in blocks_list]

    def two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
    ) -> list[ContentBlock]:
        """
        两步文档提取: (1) 检测布局, (2) 从块中提取文本。

        参数:
            image: 输入的PIL图像
            priority: 请求调度的可选优先级

        返回:
            包含提取文本内容的ContentBlock对象列表
        """
        logger.info("开始两步提取(同步)")
        blocks = self.layout_detect(image, priority)
        logger.debug(f"第1步完成: 检测到{len(blocks)}个块")

        block_images, prompts, params, indices = self.helper.prepare_for_extract(image, blocks)
        logger.debug(f"第2步开始: 从{len(block_images)}个块中提取内容")

        outputs = self.client.batch_predict(block_images, prompts, params, priority)
        for idx, output in zip(indices, outputs):
            blocks[idx].content = _extract_text_from_structured_output(output)

        blocks = self.helper.post_process(blocks)
        logger.info(f"两步提取完成: {len(blocks)}个包含内容的块")
        return blocks

    async def aio_two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        """
        带错误处理的两步提取。
        如果发生错误，返回空列表而不是抛出异常。
        """
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        try:
            blocks = await self.aio_layout_detect(image, priority, semaphore)
            block_images, prompts, params, indices = await self.helper.aio_prepare_for_extract(self.executor, image, blocks)
            outputs = await self.client.aio_batch_predict(block_images, prompts, params, priority, semaphore=semaphore)
            logger.info(f'outputs=========>{outputs}')
            for idx, output in zip(indices, outputs):
                blocks[idx].content = _extract_text_from_structured_output(output)
            return await self.helper.aio_post_process(self.executor, blocks)
        except Exception as e:
            logger.exception(f"两步提取过程中发生错误: {e}")
            logger.warning("由于提取失败，返回空结果作为占位符")
            return []

    # batch_two_step_extract and aio_batch_two_step_extract are now inherited from BaseOCRClient (unified stepping mode)
