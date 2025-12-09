"""MinerU OCR client implementation using BaseOCRClient framework."""

from __future__ import annotations

import asyncio
import math
import re
from concurrent.futures import Executor
from typing import Literal, Sequence

from loguru import logger
from PIL import Image

from .base_client import BaseOCRClient, BaseOCRClientHelper
from .post_process import post_process
from .prompt_library import MINERU_DEFAULT_PROMPTS
from .structs import BLOCK_TYPES, ContentBlock
from .vlm_client import DEFAULT_SYSTEM_PROMPT, SamplingParams, new_vlm_client
from .vlm_client.utils import gather_tasks, get_png_bytes, get_rgb_image

_LAYOUT_RE = r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$"


class MinerUSamplingParams(SamplingParams):
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


DEFAULT_MINERU_SAMPLING_PARAMS: dict[str, SamplingParams] = {
    "table": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.005),
    "equation": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[default]": MinerUSamplingParams(presence_penalty=1.0, frequency_penalty=0.05),
    "[layout]": MinerUSamplingParams(),
}

ANGLE_MAPPING: dict[str, Literal[0, 90, 180, 270]] = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}


def _convert_bbox(bbox: Sequence[int] | Sequence[str]) -> list[float] | None:
    bbox = tuple(map(int, bbox))
    if any(coord < 0 or coord > 1000 for coord in bbox):
        logger.debug(f"Invalid bbox (out-of-range): {bbox}")
        return None
    x1, y1, x2, y2 = bbox
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    if x1 == x2 or y1 == y2:
        logger.debug(f"Invalid bbox (zero size): {bbox}")
        return None
    return list(map(lambda num: num / 1000.0, (x1, y1, x2, y2)))


def _parse_angle(tail: str) -> Literal[None, 0, 90, 180, 270]:
    for token, angle in ANGLE_MAPPING.items():
        if token in tail:
            return angle
    return None


class MinerUClientHelper(BaseOCRClientHelper):
    """Helper implementing MinerU-specific parsing and extraction logic."""

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

    def parse_layout_output(self, output: str) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []
        for line_num, raw_line in enumerate(output.splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(_LAYOUT_RE, line)
            if not match:
                logger.debug(f"Line {line_num} does not match MinerU layout format: {line}")
                continue
            x1, y1, x2, y2, ref_type, tail = match.groups()
            bbox = _convert_bbox((x1, y1, x2, y2))
            if bbox is None:
                continue
            ref_type = ref_type.lower()
            if ref_type not in BLOCK_TYPES:
                logger.debug(f"Unknown block type '{ref_type}' on line {line_num}")
                continue
            angle = _parse_angle(tail)
            blocks.append(ContentBlock(ref_type, bbox, angle=angle))
        logger.info(f"MinerU布局解析完成: 共{len(blocks)}个块")
        return blocks

    def prepare_for_extract(
        self,
        image: Image.Image,
        blocks: list[ContentBlock],
    ) -> tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]:
        image = get_rgb_image(image)
        width, height = image.size
        block_images: list[Image.Image | bytes] = []
        prompts: list[str] = []
        sampling_params: list[SamplingParams | None] = []
        indices: list[int] = []

        for idx, block in enumerate(blocks):
            if block.type in ("image", "list", "equation_block"):
                logger.debug(f"跳过块{idx}，类型={block.type}")
                continue

            x1, y1, x2, y2 = block.bbox
            scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            block_image = image.crop(scaled_bbox)

            if block.angle in [90, 180, 270]:
                block_image = block_image.rotate(block.angle, expand=True)
                logger.debug(f"块{idx}: 旋转{block.angle}度")

            block_image = self.resize_by_need(block_image)
            if self.backend == "http-client":
                block_image = get_png_bytes(block_image)

            block_images.append(block_image)
            prompt = self.prompts.get(block.type) or self.prompts["[default]"]
            prompts.append(prompt)
            params = self.sampling_params.get(block.type) or self.sampling_params.get("[default]")
            sampling_params.append(params)
            indices.append(idx)

        logger.info(f"MinerU内容提取准备完成: {len(block_images)}/{len(blocks)} 块")
        return block_images, prompts, sampling_params, indices

    def post_process(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        return post_process(
            blocks,
            handle_equation_block=self.handle_equation_block,
            abandon_list=self.abandon_list,
            abandon_paratext=self.abandon_paratext,
            debug=self.debug,
        )


class MinerUClient(BaseOCRClient):
    """BaseOCRClient-compatible MinerU OCR client."""

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
        prompts: dict[str, str] = MINERU_DEFAULT_PROMPTS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: dict[str, SamplingParams] = DEFAULT_MINERU_SAMPLING_PARAMS,
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
        self.helper = MinerUClientHelper(
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
        self.backend = backend
        self.prompts = prompts
        self.sampling_params = sampling_params
        self.incremental_priority = incremental_priority
        self.max_concurrency = max_concurrency
        self.executor = executor
        self.use_tqdm = use_tqdm
        self.debug = debug
        self.client_name = "mineru"

    def layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
    ) -> list[ContentBlock]:
        layout_image = self.helper.prepare_for_layout(image)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        output = self.client.predict(layout_image, prompt, params, priority)
        return self.helper.parse_layout_output(output)

    def batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
    ) -> list[list[ContentBlock]]:
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        layout_images = self.helper.batch_prepare_for_layout(self.executor, images)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        outputs = self.client.batch_predict(layout_images, prompt, params, priority)
        return self.helper.batch_parse_layout_output(self.executor, outputs)

    async def aio_layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[ContentBlock]:
        layout_image = await self.helper.aio_prepare_for_layout(self.executor, image)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        if semaphore is None:
            output = await self.client.aio_predict(layout_image, prompt, params, priority)
        else:
            async with semaphore:
                output = await self.client.aio_predict(layout_image, prompt, params, priority)
        return await self.helper.aio_parse_layout_output(self.executor, output)

    async def aio_batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[list[ContentBlock]]:
        if priority is None and self.incremental_priority:
            priority = list(range(len(images)))
        semaphore = semaphore or asyncio.Semaphore(self.max_concurrency)
        layout_images = await gather_tasks(
            tasks=[self.helper.aio_prepare_for_layout(self.executor, im) for im in images],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Layout Preparation",
        )
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        params = self.sampling_params.get("[layout]") or self.sampling_params.get("[default]")
        outputs = await self.client.aio_batch_predict(
            layout_images,
            prompt,
            params,
            priority,
            semaphore=semaphore,
            use_tqdm=self.use_tqdm,
            tqdm_desc="Layout Detection",
        )
        return await gather_tasks(
            tasks=[self.helper.aio_parse_layout_output(self.executor, out) for out in outputs],
            use_tqdm=self.use_tqdm,
            tqdm_desc="Layout Output Parsing",
        )

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
        blocks[0].content = output
        blocks = self.helper.post_process(blocks)
        return blocks[0].content if blocks else None

    def _extract_text(self, output: str) -> str:
        return output or ""

    def two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
    ) -> list[ContentBlock]:
        logger.info("MinerU两步提取开始(同步)")
        blocks = self.layout_detect(image, priority)
        block_images, prompts, params, indices = self.helper.prepare_for_extract(image, blocks)
        logger.info(f"MinerU两步提取: 准备 {len(block_images)} 个块")
        outputs = self.client.batch_predict(block_images, prompts, params, priority)
        for idx, output in zip(indices, outputs):
            blocks[idx].content = output
        blocks = self.helper.post_process(blocks)
        logger.info(f"MinerU两步提取完成: {len(blocks)} 个块")
        return blocks

    def batch_content_extract(
        self,
        images: list[Image.Image],
        types: Sequence[str] | str = "text",
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str | None]:
        if isinstance(types, str):
            types = [types] * len(images)
        if len(types) != len(images):
            raise ValueError("Length of types must match length of images")
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
            blocks_list[img_idx][idx].content = output
        blocks_list = self.helper.batch_post_process(self.executor, blocks_list)
        return [blocks[0].content if blocks else None for blocks in blocks_list]
