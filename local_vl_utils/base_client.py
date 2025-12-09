"""Base OCR client abstraction for multi-model support.

This module provides abstract base classes and protocols for implementing
different OCR model clients (Qwen, DeepSeek, Hunyuan, etc).

Architecture:
    BaseOCRClient (ABC)
    ├── QwenClient (existing)
    ├── DeepSeekClient (new)
    └── HunyuanClient (new)

All clients share common interfaces but differ in:
- Prompt templates
- Output parsing logic
- Sampling parameters
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import Executor
from typing import Literal, Sequence

from PIL import Image
from loguru import logger

from .structs import ContentBlock
from .vlm_client import SamplingParams


class BaseOCRClientHelper(ABC):
    """Abstract base class for OCR client helpers.

    Helpers handle model-specific logic:
    - Image preprocessing
    - Prompt generation
    - Output parsing
    - Post-processing
    """

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
        self.backend = backend
        self.prompts = prompts
        self.sampling_params = sampling_params
        self.layout_image_size = layout_image_size
        self.min_image_edge = min_image_edge
        self.max_image_edge_ratio = max_image_edge_ratio
        self.handle_equation_block = handle_equation_block
        self.abandon_list = abandon_list
        self.abandon_paratext = abandon_paratext
        self.debug = debug

    @abstractmethod
    def parse_layout_output(self, output: str) -> list[ContentBlock]:
        """Parse layout detection model output to ContentBlock objects.

        Args:
            output: Raw string output from layout detection

        Returns:
            List of parsed ContentBlock objects

        Note:
            Each model has different output formats:
            - Qwen: JSON with bbox_2d/label
            - DeepSeek: JSON similar to Qwen
            - Hunyuan: XML with <ref> and <quad> tags
        """
        pass

    def parse_layout_output_as_json(self, output: str) -> str:
        """Parse layout output and return as JSON string.

        This is a convenience method that combines parsing and JSON conversion.

        Args:
            output: Raw string output from layout detection

        Returns:
            JSON string representation of blocks
        """
        from .format_converter import blocks_to_json

        blocks = self.parse_layout_output(output)
        return blocks_to_json(blocks, pretty=False)

    def resize_by_need(self, image: Image.Image) -> Image.Image:
        """Resize image by need to meet aspect ratio and min size requirements.

        This method is identical across all three clients (Qwen, DeepSeek, Hunyuan).
        It handles:
        1. Extreme aspect ratios by adding padding
        2. Too-small images by scaling up

        Args:
            image: Input PIL Image

        Returns:
            Resized image meeting size constraints
        """
        import math

        original_size = image.size
        edge_ratio = max(image.size) / min(image.size)

        # Handle extreme aspect ratios by adding padding
        if edge_ratio > self.max_image_edge_ratio:
            width, height = image.size
            if width > height:
                new_w, new_h = width, math.ceil(width / self.max_image_edge_ratio)
            else:  # width < height
                new_w, new_h = math.ceil(height / self.max_image_edge_ratio), height
            new_image = Image.new(image.mode, (new_w, new_h), (255, 255, 255))
            new_image.paste(image, (int((new_w - width) / 2), int((new_h - height) / 2)))
            image = new_image
            logger.debug(f"图像填充: {original_size} -> {image.size} (宽高比: {edge_ratio:.2f})")

        # Scale up if image is too small
        if min(image.size) < self.min_image_edge:
            scale = self.min_image_edge / min(image.size)
            new_w, new_h = round(image.width * scale), round(image.height * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
            logger.debug(f"图像放大: {original_size} -> {image.size} (缩放比例: {scale:.2f})")

        return image

    def prepare_for_layout(self, image: Image.Image) -> Image.Image | bytes:
        """Prepare image for layout detection by resizing and converting format.

        This method is identical across all three clients (Qwen, DeepSeek, Hunyuan).
        Note: Qwen uses get_rgb_image() before resizing, which should be handled
        in subclass if needed.

        Args:
            image: Input PIL Image

        Returns:
            Prepared image (PIL Image or PNG bytes, depending on backend)
        """
        from .vlm_client.utils import get_png_bytes, get_rgb_image

        original_size = image.size
        image = get_rgb_image(image)
        image = image.resize(self.layout_image_size, Image.Resampling.BICUBIC)
        logger.debug(f"准备布局图像: {original_size} -> {self.layout_image_size}")

        if self.backend == "http-client":
            return get_png_bytes(image)
        return image

    @abstractmethod
    def prepare_for_extract(
        self,
        image: Image.Image,
        blocks: list[ContentBlock],
    ) -> tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]:
        """Prepare content blocks for text extraction by cropping and processing regions.

        Args:
            image: Original PIL Image
            blocks: List of ContentBlock objects from layout detection

        Returns:
            Tuple of (block_images, prompts, sampling_params, indices):
                - block_images: Cropped and processed images for each block
                - prompts: Extraction prompts for each block type
                - sampling_params: Sampling parameters for each block type
                - indices: Original indices of blocks to extract
        """
        pass

    @abstractmethod
    def post_process(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        """Post-process extracted content blocks (clean text, merge blocks, etc).

        Args:
            blocks: List of ContentBlock objects with extracted content

        Returns:
            Post-processed list of ContentBlock objects
        """
        pass

    # Batch processing methods (can be overridden for optimization)
    def batch_prepare_for_layout(
        self,
        executor: Executor | None,
        images: list[Image.Image],
    ) -> list[Image.Image | bytes]:
        """Batch prepare multiple images for layout detection."""
        logger.debug(f"批量准备{len(images)}个图像用于布局检测")
        if executor is None:
            return [self.prepare_for_layout(im) for im in images]
        return list(executor.map(self.prepare_for_layout, images))

    def batch_parse_layout_output(
        self,
        executor: Executor | None,
        outputs: list[str],
    ) -> list[list[ContentBlock]]:
        """Batch parse multiple layout detection outputs."""
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
        """Batch prepare content blocks extraction from multiple images."""
        logger.debug(f"批量准备{len(images)}个图像用于内容提取")
        if executor is None:
            return [self.prepare_for_extract(im, bls) for im, bls in zip(images, blocks_list)]
        return list(executor.map(self.prepare_for_extract, images, blocks_list))

    def batch_post_process(
        self,
        executor: Executor | None,
        blocks_list: list[list[ContentBlock]],
    ) -> list[list[ContentBlock]]:
        """Batch post-process content blocks for multiple images."""
        logger.debug(f"批量后处理{len(blocks_list)}个块列表")
        if executor is None:
            return [self.post_process(blocks) for blocks in blocks_list]
        return list(executor.map(self.post_process, blocks_list))

    # Async methods
    async def aio_prepare_for_layout(
        self,
        executor: Executor | None,
        image: Image.Image,
    ) -> Image.Image | bytes:
        """Async version of prepare_for_layout."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.prepare_for_layout, image)

    async def aio_parse_layout_output(
        self,
        executor: Executor | None,
        output: str,
    ) -> list[ContentBlock]:
        """Async version of parse_layout_output."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.parse_layout_output, output)

    async def aio_prepare_for_extract(
        self,
        executor: Executor | None,
        image: Image.Image,
        blocks: list[ContentBlock],
    ) -> tuple[list[Image.Image | bytes], list[str], list[SamplingParams | None], list[int]]:
        """Async version of prepare_for_extract."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.prepare_for_extract, image, blocks)

    async def aio_post_process(
        self,
        executor: Executor | None,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        """Async version of post_process."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.post_process, blocks)


class BaseOCRClient(ABC):
    """Abstract base class for OCR clients.

    All OCR clients must implement these core methods:
    - layout_detect: Detect layout elements (text blocks, tables, images, etc.)
    - two_step_extract: Two-step document extraction (layout + text)
    - batch_two_step_extract: Batch version of two-step extraction
    """

    def __init__(
        self,
        backend: str,
        model_name: str | None,
        helper: BaseOCRClientHelper,
        **kwargs,
    ):
        self.backend = backend
        self.model_name = model_name
        self.helper = helper
        self.client = None  # Will be set by subclass

    @abstractmethod
    def layout_detect(
        self,
        image: Image.Image,
        priority: int | None = None,
    ) -> list[ContentBlock]:
        """Detect layout elements in a single image.

        Args:
            image: Input PIL Image
            priority: Optional priority for request scheduling

        Returns:
            List of detected ContentBlock objects with bboxes and types
        """
        pass

    def layout_detect_as_json(
        self,
        image: Image.Image,
        priority: int | None = None,
        include_metadata: bool = True
    ) -> str:
        """Detect layout and return result as JSON string.

        Args:
            image: Input PIL Image
            priority: Optional priority for request scheduling
            include_metadata: Whether to include model metadata

        Returns:
            JSON string representation of detected blocks
        """
        from .format_converter import blocks_to_standard_json

        blocks = self.layout_detect(image, priority)
        model_type = getattr(self, 'client_name', self.__class__.__name__.lower().replace('client', ''))
        result = blocks_to_standard_json(blocks, model_type, include_metadata=include_metadata)

        import json
        return json.dumps(result, ensure_ascii=False)

    @abstractmethod
    def two_step_extract(
        self,
        image: Image.Image,
        priority: int | None = None,
    ) -> list[ContentBlock]:
        """Two-step document extraction: (1) detect layout, (2) extract text from blocks.

        Args:
            image: Input PIL Image
            priority: Optional priority for request scheduling

        Returns:
            List of ContentBlock objects with extracted text content
        """
        pass

    @abstractmethod
    def batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
    ) -> list[list[ContentBlock]]:
        """Batch layout detection for multiple images.

        Args:
            images: List of input PIL Images
            priority: Optional priority for request scheduling

        Returns:
            List of lists of ContentBlock objects, one list per image
        """
        pass

    @abstractmethod
    def _extract_text(self, output: str) -> str:
        """Extract text content from model output.

        Different models have different output formats:
        - Qwen/DeepSeek: Use structured JSON extraction
        - Hunyuan: Direct output

        Args:
            output: Raw model output string

        Returns:
            Extracted text content
        """
        pass

    @abstractmethod
    def batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
    ) -> list[list[ContentBlock]]:
        """Batch layout detection for multiple images.

        Args:
            images: List of input PIL Images
            priority: Optional priority for request scheduling

        Returns:
            List of lists of ContentBlock objects, one list per image
        """
        pass

    @abstractmethod
    async def aio_batch_layout_detect(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
    ) -> list[list[ContentBlock]]:
        """Async batch layout detection for multiple images.

        Args:
            images: List of input PIL Images
            priority: Optional priority for request scheduling

        Returns:
            List of lists of ContentBlock objects, one list per image
        """
        pass

    def batch_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
    ) -> list[list[ContentBlock]]:
        """Unified batch two-step extraction using Stepping mode.

        This method implements the stepping pattern:
        1. Batch layout detection for ALL images
        2. Prepare ALL blocks from ALL images
        3. Batch content extraction for ALL blocks in one request
        4. Distribute results back to blocks
        5. Batch post-processing

        Benefits:
        - Reduces HTTP requests from N+M*N to 2
        - Better vLLM batch processing efficiency
        - Easier to use different models for layout vs content

        Args:
            images: List of input PIL Images
            priority: Optional priority for request scheduling

        Returns:
            List of lists of ContentBlock objects with extracted content
        """
        logger.info(f"批量两步提取开始 (Stepping模式)，共{len(images)}个图像")

        # Normalize priority to align with the number of images
        image_priority = priority
        if image_priority is None and hasattr(self, 'incremental_priority') and self.incremental_priority:
            image_priority = list(range(len(images)))
        if isinstance(image_priority, Sequence):
            image_priority = list(image_priority)
        else:
            image_priority = [image_priority] * len(images)
        if len(image_priority) != len(images):
            raise ValueError("Length of priority and images must match.")

        # Step 1: Batch layout detection for ALL images
        blocks_list = self.batch_layout_detect(images, image_priority)
        total_blocks = sum(len(b) for b in blocks_list)
        logger.info(f"Step 1 完成: 检测到 {total_blocks} 个块")

        # Step 2: Prepare ALL blocks from ALL images for extraction
        all_images: list[Image.Image | bytes] = []
        all_prompts: list[str] = []
        all_params: list[SamplingParams | None] = []
        all_indices: list[tuple[int, int]] = []  # (image_idx, block_idx)

        executor = getattr(self, 'executor', None)
        prepared_inputs = self.helper.batch_prepare_for_extract(executor, images, blocks_list)

        for img_idx, (block_images, prompts, params, indices) in enumerate(prepared_inputs):
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_params.extend(params)
            all_indices.extend([(img_idx, idx) for idx in indices])

        logger.info(f"Step 2 完成: 准备了 {len(all_images)} 个块用于提取")

        # Step 3: Batch content extraction for ALL blocks
        if all_images:  # Only if there are blocks to extract
            block_priority = [image_priority[img_idx] for img_idx, _ in all_indices]
            outputs = self.client.batch_predict(all_images, all_prompts, all_params, block_priority)
            logger.info(f"Step 3 完成: 提取了 {len(outputs)} 个块的内容")

            # Step 4: Distribute results back to blocks
            for (img_idx, idx), output in zip(all_indices, outputs):
                blocks_list[img_idx][idx].content = self._extract_text(output)
        else:
            logger.info("Step 3 跳过: 没有需要提取的块")

        # Step 5: Batch post-processing
        result = self.helper.batch_post_process(executor, blocks_list)
        logger.info(f"批量两步提取完成: 共 {sum(len(b) for b in result)} 个块")
        return result

    async def aio_batch_two_step_extract(
        self,
        images: list[Image.Image],
        priority: Sequence[int | None] | int | None = None,
    ) -> list[list[ContentBlock]]:
        """Unified async batch two-step extraction using Stepping mode.

        This method implements the async stepping pattern:
        1. Async batch layout detection for ALL images
        2. Prepare ALL blocks from ALL images
        3. Async batch content extraction for ALL blocks in one request
        4. Distribute results back to blocks
        5. Async batch post-processing

        Benefits:
        - Reduces HTTP requests from N+M*N to 2
        - Better vLLM batch processing efficiency
        - Easier to use different models for layout vs content

        Args:
            images: List of input PIL Images
            priority: Optional priority for request scheduling

        Returns:
            List of lists of ContentBlock objects with extracted content
        """
        logger.info(f"异步批量两步提取开始 (Stepping模式)，共{len(images)}个图像")

        # Normalize priority to align with the number of images
        image_priority = priority
        if image_priority is None and hasattr(self, 'incremental_priority') and self.incremental_priority:
            image_priority = list(range(len(images)))
        if isinstance(image_priority, Sequence):
            image_priority = list(image_priority)
        else:
            image_priority = [image_priority] * len(images)
        if len(image_priority) != len(images):
            raise ValueError("Length of priority and images must match.")

        # Step 1: Async batch layout detection for ALL images
        blocks_list = await self.aio_batch_layout_detect(images, image_priority)
        total_blocks = sum(len(b) for b in blocks_list)
        logger.info(f"Step 1 完成: 检测到 {total_blocks} 个块")

        # Step 2: Prepare ALL blocks from ALL images for extraction
        all_images: list[Image.Image | bytes] = []
        all_prompts: list[str] = []
        all_params: list[SamplingParams | None] = []
        all_indices: list[tuple[int, int]] = []  # (image_idx, block_idx)

        executor = getattr(self, 'executor', None)

        # Use async preparation if available
        if hasattr(self.helper, 'aio_batch_prepare_for_extract'):
            prepared_inputs = await self.helper.aio_batch_prepare_for_extract(executor, images, blocks_list)
        else:
            # Fallback to sync version
            prepared_inputs = self.helper.batch_prepare_for_extract(executor, images, blocks_list)

        for img_idx, (block_images, prompts, params, indices) in enumerate(prepared_inputs):
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_params.extend(params)
            all_indices.extend([(img_idx, idx) for idx in indices])

        logger.info(f"Step 2 完成: 准备了 {len(all_images)} 个块用于提取")

        # Step 3: Async batch content extraction for ALL blocks
        if all_images:  # Only if there are blocks to extract
            # Use async predict if available
            block_priority = [image_priority[img_idx] for img_idx, _ in all_indices]
            if hasattr(self.client, 'aio_batch_predict'):
                outputs = await self.client.aio_batch_predict(all_images, all_prompts, all_params, block_priority)
            else:
                # Fallback to sync version
                outputs = self.client.batch_predict(all_images, all_prompts, all_params, block_priority)
            logger.info(f"Step 3 完成: 提取了 {len(outputs)} 个块的内容")

            # Step 4: Distribute results back to blocks
            for (img_idx, idx), output in zip(all_indices, outputs):
                blocks_list[img_idx][idx].content = self._extract_text(output)
        else:
            logger.info("Step 3 跳过: 没有需要提取的块")

        # Step 5: Async batch post-processing
        if hasattr(self.helper, 'aio_batch_post_process'):
            result = await self.helper.aio_batch_post_process(executor, blocks_list)
        else:
            # Fallback to sync version
            result = self.helper.batch_post_process(executor, blocks_list)

        logger.info(f"异步批量两步提取完成: 共 {sum(len(b) for b in result)} 个块")
        return result


# Model-specific default prompts and parameters will be defined in each client implementation
