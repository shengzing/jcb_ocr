"""Recognition dimensions and image saving configuration."""

from __future__ import annotations

import os
from typing import Literal


RecognitionDimension = Literal["清晰度", "完整性", "正确性", "复杂度", "难度", "特殊性"]
ImageType = Literal["模糊图片", "残缺图片", "错误识别", "复杂图片", "困难图片", "特殊图片"]


class RecognitionConfig:
    """Manages recognition dimensions and image saving configuration."""

    def __init__(self) -> None:
        dimensions_str = os.getenv("RECOGNITION_DIMENSIONS", "清晰度,完整性,正确性,复杂度,难度,特殊性")
        self.dimensions = [d.strip() for d in dimensions_str.split(",")]

        save_types_str = os.getenv("SAVE_IMAGE_TYPES", "模糊图片,残缺图片,错误识别")
        self.save_image_types = set(t.strip() for t in save_types_str.split(","))

    def should_save_image(self, image_type: str) -> bool:
        """Check if image type should be saved."""
        return image_type in self.save_image_types

    def get_dimensions(self) -> list[str]:
        """Get recognition dimensions."""
        return self.dimensions


_config: RecognitionConfig | None = None


def get_recognition_config() -> RecognitionConfig:
    """Get singleton recognition config."""
    global _config
    if _config is None:
        _config = RecognitionConfig()
    return _config
