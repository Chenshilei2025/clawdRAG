"""
Image captioning tool for generating image descriptions.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from ..base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ImageCaptioningTool(Tool):
    """Generate descriptive captions for images."""

    name = "caption_image"
    description = "Generate a detailed caption describing the content of an image, including objects, people, actions, and context."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.captioning_provider = None
        self._init_provider()

    def _init_provider(self):
        """Initialize captioning provider."""
        provider_type = self.config.get("provider", "openai")
        provider_config = self.config.get("provider_config", {})

        if provider_type == "openai":
            from ...providers.openai import OpenAIVision
            self.captioning_provider = OpenAIVision(provider_config)
        elif provider_type == "huggingface":
            # Could add BLIP or other HF models here
            pass

    async def execute(
        self,
        image_path: str,
        detail_level: str = "detailed"
    ) -> ToolResult:
        """Generate caption for an image."""
        try:
            path = Path(image_path)
            if not path.exists():
                return ToolResult(success=False, error=f"Image not found: {image_path}")

            if not path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
                return ToolResult(success=False, error=f"Unsupported image format: {path.suffix}")

            if self.captioning_provider:
                caption = await self.captioning_provider.caption_image(image_path)
            else:
                caption = "Captioning provider not available."

            return ToolResult(
                success=True,
                data={
                    "caption": caption,
                    "image_path": image_path,
                    "detail_level": detail_level
                }
            )

        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return ToolResult(success=False, error=str(e))

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file"
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["brief", "detailed"],
                    "description": "Level of detail in the caption",
                    "default": "detailed"
                }
            },
            "required": ["image_path"]
        }
