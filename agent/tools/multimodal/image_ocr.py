"""
OCR tool for extracting text from images.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from ..base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ImageOCRTool(Tool):
    """Extract text from images using OCR."""

    name = "extract_text_from_image"
    description = "Extract all visible text from an image using OCR. Supports handwritten and printed text."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.ocr_provider = None
        self._init_provider()

    def _init_provider(self):
        """Initialize OCR provider."""
        provider_type = self.config.get("provider", "openai")
        provider_config = self.config.get("provider_config", {})

        if provider_type == "openai":
            from ...providers.openai import OpenAIVision
            self.ocr_provider = OpenAIVision(provider_config)
        elif provider_type == "tesseract":
            # Could add Tesseract support here
            pass
        else:
            logger.warning(f"Unknown OCR provider: {provider_type}, using default")

    async def execute(self, image_path: str, language: str = "auto") -> ToolResult:
        """Extract text from an image."""
        try:
            path = Path(image_path)
            if not path.exists():
                return ToolResult(success=False, error=f"Image not found: {image_path}")

            if not path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
                return ToolResult(success=False, error=f"Unsupported image format: {path.suffix}")

            if self.ocr_provider:
                extracted_text = await self.ocr_provider.extract_text(image_path)
            else:
                # Fallback: try to use pytesseract if available
                extracted_text = await self._fallback_ocr(image_path)

            return ToolResult(
                success=True,
                data={
                    "text": extracted_text,
                    "image_path": image_path,
                    "language": language
                }
            )

        except Exception as e:
            logger.error(f"Error in OCR: {e}")
            return ToolResult(success=False, error=str(e))

    async def _fallback_ocr(self, image_path: str) -> str:
        """Fallback OCR using pytesseract."""
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except ImportError:
            return "OCR not available. Please install pytesseract or configure an OCR provider."

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file"
                },
                "language": {
                    "type": "string",
                    "description": "Language for OCR (default: auto-detect)",
                    "default": "auto"
                }
            },
            "required": ["image_path"]
        }
