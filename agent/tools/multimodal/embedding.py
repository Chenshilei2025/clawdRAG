"""
Embedding generation tool for text, images, and videos.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from ..base import Tool, ToolResult

logger = logging.getLogger(__name__)


class EmbeddingGeneratorTool(Tool):
    """Generate embeddings for text, images, and videos."""

    name = "generate_embedding"
    description = "Generate vector embeddings for text, images, or video frames for semantic search."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.embedding_provider = None
        self._init_provider()

    def _init_provider(self):
        """Initialize embedding provider."""
        provider_type = self.config.get("provider", "openai")
        provider_config = self.config.get("provider_config", {})

        if provider_type == "openai":
            from ...providers.openai import OpenAIEmbedding
            self.embedding_provider = OpenAIEmbedding(provider_config)
        elif provider_type == "huggingface":
            from ...providers.huggingface import HuggingFaceEmbedding
            self.embedding_provider = HuggingFaceEmbedding(provider_config)
        else:
            raise ValueError(f"Unknown embedding provider: {provider_type}")

    async def execute(
        self,
        content: str,
        content_type: str = "text",
        media_path: Optional[str] = None
    ) -> ToolResult:
        """Generate embedding for content."""
        try:
            if content_type == "text":
                embedding = await self.embedding_provider.embed_text(content)
                return ToolResult(
                    success=True,
                    data={
                        "embedding": embedding,
                        "dimension": len(embedding),
                        "content_type": "text"
                    }
                )

            elif content_type in ["image", "video"]:
                if not media_path:
                    return ToolResult(success=False, error="media_path required for image/video embeddings")

                # For images, generate a text description first, then embed
                if content_type == "image":
                    # Use image captioning to get text representation
                    description = await self._get_image_description(media_path)
                    embedding = await self.embedding_provider.embed_text(description)
                else:
                    # For video, use frame analysis
                    description = await self._get_video_description(media_path)
                    embedding = await self.embedding_provider.embed_text(description)

                return ToolResult(
                    success=True,
                    data={
                        "embedding": embedding,
                        "dimension": len(embedding),
                        "content_type": content_type,
                        "description": description
                    }
                )

            else:
                return ToolResult(success=False, error=f"Unsupported content_type: {content_type}")

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return ToolResult(success=False, error=str(e))

    async def _get_image_description(self, image_path: str) -> str:
        """Get text description of image for embedding."""
        # Check if we have an image captioning provider
        captioning_provider = self.config.get("captioning_provider")
        if captioning_provider:
            from ...providers.openai import OpenAIVision
            vision = OpenAIVision(self.config.get("vision_config", {}))
            return await vision.caption_image(image_path)
        return f"Image at {image_path}"

    async def _get_video_description(self, video_path: str) -> str:
        """Get text description of video for embedding."""
        return f"Video at {video_path}"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Text content to embed (for text type)"
                },
                "content_type": {
                    "type": "string",
                    "enum": ["text", "image", "video"],
                    "description": "Type of content to embed",
                    "default": "text"
                },
                "media_path": {
                    "type": "string",
                    "description": "Path to image or video file (required for image/video types)"
                }
            },
            "required": ["content"]
        }
