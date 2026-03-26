"""
Image analyzer subagent for processing image files.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from .base import SimpleSubagent

logger = logging.getLogger(__name__)


class ImageAnalyzerSubagent(SimpleSubagent):
    """Subagent specialized in analyzing images."""

    def __init__(self, subagent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(subagent_id, config)
        self.task_description = """You are an image analysis specialist. Your task is to provide detailed analysis of images including:

1. Main subjects and objects
2. Actions and activities
3. Setting and context
4. Text content (if any)
5. Emotional tone
6. Technical aspects (composition, lighting, style)"""

        self.tool_registry = None

    async def initialize(self):
        """Initialize with image processing tools."""
        await super().initialize()

        from ..tools.base import ToolRegistry
        from ..tools.multimodal.image_captioning import ImageCaptioningTool
        from ..tools.multimodal.image_ocr import ImageOCRTool
        from ..tools.multimodal.embedding import EmbeddingGeneratorTool
        from ..tools.multimodal.vector_search import VectorIndexTool

        self.tool_registry = ToolRegistry()

        tool_config = self.config.get("tool_config", {})
        self.tool_registry.register(ImageCaptioningTool(tool_config))
        self.tool_registry.register(ImageOCRTool(tool_config))
        self.tool_registry.register(EmbeddingGeneratorTool(tool_config))
        self.tool_registry.register(VectorIndexTool(tool_config))

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze an image file."""
        file_path = parameters.get("file_path", "")
        extract_text = parameters.get("extract_text", True)
        index_content = parameters.get("index_content", True)
        analysis_detail = parameters.get("analysis_detail", "detailed")

        path = Path(file_path)
        if not path.exists():
            return {
                "error": f"Image not found: {file_path}",
                "subagent_id": self.subagent_id
            }

        # Get caption
        caption_result = await self.tool_registry.execute(
            "caption_image",
            image_path=str(path),
            detail_level=analysis_detail
        )

        if not caption_result.success:
            return {
                "error": f"Captioning failed: {caption_result.error}",
                "subagent_id": self.subagent_id
            }

        caption = caption_result.data.get("caption", "")

        # Extract text if requested
        text_content = None
        if extract_text:
            ocr_result = await self.tool_registry.execute(
                "extract_text_from_image",
                image_path=str(path)
            )
            if ocr_result.success:
                text_content = ocr_result.data.get("text", "")

        # Detailed analysis with LLM
        analysis = await self._analyze_with_llm(caption, text_content, parameters)

        # Index for search if requested
        indexed = False
        embedding = None
        if index_content:
            index_text = self._build_index_content(caption, text_content, analysis)
            if index_text:
                # Generate embedding
                embed_result = await self.tool_registry.execute(
                    "generate_embedding",
                    content=index_text,
                    content_type="text"
                )
                if embed_result.success:
                    embedding = embed_result.data.get("embedding")

                # Index in vector store
                index_result = await self.tool_registry.execute(
                    "index_document",
                    content=index_text,
                    metadata={
                        "source": str(path),
                        "type": "image",
                        "subagent_id": self.subagent_id,
                        "caption": caption[:500]
                    }
                )
                indexed = index_result.success

        return {
            "file_path": str(path),
            "caption": caption,
            "extracted_text": text_content,
            "analysis": analysis,
            "indexed": indexed,
            "embedding_dimension": len(embedding) if embedding else None,
            "subagent_id": self.subagent_id
        }

    async def _analyze_with_llm(
        self,
        caption: str,
        text_content: Optional[str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze image with LLM."""
        prompt_parts = ["Image Analysis Request\n"]

        prompt_parts.append(f"Image Description:\n{caption}\n")

        if text_content and text_content.strip():
            prompt_parts.append(f"Extracted Text:\n{text_content}\n")

        prompt_parts.append("""Please provide detailed analysis:
1. Primary Subjects (who/what is featured)
2. Setting/Location (where it takes place)
3. Actions/Activities (what is happening)
4. Mood/Atmosphere (emotional tone)
5. Notable Details (specific elements of interest)
6. Possible Context/Story (what might be going on)

Respond in JSON format.""")

        prompt = "\n".join(prompt_parts)

        from ...providers.base import Message
        messages = [
            Message(role="system", content=self.task_description),
            Message(role="user", content=prompt)
        ]

        response = await self.llm_provider.generate(
            messages,
            temperature=0.5,
            max_tokens=1500
        )

        import json
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "summary": "Analysis completed",
                "raw_response": response.content
            }

    def _build_index_content(
        self,
        caption: str,
        text_content: Optional[str],
        analysis: Dict[str, Any]
    ) -> str:
        """Build text content for indexing."""
        parts = [f"Image: {caption}"]

        if text_content and text_content.strip():
            parts.append(f"Text: {text_content}")

        if analysis:
            for key, value in analysis.items():
                if isinstance(value, str) and value:
                    parts.append(f"{key}: {value}")

        return "\n".join(parts)
