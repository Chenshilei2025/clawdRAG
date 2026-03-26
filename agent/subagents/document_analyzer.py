"""
Document analyzer subagent for processing PDF and PPT files.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from .base import SimpleSubagent

logger = logging.getLogger(__name__)


class DocumentAnalyzerSubagent(SimpleSubagent):
    """Subagent specialized in analyzing document files (PDF, PPT)."""

    def __init__(self, subagent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(subagent_id, config)
        self.task_description = """You are a document analysis specialist. Your task is to analyze documents (PDF, PPT) and extract key information, summaries, and insights.

Focus on:
1. Main topics and themes
2. Key findings or conclusions
3. Important data points, statistics, or metrics
4. Structure and organization
5. Action items or recommendations
6. Visual elements (charts, diagrams) and their significance"""

        self.tool_registry = None

    async def initialize(self):
        """Initialize with document processing tools."""
        await super().initialize()

        from ..tools.base import ToolRegistry
        from ..tools.document.pdf_parser import PDFContentExtractorTool
        from ..tools.document.ppt_parser import PPTContentExtractorTool
        from ..tools.multimodal.image_ocr import ImageOCRTool
        from ..tools.multimodal.vector_search import VectorIndexTool

        self.tool_registry = ToolRegistry()

        # Register document tools
        tool_config = self.config.get("tool_config", {})
        self.tool_registry.register(PDFContentExtractorTool(tool_config))
        self.tool_registry.register(PPTContentExtractorTool(tool_config))
        self.tool_registry.register(ImageOCRTool(tool_config))
        self.tool_registry.register(VectorIndexTool(tool_config))

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a document."""
        file_path = parameters.get("file_path", "")
        analysis_type = parameters.get("analysis_type", "comprehensive")
        index_content = parameters.get("index_content", True)

        path = Path(file_path)
        if not path.exists():
            return {
                "error": f"File not found: {file_path}",
                "subagent_id": self.subagent_id
            }

        # Extract content based on file type
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            extraction_result = await self.tool_registry.execute(
                "extract_pdf_content",
                file_path=str(path)
            )
        elif suffix in {".ppt", ".pptx"}:
            extraction_result = await self.tool_registry.execute(
                "extract_ppt_content",
                file_path=str(path)
            )
        else:
            return {
                "error": f"Unsupported document format: {suffix}",
                "subagent_id": self.subagent_id
            }

        if not extraction_result.success:
            return {
                "error": extraction_result.error,
                "subagent_id": self.subagent_id
            }

        document_data = extraction_result.data

        # Analyze with LLM
        analysis = await self._analyze_with_llm(document_data, analysis_type)

        # Index for search if requested
        indexed = False
        if index_content:
            full_text = document_data.get("text", "")
            if full_text:
                index_result = await self.tool_registry.execute(
                    "index_document",
                    content=full_text,
                    metadata={
                        "source": str(path),
                        "type": "document",
                        "subagent_id": self.subagent_id
                    }
                )
                indexed = index_result.success

        return {
            "file_path": str(path),
            "file_type": suffix,
            "extraction": {
                "total_pages": document_data.get("total_pages", 0),
                "text_length": len(document_data.get("text", ""))
            },
            "analysis": analysis,
            "indexed": indexed,
            "subagent_id": self.subagent_id
        }

    async def _analyze_with_llm(
        self,
        document_data: Dict[str, Any],
        analysis_type: str
    ) -> Dict[str, Any]:
        """Analyze document content with LLM."""
        text = document_data.get("text", "")
        total_pages = document_data.get("total_pages", 0)

        # Truncate if too long
        max_chars = 12000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n... (content truncated for analysis)"

        prompt = f"""Analyze this document and provide insights:

Document Info:
- Total Pages: {total_pages}
- Content Length: {len(document_data.get('text', ''))} characters

Content:
{text}

Please provide:
1. Executive Summary (3-5 sentences)
2. Key Topics (list of main themes)
3. Important Findings (bullet points)
4. Structure Overview (how the document is organized)
5. Recommended Actions (if applicable)

Respond in JSON format."""

        from ...providers.base import Message
        messages = [
            Message(role="system", content=self.task_description),
            Message(role="user", content=prompt)
        ]

        response = await self.llm_provider.generate(
            messages,
            temperature=0.4,
            max_tokens=2000
        )

        # Try to parse as JSON
        import json
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "summary": "Analysis completed but response format was not JSON",
                "raw_response": response.content
            }
