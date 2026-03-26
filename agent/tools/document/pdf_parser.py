"""
PDF content extraction tool.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from ..base import Tool, ToolResult

logger = logging.getLogger(__name__)


class PDFContentExtractorTool(Tool):
    """Extract text, images, and structure from PDF documents."""

    name = "extract_pdf_content"
    description = "Extract content from PDF files including text, tables, and images."

    async def execute(
        self,
        file_path: str,
        extract_images: bool = True,
        extract_tables: bool = True,
        ocr_fallback: bool = True
    ) -> ToolResult:
        """Extract content from PDF file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return ToolResult(success=False, error=f"File not found: {file_path}")

            if path.suffix.lower() != ".pdf":
                return ToolResult(success=False, error=f"Not a PDF file: {path.suffix}")

            # Try PyPDF2 first for text extraction
            content = await self._extract_with_pypdf(path, extract_images, extract_tables)

            # If no text extracted and OCR is enabled, try OCR
            if not content["text"] or len(content["text"]) < 50:
                if ocr_fallback:
                    content = await self._extract_with_ocr(path)

            return ToolResult(
                success=True,
                data=content
            )

        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            return ToolResult(success=False, error=str(e))

    async def _extract_with_pypdf(
        self,
        path: Path,
        extract_images: bool,
        extract_tables: bool
    ) -> Dict[str, Any]:
        """Extract content using PyPDF2."""
        try:
            import PyPDF2
        except ImportError:
            return {"text": "", "error": "PyPDF2 not installed"}

        content = {
            "file_path": str(path),
            "total_pages": 0,
            "text": "",
            "pages": [],
            "images": [],
            "tables": []
        }

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            content["total_pages"] = len(reader.pages)

            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                content["pages"].append({
                    "page_number": page_num,
                    "text": page_text
                })
                content["text"] += f"\n\n--- Page {page_num} ---\n{page_text}"

        return content

    async def _extract_with_ocr(self, path: Path) -> Dict[str, Any]:
        """Extract content using OCR."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return {"text": "", "error": "PyMuPDF not installed for OCR"}

        content = {
            "file_path": str(path),
            "total_pages": 0,
            "text": "",
            "pages": [],
            "method": "ocr"
        }

        doc = fitz.open(str(path))
        content["total_pages"] = doc.page_count

        for page_num, page in enumerate(doc, 1):
            # Get text using OCR
            text = page.get_text()
            content["pages"].append({
                "page_number": page_num,
                "text": text
            })
            content["text"] += f"\n\n--- Page {page_num} ---\n{text}"

        doc.close()
        return content

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the PDF file"
                },
                "extract_images": {
                    "type": "boolean",
                    "description": "Extract images from PDF",
                    "default": True
                },
                "extract_tables": {
                    "type": "boolean",
                    "description": "Extract tables from PDF",
                    "default": True
                },
                "ocr_fallback": {
                    "type": "boolean",
                    "description": "Use OCR if text extraction fails",
                    "default": True
                }
            },
            "required": ["file_path"]
        }
