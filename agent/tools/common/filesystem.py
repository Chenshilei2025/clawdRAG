"""
Filesystem tools for reading and manipulating files.
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import mimetypes
import logging

from ..base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ReadFileTool(Tool):
    """Tool for reading file contents."""

    name = "read_file"
    description = "Read the contents of a text file. Supports .txt, .md, .py, .js, .json, .csv and other text formats."

    async def execute(self, file_path: str, encoding: str = "utf-8") -> ToolResult:
        """Read file contents."""
        try:
            path = Path(file_path)
            if not path.exists():
                return ToolResult(success=False, error=f"File not found: {file_path}")

            if not path.is_file():
                return ToolResult(success=False, error=f"Path is not a file: {file_path}")

            # Check if it's a text file
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and not mime_type.startswith("text/") and not path.suffix in {".json", ".md", ".yml", ".yaml", ".py", ".js", ".ts", ".jsx", ".tsx", ".css", ".html", ".sh", ".csv"}:
                return ToolResult(
                    success=False,
                    error=f"File appears to be binary. Use specialized tools for images, videos, or documents."
                )

            content = path.read_text(encoding=encoding)

            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "path": str(path),
                    "size": len(content),
                    "encoding": encoding
                }
            )

        except UnicodeDecodeError:
            return ToolResult(success=False, error=f"Failed to decode file with encoding: {encoding}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8"
                }
            },
            "required": ["file_path"]
        }


class ListDirTool(Tool):
    """Tool for listing directory contents."""

    name = "list_directory"
    description = "List files and directories in a given path."

    async def execute(self, directory: str, recursive: bool = False, pattern: str = "*") -> ToolResult:
        """List directory contents."""
        try:
            path = Path(directory)
            if not path.exists():
                return ToolResult(success=False, error=f"Directory not found: {directory}")

            if not path.is_dir():
                return ToolResult(success=False, error=f"Path is not a directory: {directory}")

            if recursive:
                files = [str(p) for p in path.rglob(pattern)]
            else:
                files = [str(p) for p in path.glob(pattern)]

            # Separate files and directories
            items = []
            for f in files:
                p = Path(f)
                items.append({
                    "name": p.name,
                    "path": str(p),
                    "type": "directory" if p.is_dir() else "file",
                    "size": p.stat().st_size if p.is_file() else None
                })

            return ToolResult(
                success=True,
                data={
                    "directory": directory,
                    "items": items,
                    "count": len(items)
                }
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path to the directory to list"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List recursively (default: false)",
                    "default": False
                },
                "pattern": {
                    "type": "string",
                    "description": "File pattern to match (default: *)",
                    "default": "*"
                }
            },
            "required": ["directory"]
        }


class WriteFileTool(Tool):
    """Tool for writing content to files."""

    name = "write_file"
    description = "Write content to a file. Creates the file if it doesn't exist."

    async def execute(self, file_path: str, content: str, encoding: str = "utf-8") -> ToolResult:
        """Write content to file."""
        try:
            path = Path(file_path)

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            path.write_text(content, encoding=encoding)

            return ToolResult(
                success=True,
                data={
                    "path": str(path),
                    "size": len(content)
                }
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8"
                }
            },
            "required": ["file_path", "content"]
        }
