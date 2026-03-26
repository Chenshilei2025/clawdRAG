"""
Subagent manager for dynamically spawning subagents.
"""
from typing import Dict, Any, Optional, List
import asyncio
import logging
from uuid import uuid4

from .base import Tool, ToolResult

logger = logging.getLogger(__name__)


class SubagentManagerTool(Tool):
    """Tool for spawning and managing subagents."""

    name = "spawn_subagent"
    description = "Spawn a specialized subagent to handle a specific task (document analysis, video analysis, etc.)."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.subagent_factory = None
        self.active_subagents: Dict[str, Any] = {}

    def set_subagent_factory(self, factory):
        """Set the subagent factory for creating subagents."""
        self.subagent_factory = factory

    async def execute(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> ToolResult:
        """Spawn a subagent for the specified task."""
        try:
            if not self.subagent_factory:
                return ToolResult(
                    success=False,
                    error="Subagent factory not configured"
                )

            # Generate subagent ID
            subagent_id = f"{task_type}_{uuid4().hex[:8]}"

            # Create subagent
            subagent = await self.subagent_factory.create_subagent(
                task_type=task_type,
                subagent_id=subagent_id,
                parameters=parameters
            )

            if not subagent:
                return ToolResult(
                    success=False,
                    error=f"Failed to create subagent for task type: {task_type}"
                )

            # Track subagent
            self.active_subagents[subagent_id] = subagent

            # Execute subagent task
            if timeout:
                result = await asyncio.wait_for(subagent.execute(parameters), timeout=timeout)
            else:
                result = await subagent.execute(parameters)

            # Clean up
            if subagent_id in self.active_subagents:
                del self.active_subagents[subagent_id]
            await subagent.cleanup()

            return ToolResult(
                success=True,
                data={
                    "subagent_id": subagent_id,
                    "task_type": task_type,
                    "result": result
                }
            )

        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Subagent task timed out after {timeout} seconds"
            )
        except Exception as e:
            logger.error(f"Error in subagent execution: {e}")
            return ToolResult(success=False, error=str(e))

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_type": {
                    "type": "string",
                    "enum": [
                        "query_rewriter",
                        "document_analyzer",
                        "video_analyzer",
                        "image_analyzer"
                    ],
                    "description": "Type of subagent to spawn"
                },
                "parameters": {
                    "type": "object",
                    "description": "Task-specific parameters for the subagent"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (optional)"
                }
            },
            "required": ["task_type", "parameters"]
        }


class SubagentFactory:
    """Factory for creating subagents."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._subagent_classes = {}
        self._register_default_subagents()

    def _register_default_subagents(self):
        """Register default subagent types."""
        from ..subagents.query_rewriter import QueryRewriterSubagent
        from ..subagents.document_analyzer import DocumentAnalyzerSubagent
        from ..subagents.video_analyzer import VideoAnalyzerSubagent
        from ..subagents.image_analyzer import ImageAnalyzerSubagent

        self._subagent_classes = {
            "query_rewriter": QueryRewriterSubagent,
            "document_analyzer": DocumentAnalyzerSubagent,
            "video_analyzer": VideoAnalyzerSubagent,
            "image_analyzer": ImageAnalyzerSubagent
        }

    def register_subagent(self, task_type: str, subagent_class):
        """Register a custom subagent type."""
        self._subagent_classes[task_type] = subagent_class

    async def create_subagent(
        self,
        task_type: str,
        subagent_id: str,
        parameters: Dict[str, Any]
    ):
        """Create a subagent instance."""
        subagent_class = self._subagent_classes.get(task_type)

        if not subagent_class:
            logger.error(f"Unknown subagent type: {task_type}")
            return None

        try:
            subagent = subagent_class(
                subagent_id=subagent_id,
                config=self.config
            )
            await subagent.initialize()
            return subagent
        except Exception as e:
            logger.error(f"Error creating subagent: {e}")
            return None
