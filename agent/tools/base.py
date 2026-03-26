"""
Base tool class and tool registry.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ToolSpec:
    """Specification of a tool for LLM function calling."""
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: str = "Result of the tool execution"


class Tool(ABC):
    """Abstract base class for all tools."""

    name: str = ""
    description: str = ""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def get_spec(self) -> ToolSpec:
        """Get the tool specification for function calling."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.get_parameters_schema()
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {},
            "required": []
        }


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, tool_name: str):
        """Unregister a tool."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")

    def get(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_all_specs(self) -> List[ToolSpec]:
        """Get all tool specifications."""
        return [tool.get_spec() for tool in self._tools.values()]

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool not found: {tool_name}"
            )

        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )

    def format_for_llm(self) -> str:
        """Format tool descriptions for LLM context."""
        descriptions = []
        for tool in self._tools.values():
            spec = tool.get_spec()
            descriptions.append(f"- {spec.name}: {spec.description}")

        return "\n".join(descriptions)

    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """Get function schemas for OpenAI function calling."""
        schemas = []
        for tool in self._tools.values():
            spec = tool.get_spec()
            schemas.append({
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters
                }
            })
        return schemas
