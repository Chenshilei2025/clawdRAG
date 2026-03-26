"""
Base subagent class and interface definitions.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Subagent(ABC):
    """Abstract base class for all subagents."""

    def __init__(self, subagent_id: str, config: Optional[Dict[str, Any]] = None):
        self.subagent_id = subagent_id
        self.config = config or {}
        self._initialized = False

    @abstractmethod
    async def initialize(self):
        """Initialize the subagent (called after creation)."""
        pass

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the subagent's primary task."""
        pass

    @abstractmethod
    async def cleanup(self):
        """Clean up resources (called after execution)."""
        pass

    @property
    def initialized(self) -> bool:
        """Check if subagent is initialized."""
        return self._initialized


class SimpleSubagent(Subagent):
    """Simple subagent implementation for basic tasks."""

    def __init__(self, subagent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(subagent_id, config)
        self.llm_provider = None
        self.task_description = ""

    async def initialize(self):
        """Initialize LLM provider."""
        provider_type = self.config.get("llm_provider", "openai")
        provider_config = self.config.get("llm_config", {})

        if provider_type == "openai":
            from ...providers.openai import OpenAILLM
            self.llm_provider = OpenAILLM(provider_config)

        self._initialized = True
        logger.info(f"Subagent {self.subagent_id} initialized")

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the task using LLM."""
        if not self._initialized:
            await self.initialize()

        # Build prompt from parameters
        prompt = self._build_prompt(parameters)

        # Execute via LLM
        from ...providers.base import Message
        messages = [
            Message(role="system", content=self.task_description),
            Message(role="user", content=prompt)
        ]

        response = await self.llm_provider.generate(messages)

        return {
            "result": response.content,
            "subagent_id": self.subagent_id,
            "model": response.model
        }

    def _build_prompt(self, parameters: Dict[str, Any]) -> str:
        """Build prompt from parameters."""
        return str(parameters)

    async def cleanup(self):
        """Clean up resources."""
        self._initialized = False
        logger.info(f"Subagent {self.subagent_id} cleaned up")
