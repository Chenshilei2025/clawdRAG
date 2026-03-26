"""
Main agent implementing the PEO (Perceive-Execute-Optimize) loop.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from ..bus.events import InboundMessage, OutboundMessage
from ..bus.queue import MessageBus
from ..session.manager import SessionManager, Session
from .context.builder import ContextBuilder, RAGContext
from .memory.consolidator import MemoryConsolidator, MemoryItem
from .tools.base import ToolRegistry
from .tools.subagent_manager import SubagentFactory, SubagentManagerTool

logger = logging.getLogger(__name__)


class MainAgent:
    """
    Main agent implementing the PEO loop:
    - Perceive: Receive input and build context
    - Execute: Process with LLM and tools
    - Optimize: Learn and improve from feedback
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_id = f"main_agent_{uuid.uuid4().hex[:8]}"

        # Initialize components
        self.message_bus = MessageBus()
        self.session_manager = SessionManager(
            base_workspace=config.get("workspace_path", "./workspace/sessions")
        )
        self.context_builder = ContextBuilder(config.get("context_config", {}))
        self.memory_consolidator = MemoryConsolidator(config.get("memory_config", {}))

        # Initialize LLM provider
        self._init_llm_provider()

        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        self._init_tools()

        # Initialize subagent factory
        self.subagent_factory = SubagentFactory(config)
        subagent_manager = SubagentManagerTool(config)
        subagent_manager.set_subagent_factory(self.subagent_factory)
        self.tool_registry.register(subagent_manager)

        # State
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

    def _init_llm_provider(self):
        """Initialize the LLM provider."""
        llm_config = self.config.get("llm", {})
        provider_type = llm_config.get("provider", "openai")

        if provider_type == "openai":
            from ..providers.openai import OpenAILLM
            provider_config = {
                "api_key": llm_config.get("api_key"),
                "api_base": llm_config.get("api_base"),
                "model": llm_config.get("model", "gpt-4o")
            }
            self.llm_provider = OpenAILLM(provider_config)
        elif provider_type == "huggingface":
            from ..providers.huggingface import HuggingFaceLLM
            provider_config = {
                "model": llm_config.get("model", "meta-llama/Llama-3-8B"),
                "api_key": llm_config.get("api_key"),
                "use_api": llm_config.get("use_api", False)
            }
            self.llm_provider = HuggingFaceLLM(provider_config)
        else:
            raise ValueError(f"Unknown LLM provider: {provider_type}")

    def _init_tools(self):
        """Initialize available tools."""
        from .tools.common.filesystem import ReadFileTool, ListDirTool
        from .tools.common.message import MessageTool
        from .tools.multimodal.vector_search import VectorSearchTool
        from .tools.multimodal.image_ocr import ImageOCRTool
        from .tools.multimodal.image_captioning import ImageCaptioningTool

        tool_config = self.config.get("tool_config", {})

        # Common tools
        self.tool_registry.register(ReadFileTool(tool_config))
        self.tool_registry.register(ListDirTool(tool_config))
        self.tool_registry.register(MessageTool(tool_config))

        # Multimodal tools
        self.tool_registry.register(VectorSearchTool(tool_config))
        self.tool_registry.register(ImageOCRTool(tool_config))
        self.tool_registry.register(ImageCaptioningTool(tool_config))

        logger.info(f"Initialized {len(self.tool_registry.list_tools())} tools")

    async def start(self):
        """Start the main agent."""
        if self._running:
            return

        self._running = True
        await self.message_bus.start()
        self._worker_task = asyncio.create_task(self._main_loop())
        logger.info(f"Main agent {self.agent_id} started")

    async def stop(self):
        """Stop the main agent."""
        self._running = False
        await self.message_bus.stop()

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Main agent {self.agent_id} stopped")

    async def process_message(
        self,
        content: str,
        session_id: str,
        attachments: Optional[List[str]] = None
    ) -> OutboundMessage:
        """
        Process an incoming message and return a response.

        This is the main entry point for the PEO loop.
        """
        # Perceive: Get or create session and build context
        session = await self.session_manager.get_or_create_session(session_id)

        # Execute: Process the message
        response = await self._execute(content, session, attachments or [])

        # Save to session
        session.add_message("user", content)
        session.add_message("assistant", response.content)

        # Optimize: Store in memory
        await self._optimize(content, response.content, session)

        # Create outbound message
        return OutboundMessage(
            content=response.content,
            session_id=session_id,
            is_complete=True
        )

    async def _execute(
        self,
        query: str,
        session: Session,
        attachments: List[str]
    ) -> "LLMResponse":
        """Execute the PEO loop's execute phase."""
        # Step 1: Vector search for relevant context
        retrieved_docs = []
        try:
            from .tools.multimodal.vector_search import VectorSearchTool
            search_tool = VectorSearchTool(self.config.get("tool_config", {}))
            search_result = await search_tool.execute(query=query, top_k=5)
            if search_result.success:
                retrieved_docs = search_result.data.get("results", [])
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

        # Step 2: Build context
        context = await self.context_builder.build(
            query=query,
            session=session,
            retrieved_docs=retrieved_docs,
            attachments=attachments
        )

        # Step 3: Format for LLM
        messages = self.context_builder.format_for_llm(context)

        # Step 4: Generate response
        from ..providers.base import Message
        provider_messages = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]

        response = await self.llm_provider.generate(
            messages=provider_messages,
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 2048)
        )

        return response

    async def _optimize(self, query: str, response: str, session: Session):
        """Optimize by storing experience in memory."""
        # Create memory item
        interaction = f"Q: {query}\nA: {response}"

        # Generate embedding for the memory
        try:
            from .tools.multimodal.embedding import EmbeddingGeneratorTool
            embedder = EmbeddingGeneratorTool(self.config.get("tool_config", {}))
            embed_result = await embedder.execute(content=interaction, content_type="text")

            embedding = None
            if embed_result.success:
                embedding = embed_result.data.get("embedding")

            # Store in memory
            await self.memory_consolidator.add_memory(
                content=interaction,
                source="conversation",
                embedding=embedding,
                metadata={
                    "session_id": session.session_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        except Exception as e:
            logger.warning(f"Failed to store in memory: {e}")

    async def _main_loop(self):
        """Main event loop for processing messages from the bus."""
        while self._running:
            try:
                # Check for inbound messages
                message = await self.message_bus.receive_inbound()

                # Process the message
                response = await self.process_message(
                    content=message.content,
                    session_id=message.session_id or "default",
                    attachments=message.attachments
                )

                # Send response
                await self.message_bus.send_outbound(response)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        memory_stats = self.memory_consolidator.get_stats()

        return {
            "agent_id": self.agent_id,
            "running": self._running,
            "tools_available": len(self.tool_registry.list_tools()),
            "memory_stats": {
                "total_items": memory_stats.total_items,
                "total_tokens": memory_stats.total_tokens
            }
        }
