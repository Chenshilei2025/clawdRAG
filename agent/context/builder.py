"""
Context builder for multimodal RAG.
Constructs context for LLM input from various sources.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MultimodalContent:
    """Represents content that can be text, image, or mixed."""
    type: str  # "text", "image", "video", "audio", "mixed"
    text: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGContext:
    """Context for RAG generation."""
    query: str
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: Optional[str] = None
    multimodal_content: List[MultimodalContent] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextBuilder:
    """
    Builds context for LLM input in multimodal RAG scenarios.
    Handles text, images, and mixed content.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_context_length = self.config.get("max_context_length", 128000)
        self.max_history = self.config.get("max_history", 10)

    async def build(
        self,
        query: str,
        session,
        retrieved_docs: Optional[List[Dict]] = None,
        attachments: Optional[List[str]] = None,
        tool_results: Optional[List[Dict]] = None
    ) -> RAGContext:
        """Build complete RAG context."""
        context = RAGContext(query=query)

        # Add conversation history
        history = session.get_messages(limit=self.max_history)
        context.conversation_history = history

        # Add retrieved documents
        if retrieved_docs:
            context.retrieved_documents = retrieved_docs[:self.config.get("max_retrieved", 20)]

        # Process attachments as multimodal content
        if attachments:
            for attachment_path in attachments:
                content = await self._process_attachment(attachment_path)
                if content:
                    context.multimodal_content.append(content)

        # Add tool results
        if tool_results:
            context.tool_results = tool_results

        # Set system prompt
        context.system_prompt = self._get_system_prompt()

        return context

    async def _process_attachment(self, file_path: str) -> Optional[MultimodalContent]:
        """Process an attachment file."""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Attachment not found: {file_path}")
            return None

        suffix = path.suffix.lower()

        # Image files
        if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}:
            return MultimodalContent(
                type="image",
                image_path=str(path),
                metadata={"filename": path.name}
            )

        # Video files
        elif suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            return MultimodalContent(
                type="video",
                text=f"[Video file: {path.name}]",
                metadata={"filename": path.name, "path": str(path)}
            )

        # Audio files
        elif suffix in {".mp3", ".wav", ".m4a", ".flac", ".aac"}:
            return MultimodalContent(
                type="audio",
                text=f"[Audio file: {path.name}]",
                metadata={"filename": path.name, "path": str(path)}
            )

        # Document files
        elif suffix in {".pdf", ".ppt", ".pptx", ".doc", ".docx", ".txt"}:
            return MultimodalContent(
                type="document",
                text=f"[Document: {path.name}]",
                metadata={"filename": path.name, "path": str(path)}
            )

        # Text files
        elif suffix == ".txt":
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                return MultimodalContent(
                    type="text",
                    text=content,
                    metadata={"filename": path.name}
                )
            except Exception as e:
                logger.error(f"Error reading text file: {e}")
                return None

        else:
            return MultimodalContent(
                type="unknown",
                text=f"[File: {path.name}]",
                metadata={"filename": path.name, "path": str(path)}
            )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are a helpful multimodal AI assistant with RAG capabilities. You can:

1. Answer questions using retrieved contextual information
2. Analyze images and extract text from them
3. Process and understand documents (PDF, PPT, etc.)
4. Analyze video content
5. Use various tools to help users

Always base your answers on the provided context when available.
If the context doesn't contain relevant information, clearly state that.
For multimodal content, describe what you see and extract relevant information."""

    def format_for_llm(self, context: RAGContext) -> List[Dict[str, Any]]:
        """Format context for LLM input."""
        messages = []

        # System prompt
        if context.system_prompt:
            messages.append({"role": "system", "content": context.system_prompt})

        # Add retrieved documents as context
        if context.retrieved_documents:
            doc_context = self._format_retrieved_docs(context.retrieved_documents)
            messages.append({
                "role": "system",
                "content": f"RELEVANT CONTEXT:\n{doc_context}"
            })

        # Conversation history
        for msg in context.conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Current query with multimodal content
        user_message = self._format_user_message(context)
        messages.append({"role": "user", "content": user_message})

        return messages

    def _format_retrieved_docs(self, docs: List[Dict]) -> str:
        """Format retrieved documents."""
        formatted = []
        for i, doc in enumerate(docs[:5], 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")
            formatted.append(f"[Doc {i} from {source}]: {content[:500]}")
        return "\n\n".join(formatted)

    def _format_user_message(self, context: RAGContext) -> Any:
        """Format user message with multimodal content."""
        if not context.multimodal_content:
            return context.query

        # Build multimodal message
        content = [{"type": "text", "text": context.query}]

        for mc in context.multimodal_content:
            if mc.type == "image" and mc.image_path:
                import base64
                with open(mc.image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
            elif mc.text:
                content.append({"type": "text", "text": mc.text})

        return content
