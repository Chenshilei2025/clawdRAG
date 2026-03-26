"""
Query rewriter subagent for optimizing user queries.
"""
from typing import Dict, Any, Optional
import logging

from .base import SimpleSubagent

logger = logging.getLogger(__name__)


class QueryRewriterSubagent(SimpleSubagent):
    """Subagent that rewrites and optimizes user queries for better retrieval."""

    def __init__(self, subagent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(subagent_id, config)
        self.task_description = """You are a query rewriting specialist. Your task is to rewrite user queries to make them more effective for semantic search and RAG systems.

Rules:
1. Preserve the original intent and meaning
2. Add relevant context and keywords that might improve retrieval
3. Break down complex queries into sub-queries if needed
4. Suggest filters or constraints that might be helpful
5. Return the rewritten query in JSON format with fields: rewritten_query, sub_queries, suggested_filters"""

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite the user query."""
        original_query = parameters.get("query", "")
        context = parameters.get("context", "")
        conversation_history = parameters.get("conversation_history", [])

        # Build prompt
        prompt_parts = [f"Original Query: {original_query}"]

        if context:
            prompt_parts.append(f"Context: {context}")

        if conversation_history:
            history_text = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in conversation_history[-5:]
            ])
            prompt_parts.append(f"Recent Conversation:\n{history_text}")

        prompt_parts.append("\nPlease rewrite this query for optimal retrieval. Respond in JSON format.")

        prompt = "\n\n".join(prompt_parts)

        # Create messages
        from ...providers.base import Message
        messages = [
            Message(role="system", content=self.task_description),
            Message(role="user", content=prompt)
        ]

        response = await self.llm_provider.generate(
            messages,
            temperature=0.3,
            max_tokens=1000
        )

        # Parse response
        import json
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            result = {
                "rewritten_query": original_query,
                "sub_queries": [],
                "suggested_filters": {}
            }

        return {
            "original_query": original_query,
            "rewritten_query": result.get("rewritten_query", original_query),
            "sub_queries": result.get("sub_queries", []),
            "suggested_filters": result.get("suggested_filters", {}),
            "subagent_id": self.subagent_id
        }
