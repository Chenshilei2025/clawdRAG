"""
Memory consolidator for multimodal RAG.
Handles long-term and short-term memory with archival.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """A single memory item."""
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "conversation"
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5  # 0.0 to 1.0
    access_count: int = 0


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_items: int
    total_tokens: int
    oldest_memory: Optional[datetime]
    newest_memory: Optional[datetime]


class MemoryConsolidator:
    """
    Manages memory consolidation for multimodal contexts.
    Handles short-term working memory and long-term archival.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_short_term = self.config.get("max_short_term_items", 100)
        self.max_long_term = self.config.get("max_long_term_items", 10000)
        self.archive_path = Path(self.config.get("archive_path", "./workspace/memory_archive"))

        self._short_term: List[MemoryItem] = []
        self._long_term: List[MemoryItem] = []

        self.archive_path.mkdir(parents=True, exist_ok=True)

    async def add_memory(
        self,
        content: str,
        source: str = "conversation",
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> MemoryItem:
        """Add a new memory item."""
        memory = MemoryItem(
            content=content,
            source=source,
            embedding=embedding,
            metadata=metadata or {},
            importance=importance
        )

        self._short_term.append(memory)

        # Check if we need to consolidate
        if len(self._short_term) >= self.max_short_term:
            await self._consolidate()

        return memory

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_similarity: float = 0.7
    ) -> List[MemoryItem]:
        """Search memories by embedding similarity."""
        import numpy as np

        candidates = self._short_term + self._long_term

        if not candidates:
            return []

        # Calculate similarities
        results = []
        query_vec = np.array(query_embedding)

        for memory in candidates:
            if memory.embedding:
                mem_vec = np.array(memory.embedding)
                similarity = np.dot(query_vec, mem_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(mem_vec)
                )
                if similarity >= min_similarity:
                    memory.access_count += 1
                    results.append((similarity, memory))

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in results[:top_k]]

    async def get_recent(self, limit: int = 20) -> List[MemoryItem]:
        """Get recent short-term memories."""
        return self._short_term[-limit:]

    async def get_important(self, limit: int = 20) -> List[MemoryItem]:
        """Get most important memories."""
        all_memories = self._short_term + self._long_term
        sorted_memories = sorted(
            all_memories,
            key=lambda m: m.importance,
            reverse=True
        )
        return sorted_memories[:limit]

    async def _consolidate(self):
        """Consolidate short-term memory to long-term or archive."""
        logger.info("Starting memory consolidation...")

        # Calculate importance scores if not set
        await self._update_importance_scores()

        # Keep important items in short-term
        important = [m for m in self._short_term if m.importance > 0.7]
        less_important = [m for m in self._short_term if m.importance <= 0.7]

        # Move less important to long-term if space
        for memory in less_important:
            if len(self._long_term) < self.max_long_term:
                self._long_term.append(memory)
            else:
                # Archive to disk
                await self._archive_memory(memory)

        self._short_term = important[-self.max_short_term // 2:]

        logger.info(f"Consolidation complete. Short-term: {len(self._short_term)}, Long-term: {len(self._long_term)}")

    async def _update_importance_scores(self):
        """Update importance scores based on various factors."""
        now = datetime.utcnow()
        time_decay = 0.1  # Decay per day

        for memory in self._short_term:
            # Base score on recency
            age_days = (now - memory.timestamp).total_seconds() / 86400
            recency_score = max(0, 1 - age_days * time_decay)

            # Factor in access count
            access_bonus = min(0.3, memory.access_count * 0.05)

            # Combine scores
            memory.importance = min(1.0, recency_score + access_bonus)

    async def _archive_memory(self, memory: MemoryItem):
        """Archive a memory to disk."""
        archive_file = self.archive_path / f"{memory.timestamp.strftime('%Y%m%d')}_memory.jsonl"

        # Append to archive file
        with open(archive_file, "a") as f:
            f.write(json.dumps({
                "content": memory.content,
                "timestamp": memory.timestamp.isoformat(),
                "source": memory.source,
                "metadata": memory.metadata,
                "importance": memory.importance
            }) + "\n")

    def get_stats(self) -> MemoryStats:
        """Get memory statistics."""
        all_memories = self._short_term + self._long_term

        timestamps = [m.timestamp for m in all_memories]

        return MemoryStats(
            total_items=len(all_memories),
            total_tokens=sum(len(m.content.split()) for m in all_memories),
            oldest_memory=min(timestamps) if timestamps else None,
            newest_memory=max(timestamps) if timestamps else None
        )

    async def clear_old(self, days: int = 30):
        """Clear memories older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days)

        original_count = len(self._short_term) + len(self._long_term)

        self._short_term = [m for m in self._short_term if m.timestamp > cutoff]
        self._long_term = [m for m in self._long_term if m.timestamp > cutoff]

        cleared = original_count - (len(self._short_term) + len(self._long_term))
        logger.info(f"Cleared {cleared} old memories")
