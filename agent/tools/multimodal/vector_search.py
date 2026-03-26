"""
Vector search tool for semantic retrieval.
"""
from typing import Dict, Any, List, Optional
import logging

from ..base import Tool, ToolResult

logger = logging.getLogger(__name__)


class VectorSearchTool(Tool):
    """Search vector database for semantically similar content."""

    name = "vector_search"
    description = "Search the vector database for semantically similar content using embeddings."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vector_store = None
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize vector store."""
        store_type = self.config.get("store_type", "chroma")
        store_path = self.config.get("store_path", "./workspace/vector_store")
        collection_name = self.config.get("collection_name", "mm_rag")

        if store_type == "chroma":
            try:
                import chromadb
                client = chromadb.PersistentClient(path=store_path)
                self.vector_store = client.get_or_create_collection(name=collection_name)
            except ImportError:
                logger.warning("chromadb not installed, using in-memory store")
                self.vector_store = None
        else:
            logger.warning(f"Unknown store type: {store_type}")

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Search for similar documents."""
        try:
            if not self.vector_store:
                return ToolResult(
                    success=False,
                    error="Vector store not initialized"
                )

            # Generate query embedding
            from ...multimodal.embedding import EmbeddingGeneratorTool
            embedder = EmbeddingGeneratorTool(self.config)
            result = await embedder.execute(content=query, content_type="text")

            if not result.success:
                return ToolResult(success=False, error="Failed to generate query embedding")

            query_embedding = result.data["embedding"]

            # Search in vector store
            search_results = self.vector_store.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters
            )

            # Format results
            documents = []
            if search_results and search_results.get("documents"):
                for i, doc in enumerate(search_results["documents"][0]):
                    documents.append({
                        "content": doc,
                        "metadata": search_results["metadatas"][0][i] if search_results.get("metadatas") else {},
                        "distance": search_results["distances"][0][i] if search_results.get("distances") else None
                    })

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": documents,
                    "count": len(documents)
                }
            )

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return ToolResult(success=False, error=str(e))

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query text to search for"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                },
                "filters": {
                    "type": "object",
                    "description": "Optional filters to apply to search"
                }
            },
            "required": ["query"]
        }


class VectorIndexTool(Tool):
    """Tool for adding documents to the vector store."""

    name = "index_document"
    description = "Add a document to the vector store for semantic search."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.vector_store = None
        self._init_vector_store()

    def _init_vector_store(self):
        """Initialize vector store."""
        store_type = self.config.get("store_type", "chroma")
        store_path = self.config.get("store_path", "./workspace/vector_store")
        collection_name = self.config.get("collection_name", "mm_rag")

        if store_type == "chroma":
            try:
                import chromadb
                client = chromadb.PersistentClient(path=store_path)
                self.vector_store = client.get_or_create_collection(name=collection_name)
            except ImportError:
                self.vector_store = None

    async def execute(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> ToolResult:
        """Index a document in the vector store."""
        try:
            if not self.vector_store:
                return ToolResult(
                    success=False,
                    error="Vector store not initialized"
                )

            # Generate embedding
            from ...multimodal.embedding import EmbeddingGeneratorTool
            embedder = EmbeddingGeneratorTool(self.config)
            result = await embedder.execute(content=content, content_type="text")

            if not result.success:
                return ToolResult(success=False, error="Failed to generate embedding")

            embedding = result.data["embedding"]

            # Add to vector store
            import uuid
            doc_id = doc_id or str(uuid.uuid4())

            self.vector_store.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )

            return ToolResult(
                success=True,
                data={
                    "doc_id": doc_id,
                    "indexed": True
                }
            )

        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            return ToolResult(success=False, error=str(e))

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Document content to index"
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata for the document"
                },
                "doc_id": {
                    "type": "string",
                    "description": "Optional document ID (auto-generated if not provided)"
                }
            },
            "required": ["content"]
        }
