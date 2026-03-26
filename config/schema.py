"""
Configuration schema for MM RAG Agent.
Defines Pydantic models for system configuration.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = Field(default="openai", description="LLM provider: openai, huggingface")
    model: str = Field(default="gpt-4o", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    api_base: Optional[str] = Field(default=None, description="API base URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)


class EmbeddingConfig(BaseModel):
    """Embedding provider configuration."""
    provider: str = Field(default="openai", description="Embedding provider")
    model: str = Field(default="text-embedding-3-large", description="Embedding model")
    dimension: int = Field(default=3072, description="Embedding dimension")


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    type: str = Field(default="chroma", description="Vector store type")
    path: str = Field(default="./workspace/vector_store", description="Storage path")
    collection_name: str = Field(default="mm_rag", description="Collection name")


class OCRConfig(BaseModel):
    """OCR provider configuration."""
    provider: str = Field(default="openai", description="OCR provider")
    model: str = Field(default="gpt-4o", description="Vision model for OCR")


class ImageCaptioningConfig(BaseModel):
    """Image captioning configuration."""
    provider: str = Field(default="openai", description="Captioning provider")
    model: str = Field(default="gpt-4o", description="Vision model for captioning")


class SystemConfig(BaseModel):
    """Main system configuration."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    image_captioning: ImageCaptioningConfig = Field(default_factory=ImageCaptioningConfig)
    workspace_path: str = Field(default="./workspace", description="Workspace directory")
    max_memory_size: int = Field(default=10000, description="Max memory tokens")
    enable_subagents: bool = Field(default=True, description="Enable subagent spawning")
