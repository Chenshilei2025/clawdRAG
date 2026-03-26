"""
Base provider classes for LLM and multimodal models.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class Message:
    """Message format for LLM communication."""
    role: str
    content: Union[str, List[Dict[str, Any]]]


@dataclass
class LLMResponse:
    """LLM response wrapper."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate response from LLM."""
        pass

    @abstractmethod
    async def stream_generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        """Stream generate response from LLM."""
        pass


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


class ImageCaptioningProvider(ABC):
    """Abstract base class for image captioning providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def caption_image(self, image_path: str) -> str:
        """Generate caption for an image."""
        pass


class OCRProvider(ABC):
    """Abstract base class for OCR providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def extract_text(self, image_path: str) -> str:
        """Extract text from an image."""
        pass
