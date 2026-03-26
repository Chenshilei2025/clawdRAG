"""
OpenAI provider implementations.
"""
from typing import List, Dict, Any, Union
import asyncio
from .base import (
    LLMProvider,
    EmbeddingProvider,
    ImageCaptioningProvider,
    OCRProvider,
    Message,
    LLMResponse
)


class OpenAILLM(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=config.get("api_key"),
                base_url=config.get("api_base")
            )
            self.model = config.get("model", "gpt-4o")
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        openai_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )

    async def stream_generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ):
        """Stream generate response using OpenAI API."""
        openai_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OpenAIEmbedding(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=config.get("api_key"))
            self.model = config.get("model", "text-embedding-3-large")
        except ImportError:
            raise ImportError("openai package is required")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]


class OpenAIVision(ImageCaptioningProvider, OCRProvider):
    """OpenAI Vision provider for both captioning and OCR."""

    def __init__(self, config: Dict[str, Any]):
        ImageCaptioningProvider.__init__(self, config)
        OCRProvider.__init__(self, config)
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=config.get("api_key"))
            self.model = config.get("model", "gpt-4o")
        except ImportError:
            raise ImportError("openai package is required")

    async def _process_image(self, image_path: str, prompt: str) -> str:
        """Process image with vision model."""
        import base64

        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode()

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content

    async def caption_image(self, image_path: str) -> str:
        """Generate caption for an image."""
        return await self._process_image(
            image_path,
            "Describe this image in detail, including the main subjects, actions, and context."
        )

    async def extract_text(self, image_path: str) -> str:
        """Extract text from an image."""
        return await self._process_image(
            image_path,
            "Extract all visible text from this image. Preserve the layout and structure."
        )
