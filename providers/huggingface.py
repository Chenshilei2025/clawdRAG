"""
Hugging Face provider implementations.
"""
from typing import List, Dict, Any
import asyncio
from .base import (
    LLMProvider,
    EmbeddingProvider,
    Message,
    LLMResponse
)


class HuggingFaceLLM(LLMProvider):
    """Hugging Face LLM provider using transformers or inference API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model", "meta-llama/Llama-3-8B")
        self.use_api = config.get("use_api", False)

        if self.use_api:
            self.api_key = config.get("api_key")
        else:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            except ImportError:
                raise ImportError("transformers and torch are required for local inference")

    async def generate(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Hugging Face model."""
        if self.use_api:
            return await self._generate_via_api(messages, temperature, max_tokens, **kwargs)
        else:
            return await self._generate_local(messages, temperature, max_tokens, **kwargs)

    async def _generate_via_api(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Generate via Hugging Face Inference API."""
        import aiohttp

        prompt = self._format_messages(messages)
        url = f"https://api-inference.huggingface.co/models/{self.model_name}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "temperature": temperature,
                        "max_new_tokens": max_tokens
                    }
                }
            ) as response:
                result = await response.json()
                content = result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")

        return LLMResponse(content=content, model=self.model_name)

    async def _generate_local(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> LLMResponse:
        """Generate locally using transformers."""
        prompt = self._format_messages(messages)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return LLMResponse(content=response, model=self.model_name)

    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages for the model."""
        formatted = []
        for msg in messages:
            formatted.append(f"{msg.role}: {msg.content}")
        return "\n".join(formatted)

    async def stream_generate(self, messages, temperature=0.7, max_tokens=2048, **kwargs):
        """Stream generate (not fully implemented for HF)."""
        response = await self.generate(messages, temperature, max_tokens, **kwargs)
        for char in response.content:
            yield char


class HuggingFaceEmbedding(EmbeddingProvider):
    """Hugging Face embedding provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model", "sentence-transformers/all-MiniLM-L6-v2")

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
