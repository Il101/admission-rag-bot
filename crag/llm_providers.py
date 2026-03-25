"""
LLM Provider Abstraction Layer

Supports multiple AI providers:
- Google Gemini (default)
- OpenAI-compatible APIs (NVIDIA, OpenRouter, etc.)
- OpenAI
- Anthropic Claude (future)
"""

import os
import asyncio
from typing import Optional, Dict, Any, AsyncIterator, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: str  # "google", "openai", "nvidia", "anthropic"
    model: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0

    # Embedding settings
    embedding_provider: Optional[str] = None  # "google", "nvidia", "openai"
    embedding_model: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load config from environment variables"""
        provider = os.getenv("LLM_PROVIDER", "google").lower()

        # Embedding config
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", provider).lower()
        embedding_model = os.getenv("EMBEDDING_MODEL")
        embedding_api_key = os.getenv("EMBEDDING_API_KEY")
        embedding_base_url = os.getenv("EMBEDDING_BASE_URL")

        if provider == "google":
            return cls(
                provider="google",
                model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-exp"),
                api_key=os.getenv("GOOGLE_API_KEY", ""),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
                embedding_provider=embedding_provider or "google",
                embedding_model=embedding_model or "gemini-embedding-001",
                embedding_api_key=embedding_api_key or os.getenv("GOOGLE_API_KEY", ""),
                embedding_base_url=embedding_base_url,
            )
        elif provider in ["nvidia", "openai"]:
            base_url = os.getenv("OPENAI_BASE_URL")
            if provider == "nvidia" and not base_url:
                base_url = "https://integrate.api.nvidia.com/v1"

            # Default embedding settings for NVIDIA/OpenAI
            if not embedding_base_url and embedding_provider == "nvidia":
                embedding_base_url = "https://integrate.api.nvidia.com/v1"

            if not embedding_model:
                if embedding_provider == "nvidia":
                    embedding_model = "nvidia/nv-embed-v1"
                elif embedding_provider == "openai":
                    embedding_model = "text-embedding-3-small"
                else:
                    embedding_model = "gemini-embedding-001"

            return cls(
                provider=provider,
                model=os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b"),
                api_key=os.getenv("NVIDIA_API_KEY" if provider == "nvidia" else "OPENAI_API_KEY", ""),
                base_url=base_url,
                temperature=float(os.getenv("LLM_TEMPERATURE", "1.0")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
                top_p=float(os.getenv("LLM_TOP_P", "1.0")),
                embedding_provider=embedding_provider or provider,
                embedding_model=embedding_model,
                embedding_api_key=embedding_api_key or os.getenv("NVIDIA_API_KEY" if provider == "nvidia" else "OPENAI_API_KEY", ""),
                embedding_base_url=embedding_base_url,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class BaseLLMProvider:
    """Base class for LLM providers"""

    def __init__(self, config: LLMConfig):
        self.config = config

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text response"""
        raise NotImplementedError

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncIterator[str]:
        """Generate streaming text response"""
        raise NotImplementedError

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        raise NotImplementedError


class GoogleLLMProvider(BaseLLMProvider):
    """Google Gemini provider using new google-genai SDK"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=config.api_key)
        self.model_name = config.model
        self.types = types
        self.config_obj = config

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text response"""
        from crag.simple_rag import retry_on_503

        config = self.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.config_obj.temperature,
        )

        response = await retry_on_503(
            self.client.aio.models.generate_content,
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncIterator[str]:
        """Generate streaming text response"""
        from crag.simple_rag import retry_on_503

        config = self.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.config_obj.temperature,
        )

        stream = await retry_on_503(
            self.client.aio.models.generate_content_stream,
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        from crag.simple_rag import retry_on_503

        response = await retry_on_503(
            self.client.aio.models.embed_content,
            model="models/gemini-embedding-001",
            contents=texts,
        )
        return [list(emb.values) for emb in response.embeddings]


class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI-compatible provider (works with NVIDIA, OpenRouter, etc.)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

        # Initialize embedding client based on embedding_provider
        self.embedding_provider = config.embedding_provider
        self.embedding_model = config.embedding_model

        if config.embedding_provider == "nvidia":
            self.embedding_client = AsyncOpenAI(
                api_key=config.embedding_api_key or config.api_key,
                base_url=config.embedding_base_url or "https://integrate.api.nvidia.com/v1"
            )
        elif config.embedding_provider == "openai":
            self.embedding_client = AsyncOpenAI(
                api_key=config.embedding_api_key or config.api_key,
                base_url=config.embedding_base_url or "https://api.openai.com/v1"
            )
        elif config.embedding_provider == "google":
            # Use Google for embeddings
            from google import genai
            google_key = config.embedding_api_key or os.getenv("GOOGLE_API_KEY")
            if not google_key:
                raise ValueError("GOOGLE_API_KEY required for Google embeddings")
            self.google_client = genai.Client(api_key=google_key)
            self.embedding_client = None
        else:
            raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text response"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        )
        return response.choices[0].message.content

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncIterator[str]:
        """Generate streaming text response"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            # Handle reasoning content if present (for models like GPT-OSS)
            reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
            if reasoning:
                yield reasoning

            # Handle regular content
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        if self.embedding_provider == "google":
            # Use Google embeddings
            from crag.simple_rag import retry_on_503
            response = await retry_on_503(
                self.google_client.aio.models.embed_content,
                model="models/gemini-embedding-001",
                contents=texts,
            )
            return [list(emb.values) for emb in response.embeddings]
        else:
            # Use OpenAI-compatible embeddings (NVIDIA, OpenAI)
            response = await self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]


class LLMProviderFactory:
    """Factory for creating LLM providers"""

    _instance: Optional[BaseLLMProvider] = None

    @classmethod
    def get_provider(cls, config: Optional[LLMConfig] = None) -> BaseLLMProvider:
        """Get or create LLM provider instance (singleton)"""
        if cls._instance is None:
            if config is None:
                config = LLMConfig.from_env()

            if config.provider == "google":
                cls._instance = GoogleLLMProvider(config)
            elif config.provider in ["openai", "nvidia"]:
                cls._instance = OpenAICompatibleProvider(config)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")

            logger.info(f"Initialized LLM provider: {config.provider} with model {config.model}")

        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton instance (useful for testing)"""
        cls._instance = None


# Convenience function
def get_llm() -> BaseLLMProvider:
    """Get the default LLM provider"""
    return LLMProviderFactory.get_provider()
