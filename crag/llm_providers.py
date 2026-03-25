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

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load config from environment variables"""
        provider = os.getenv("LLM_PROVIDER", "google").lower()

        if provider == "google":
            return cls(
                provider="google",
                model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash-exp"),
                api_key=os.getenv("GOOGLE_API_KEY", ""),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            )
        elif provider in ["nvidia", "openai"]:
            base_url = os.getenv("OPENAI_BASE_URL")
            if provider == "nvidia" and not base_url:
                base_url = "https://integrate.api.nvidia.com/v1"

            return cls(
                provider=provider,
                model=os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b"),
                api_key=os.getenv("NVIDIA_API_KEY" if provider == "nvidia" else "OPENAI_API_KEY", ""),
                base_url=base_url,
                temperature=float(os.getenv("LLM_TEMPERATURE", "1.0")),
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
                top_p=float(os.getenv("LLM_TOP_P", "1.0")),
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
    """Google Gemini provider"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        import google.generativeai as genai
        genai.configure(api_key=config.api_key)

        self.model = genai.GenerativeModel(
            model_name=config.model,
            generation_config={
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
            }
        )
        self.embed_model = genai.GenerativeModel("gemini-embedding-001")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text response"""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.model.generate_content(full_prompt)
        )
        return response.text

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None) -> AsyncIterator[str]:
        """Generate streaming text response"""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.model.generate_content(full_prompt, stream=True)
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        import google.generativeai as genai

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(
                model="models/gemini-embedding-001",
                content=texts,
                task_type="retrieval_document"
            )
        )
        return result['embedding']


class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI-compatible provider (works with NVIDIA, OpenRouter, etc.)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

        # For embeddings, use Google by default (can be overridden)
        self.embed_provider = None
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key:
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            self.has_google_embed = True
        else:
            self.has_google_embed = False
            logger.warning("No GOOGLE_API_KEY found. Embeddings will not work unless using OpenAI embeddings.")

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
        # Try OpenAI embeddings first
        if self.config.base_url and "openai.com" in self.config.base_url:
            response = await self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [item.embedding for item in response.data]

        # Fallback to Google embeddings
        if self.has_google_embed:
            import google.generativeai as genai
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=texts,
                    task_type="retrieval_document"
                )
            )
            return result['embedding']
        else:
            raise ValueError("No embedding provider available. Set GOOGLE_API_KEY or use OpenAI.")


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
