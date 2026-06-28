
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Cloud model (DeepSeek / Qwen via OpenAI-compatible API)
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "deepseek-chat")
CLOUD_LIGHT_MODEL = os.getenv("CLOUD_LIGHT_MODEL", CLOUD_MODEL)
CLOUD_BASE_URL = os.getenv("CLOUD_BASE_URL", "https://api.deepseek.com/v1")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "")


class LLMClient:
    def __init__(self):
        self._openai_client = None

    async def close(self) -> None:
        if self._openai_client is not None:
            await self._openai_client.close()
            self._openai_client = None


    def _get_openai(self):
        if self._openai_client is None:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(
                base_url=CLOUD_BASE_URL,
                api_key=CLOUD_API_KEY or "sk-placeholder",
            )
        return self._openai_client

    async def chat(
        self,
        prompt: str,
        model: str | None = None,
        use_local: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request.

        Args:
            prompt: The user prompt.
            model: Override default model selection.
            use_local: If True, route to local Ollama; if False, use cloud API.
            temperature: Generation temperature.
            max_tokens: Max output tokens.

        Returns:
            Generated text content.
        """
        if use_local:
            return await self._chat_cloud(prompt, model or CLOUD_LIGHT_MODEL, temperature, max_tokens)
        return await self._chat_cloud(prompt, model or CLOUD_MODEL, temperature, max_tokens)

    async def _chat_cloud(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        client = self._get_openai()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Cloud LLM call failed: {e}")
            raise