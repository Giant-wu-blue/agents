from __future__ import annotations

import os
import asyncio
import logging

logger = logging.getLogger(__name__)

BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMB_MODEL = os.getenv("BAILIAN_EMB_MODEL", "text-embedding-v4")
EMB_DIM = int(os.getenv("BAILIAN_EMB_DIM", "1024"))
_BATCH = 10


class BailianEmbedder:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=BAILIAN_BASE_URL,
                api_key=os.getenv("DASHSCOPE_API_KEY", "sk-placeholder"),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """文本 → 向量列表。自动分批(每批≤10),保持输入顺序。"""
        if not texts:
            return []
        client = self._get_client()
        vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            batch = texts[i : i + _BATCH]
            resp = await client.embeddings.create(
                model=EMB_MODEL,
                input=batch,
                dimensions=EMB_DIM,
                encoding_format="float",
            )
            vectors.extend([item.embedding for item in resp.data])
        return vectors

    async def encode_one(self, text: str) -> list[float]:
        out = await self.encode([text])
        return out[0] if out else []
