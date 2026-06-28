import logging
from typing import Any

import httpx

from app.schemas import (
    ChunkItem,
    EmbeddingResponse,
    MCPInvokeRequest,
    MCPInvokeResponse,
    RetrievalRequest,
    RetrievalResponse,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8080/api/internal"


class JavaClientError(Exception):
    """Wrapper for errors from the Java backend."""


class JavaClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def retrieval_search(
        self, query: str, intent_node: str = "general", top_k: int = 10
    ) -> RetrievalResponse:
        """POST /api/internal/retrieval/search"""
        client = await self._get_client()
        body = RetrievalRequest(query=query, topK=top_k, intentNode=intent_node)
        try:
            resp = await client.post(
                f"{self.base_url}/retrieval/search",
                json=body.model_dump(by_alias=True),
            )
            resp.raise_for_status()
            return RetrievalResponse(**resp.json())
        except httpx.HTTPError as e:
            logger.error(f"Retrieval search failed: {e}")
            raise JavaClientError(f"Retrieval search failed: {e}") from e

    async def mcp_invoke(self, tool_id: str, params: dict[str, Any]) -> MCPInvokeResponse:
        """POST /api/internal/mcp/invoke"""
        client = await self._get_client()
        body = MCPInvokeRequest(toolId=tool_id, params=params)
        try:
            resp = await client.post(
                f"{self.base_url}/mcp/invoke",
                json=body.model_dump(by_alias=True),
            )
            resp.raise_for_status()
            return MCPInvokeResponse(**resp.json())
        except httpx.HTTPError as e:
            logger.error(f"MCP invoke failed: {e}")
            raise JavaClientError(f"MCP invoke failed: {e}") from e

    async def embedding_encode(self, texts: list[str]) -> dict[str, Any]:
        """POST /api/internal/embedding/encode

        Returns raw dict (not pydantic model) for compat with compression pipeline.
        """
        client = await self._get_client()
        try:
            resp = await client.post(
                f"{self.base_url}/embedding/encode",
                json={"texts": texts},
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            logger.error(f"Embedding encode failed: {e}")
            raise JavaClientError(f"Embedding encode failed: {e}") from e
        
    async def list_all_citations(self, kb_id: str | None = None) -> list[str]:
        """GET all legal citations from the knowledge base — for Red Agent hallucination detection."""
        client = await self._get_client()
        try:
            resp = await client.post(
                f"{self.base_url}/agent/citations/list-all",
                json={"knowledgeBaseId": kb_id},
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                raise JavaClientError(f"list citations error: {data.get('message')}")
            return data["data"]["citations"]
        except httpx.HTTPError as e:
            logger.error(f"List all citations failed: {e}")
            raise JavaClientError(f"List all citations failed: {e}") from e
