from __future__ import annotations

import logging
from typing import Any

from app.schemas import (
    ChunkItem,
    RetrievalResponse,
    MCPInvokeResponse,
)
from app.local.embedding import BailianEmbedder
from app.local.retrieval_store import LocalRetrievalStore
from app.local.sandbox import CodeActSandbox

logger = logging.getLogger(__name__)


class LocalProvider:

    def __init__(self):
        self.embedder = BailianEmbedder()
        self.store = LocalRetrievalStore(embedder=self.embedder)
        self.sandbox = CodeActSandbox()
        self._citations_cache: list[str] | None = None

    async def close(self) -> None:
        await self.embedder.close()

    async def retrieval_search(
        self, query: str, intent_node: str = "general", top_k: int = 10
    ) -> RetrievalResponse:
        chunks = await self.store.search(query, top_k=top_k)
        return RetrievalResponse(
            chunks=[
                ChunkItem(id=c.id, text=c.text, score=c.score, docId=c.doc_id)
                for c in chunks
            ]
        )

    async def mcp_invoke(self, tool_id: str, params: dict[str, Any]) -> MCPInvokeResponse:
        """tool_id 约定:
        - "code_exec": params={"code": "...python..."} 在沙箱执行
        - 其他:返回未知工具
        """
        if tool_id == "code_exec":
            out = await self.sandbox.run(params.get("code", ""))
            if out["success"]:
                return MCPInvokeResponse(
                    success=True,
                    result=str(out["result"]),
                    structured={"result": out["result"]},
                )
            return MCPInvokeResponse(
                success=False, result=f"sandbox error: {out['error']}", structured=None
            )
        return MCPInvokeResponse(
            success=False, result=f"unknown tool: {tool_id}", structured=None
        )

    async def embedding_encode(self, texts: list[str]) -> dict[str, Any]:
        vectors = await self.embedder.encode(texts)
        return {"vectors": vectors}

    async def list_all_citations(self, kb_id: str | None = None) -> list[str]:
        """从已建检索库里抽取所有法条引用,供红队幻觉检测。
        简化实现:扫描库中文档文本里的《XX》第N条。
        """
        if self._citations_cache is not None:
            return self._citations_cache
        import re

        col = self.store._get_collection()
        try:
            data = col.get()  # 取全部
            docs = data.get("documents", []) or []
        except Exception:
            docs = []
        pat = re.compile(r"《([^》]+)》第([一二三四五六七八九十百千\d]+)条")
        cites = set()
        for d in docs:
            for m in pat.findall(d or ""):
                cites.add(f"《{m[0]}》第{m[1]}条")
        self._citations_cache = sorted(cites)
        return self._citations_cache
