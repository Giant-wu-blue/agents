from __future__ import annotations

import os
import logging

from app.memory.schema import MemoryUnit

logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")


class SharedMemory:
    """跨任务共享记忆库。"""

    def __init__(self, embedder=None):
        self.embedder = embedder           # 复用 LocalProvider.embedder(百炼)
        self._client = None
        self._col = None

    def _col_(self):
        if self._col is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=CHROMA_PATH)
            self._col = self._client.get_or_create_collection(
                name="shared_memory", metadata={"hnsw:space": "cosine"}
            )
        return self._col

    # ── 存储 ───────────────────────────────────────────────────
    async def add(self, unit: MemoryUnit) -> str:
        """把记忆单元编码入库,返回 memory_id。"""
        col = self._col_()
        vec = (await self.embedder.encode([unit.summary]))[0]
        col.add(
            ids=[unit.memory_id],
            embeddings=[vec],
            documents=[unit.summary],
            metadatas=[unit.to_metadata()],
        )
        logger.info(f"[memory] 入库 {unit.memory_id} 来源={unit.source_agent} 主题={unit.task_topic}")
        return unit.memory_id

    async def by_semantic(self, query: str, top_k: int = 5,
                          min_score: float = 0.5) -> list[MemoryUnit]:
        col = self._col_()
        if col.count() == 0:
            return []
        qvec = (await self.embedder.encode([query]))[0]
        res = col.query(query_embeddings=[qvec], n_results=min(top_k, col.count()))
        out = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            score = 1.0 - float(dist)
            if score >= min_score:          # 命中阈值,低于则视为未命中
                out.append(MemoryUnit.from_chroma(doc, meta))
        return out

    def by_keyword(self, kw: str) -> list[MemoryUnit]:
        col = self._col_()
        data = col.get()
        out = []
        for doc, meta in zip(data.get("documents", []), data.get("metadatas", [])):
            if kw in (doc or "") or kw in meta.get("task_topic", ""):
                out.append(MemoryUnit.from_chroma(doc, meta))
        return out

    def by_tag(self, tag: str) -> list[MemoryUnit]:
        col = self._col_()
        data = col.get()
        out = []
        for doc, meta in zip(data.get("documents", []), data.get("metadatas", [])):
            tags = meta.get("tags", "").split(",")
            if tag in tags:
                out.append(MemoryUnit.from_chroma(doc, meta))
        return out


    def clear(self) -> None:
        try:
            self._client.delete_collection("shared_memory")
        except Exception:
            pass
        self._col = None

    def count(self) -> int:
        return self._col_().count()
