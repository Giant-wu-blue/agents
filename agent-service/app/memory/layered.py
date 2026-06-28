from __future__ import annotations

import os
import time
import asyncio
import logging

from app.memory.schema import MemoryUnit
from app.memory.backends import MemoryBackend, ChromaBackend
from app.memory.security import desensitize, check_access

logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")


# ── 短期工作记忆:任务级,易失,无需持久化 ──────────────────────
class ShortTermMemory:
    """单个任务内 agent 间共享的临时记忆。任务结束即清空。"""

    def __init__(self):
        self._items: list[MemoryUnit] = []

    def add(self, unit: MemoryUnit) -> None:
        self._items.append(unit)

    def all(self) -> list[MemoryUnit]:
        return list(self._items)

    def find(self, keyword: str) -> list[MemoryUnit]:
        return [u for u in self._items
                if keyword in u.summary or keyword in u.task_topic]

    def clear(self) -> None:
        self._items.clear()

    def count(self) -> int:
        return len(self._items)


class LayeredMemory:
    """短期 + 长期双层记忆,带并发锁、脱敏、访问控制、可插拔后端。"""

    def __init__(self, embedder=None, backend: MemoryBackend | None = None):
        self.embedder = embedder
        self.backend = backend or ChromaBackend(CHROMA_PATH)
        self.short_term = ShortTermMemory()
        self._write_lock = asyncio.Lock()   # 长期记忆写并发锁

    def remember_short(self, unit: MemoryUnit) -> None:
        self.short_term.add(unit)

    async def remember_long(self, unit: MemoryUnit) -> str:
        unit.summary = desensitize(unit.summary)        # 脱敏
        vec = (await self.embedder.encode([unit.summary]))[0]
        async with self._write_lock:                    # 并发安全
            await self.backend.upsert(
                unit.memory_id, vec, unit.summary, unit.to_metadata()
            )
        logger.info(f"[memory:long] 固化 {unit.memory_id} 来源={unit.source_agent}")
        return unit.memory_id

    async def consolidate(self, min_len: int = 50) -> int:
        """把短期记忆中有长期价值的(摘要够长的)固化进长期层。返回固化条数。"""
        n = 0
        for unit in self.short_term.all():
            if len(unit.summary) >= min_len:
                await self.remember_long(unit)
                n += 1
        self.short_term.clear()   # 固化后清空短期
        logger.info(f"[memory] 固化 {n} 条短期记忆 → 长期")
        return n

    async def recall_semantic(self, query: str, top_k: int = 5,
                              min_score: float = 0.5,
                              requester_role: str = "") -> list[MemoryUnit]:
        if self.backend.count() == 0:
            return []
        qvec = (await self.embedder.encode([query]))[0]
        rows = await self.backend.query(qvec, top_k)
        out = []
        for r in rows:
            score = 1.0 - r["distance"]
            if score < min_score:
                continue
            unit = MemoryUnit.from_chroma(r["document"], r["metadata"])
            scope = r["metadata"].get("access_scope", "")
            scope_list = [s for s in scope.split(",") if s] if scope else []
            if check_access(scope_list, requester_role):   # 访问控制
                out.append(unit)
        return out

    def recall_keyword(self, kw: str) -> list[MemoryUnit]:
        rows = self.backend.get_all()
        return [MemoryUnit.from_chroma(r["document"], r["metadata"])
                for r in rows
                if kw in (r["document"] or "") or kw in r["metadata"].get("task_topic", "")]

    def recall_tag(self, tag: str) -> list[MemoryUnit]:
        rows = self.backend.get_all()
        out = []
        for r in rows:
            tags = r["metadata"].get("tags", "").split(",")
            if tag in tags:
                out.append(MemoryUnit.from_chroma(r["document"], r["metadata"]))
        return out

    def clear_long(self) -> None:
        self.backend.clear()

    def stats(self) -> dict:
        return {"short_term": self.short_term.count(), "long_term": self.backend.count()}
