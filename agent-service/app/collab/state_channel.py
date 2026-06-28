from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class StateChannel:
    """Agent 间向量状态直传通道。一个研究任务共享一个实例(放进 ctx)。"""

    def __init__(self, embedder=None):
        self.embedder = embedder
        self._store: dict[str, dict] = {}   # handle -> {vector(np), id, doc_id, text}

    async def publish(self, chunks: list[dict]) -> list[str]:
        """上游把检索证据编码为向量并登记,返回句柄列表。
        chunks: [{"id":..., "text":..., "doc_id":...}, ...]
        """
        if not chunks:
            return []
        texts = [c["text"] for c in chunks]
        vectors = await self.embedder.encode(texts)
        handles = []
        for c, vec in zip(chunks, vectors):
            h = f"vec::{c['id']}"
            self._store[h] = {
                "vector": np.asarray(vec, dtype=np.float32),
                "id": c["id"],
                "doc_id": c.get("doc_id", ""),
                "text": c["text"],   # 保留原文供最终取用,但传递阶段不传它
            }
            handles.append(h)
        logger.info(f"[StateChannel] publish {len(handles)} 个向量句柄")
        return handles

    async def consume(
        self, handles: list[str], query: str, top_k: int = 3
    ) -> list[dict]:
        """下游用 query 在向量空间筛选 handles,返回最相关的 top_k 证据。

        关键:筛选完全在向量空间完成(余弦相似度),
        只有最终选中的少量证据才取出 text,大幅减少传给下游 LLM 的文本量。
        """
        items = [self._store[h] for h in handles if h in self._store]
        if not items:
            return []
        qvec = np.asarray((await self.embedder.encode([query]))[0], dtype=np.float32)
        qn = qvec / (np.linalg.norm(qvec) + 1e-8)

        scored = []
        for it in items:
            v = it["vector"]
            vn = v / (np.linalg.norm(v) + 1e-8)
            sim = float(np.dot(qn, vn))
            scored.append((sim, it))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {"id": it["id"], "doc_id": it["doc_id"], "text": it["text"], "score": sim}
            for sim, it in scored[:top_k]
        ]

    def transfer_stats(self, handles: list[str], dim: int = 1024) -> dict:
        n = len([h for h in handles if h in self._store])
        return {"vector_transfers": n, "vector_bytes": n * dim * 4}  # float32=4B/维
