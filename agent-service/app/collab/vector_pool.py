from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class SharedVectorPool:
    """跨 agent 共享的证据向量池。一个研究任务一个实例,放进 ctx。"""

    def __init__(self, embedder=None):
        self.embedder = embedder
        self._pool: dict[str, dict] = {}    # handle -> {vector(np), id, doc_id, text, source}
        self._access_log: list[dict] = []   # 取用日志,用于统计"按需取用"效果

    async def publish(self, chunks: list[dict], source_agent: str = "retriever") -> list[str]:
        """检索 agent 把证据编码入池。返回句柄列表。"""
        if not chunks:
            return []
        vectors = await self.embedder.encode([c["text"] for c in chunks])
        handles = []
        for c, vec in zip(chunks, vectors):
            h = f"vec::{c['id']}"
            self._pool[h] = {
                "vector": np.asarray(vec, dtype=np.float32),
                "id": c["id"], "doc_id": c.get("doc_id", ""),
                "text": c["text"], "source": source_agent,
            }
            handles.append(h)
        logger.info(f"[pool] {source_agent} 入池 {len(handles)} 条证据,池总量={len(self._pool)}")
        return handles

    async def fetch(self, query: str, top_k: int = 3,
                    requester: str = "") -> list[dict]:
        """下游 agent 用自己的子任务作 query,从池中取最相关的 top_k 条原文。

        关键:不同 agent 用不同 query → 各取不同子集(按需取用),
        这是相比 TEXT"一份摘要喂所有人"的核心优化。
        """
        if not self._pool:
            return []
        qvec = np.asarray((await self.embedder.encode([query]))[0], dtype=np.float32)
        qn = qvec / (np.linalg.norm(qvec) + 1e-8)

        scored = []
        for item in self._pool.values():
            v = item["vector"]
            sim = float(np.dot(qn, v / (np.linalg.norm(v) + 1e-8)))
            scored.append((sim, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = scored[:top_k]

        self._access_log.append({
            "requester": requester, "query": query[:40],
            "picked": [it["id"] for _, it in picked],
        })
        return [{"id": it["id"], "doc_id": it["doc_id"],
                 "text": it["text"], "score": s} for s, it in picked]

    def stats(self, dim: int = 1024) -> dict:
        n = len(self._pool)
        distinct_subsets = len({tuple(sorted(log["picked"])) for log in self._access_log})
        return {
            "pool_size": n,
            "vector_bytes": n * dim * 4,
            "fetch_count": len(self._access_log),
            "distinct_subsets": distinct_subsets,   # 不同下游取了几种不同子集
        }

    def access_log(self) -> list[dict]:
        return self._access_log
