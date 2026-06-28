from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.clients.java_client import JavaClient


async def deduplicate(
    chunks: list[dict],
    java_client: JavaClient,
    sim_threshold: float = 0.92,
) -> list[dict]:
    """Remove duplicate and near-duplicate chunks across agents."""
    # L1: Exact dedup by chunk id
    seen: set[str] = set()
    unique: list[dict] = []
    for c in chunks:
        cid = c.get("id", c.get("chunk_id", ""))
        if cid and cid not in seen:
            seen.add(cid)
            unique.append(c)
        elif not cid:
            unique.append(c)  # no id → can't dedup, keep it

    if len(unique) <= 1:
        return unique

    # L2: Semantic dedup via embedding cosine similarity
    texts = [c["text"] for c in unique]
    resp = await java_client.embedding_encode(texts)
    vectors = resp["vectors"]
    vecs = [np.array(v) for v in vectors]

    keep: list[dict] = []
    removed: set[int] = set()
    for i in range(len(unique)):
        if i in removed:
            continue
        keep.append(unique[i])
        for j in range(i + 1, len(unique)):
            if j in removed:
                continue
            sim = float(
                np.dot(vecs[i], vecs[j])
                / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-8)
            )
            if sim > sim_threshold:
                removed.add(j)

    return keep
