from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from app.clients.java_client import JavaClient


async def refilter_by_report_topic(
    chunks: list[dict],
    report_topic: str,
    java_client: JavaClient,
    threshold: float = 0.35,
) -> list[dict]:
    """Re-filter chunks by cosine similarity to the final report topic.

    Protected chunks (statute citations, key numbers) bypass the filter.
    """
    if not chunks:
        return []

    texts = [report_topic] + [c["text"] for c in chunks]
    resp = await java_client.embedding_encode(texts)
    vectors = resp["vectors"]
    topic_vec = np.array(vectors[0])

    kept: list[dict] = []
    for c, v in zip(chunks, vectors[1:]):
        if c.get("_protected"):
            kept.append(c)
            continue
        sim = float(
            np.dot(topic_vec, v)
            / (np.linalg.norm(topic_vec) * np.linalg.norm(v) + 1e-8)
        )
        if sim >= threshold:
            c["_report_sim"] = sim
            kept.append(c)

    return kept
