from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from app.compression.protector import contains_protected

if TYPE_CHECKING:
    from app.clients.java_client import JavaClient


async def extractive_compress(
    text: str,
    topic: str,
    java_client: JavaClient,
    ratio: float = 0.5,
) -> str:
    """Keep top-ranked sentences within a chunk, protecting legal citations."""
    sentences = [s.strip() for s in re.split(r"[。!?\n]+", text) if len(s.strip()) > 5]
    if len(sentences) <= 3:
        return text

    # Sentences containing protected content are unconditionally kept
    forced = [i for i, s in enumerate(sentences) if contains_protected(s)]

    # Rank remaining sentences by cosine similarity to topic
    resp = await java_client.embedding_encode([topic] + sentences)
    vectors = resp["vectors"]
    topic_vec = np.array(vectors[0])
    scores = [
        float(
            np.dot(topic_vec, np.array(v))
            / (np.linalg.norm(topic_vec) * np.linalg.norm(v) + 1e-8)
        )
        for v in vectors[1:]
    ]

    n_keep = max(len(forced), int(len(sentences) * ratio))
    ranked = sorted(range(len(sentences)), key=lambda i: -scores[i])
    keep_idx = set(forced) | set(ranked[:n_keep])

    return "。".join(sentences[i] for i in sorted(keep_idx)) + "。"
