from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.compression.compressor import extractive_compress
from app.compression.deduplicator import deduplicate
from app.compression.protector import contains_protected
from app.compression.relevance_refilter import refilter_by_report_topic

if TYPE_CHECKING:
    from app.clients.java_client import JavaClient

logger = logging.getLogger(__name__)


class AggregationStats:
    """Quantitative metrics for evaluation / debugging."""

    input_chunks: int = 0
    after_dedup: int = 0
    after_refilter: int = 0
    protected_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    compression_triggered: bool = False

    def to_dict(self) -> dict:
        """供响应 metadata / 评测使用的可序列化统计。"""
        saved = self.input_tokens - self.output_tokens
        return {
            "input_chunks": self.input_chunks,
            "after_dedup": self.after_dedup,
            "after_refilter": self.after_refilter,
            "protected_count": self.protected_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_saved": saved,
            "saved_pct": round(saved / self.input_tokens * 100, 1) if self.input_tokens else 0.0,
            "compression_triggered": self.compression_triggered,
        }


def estimate_tokens(text: str) -> int:
    """Rough token estimate for Chinese text (~1.3 chars per token)."""
    return int(len(text) * 1.3)


async def aggregate(
    chunks: list[dict],
    report_topic: str,
    java_client: JavaClient,
    budget_tokens: int = 6000,
) -> tuple[str, AggregationStats]:
    """Assemble and compress multi-agent results into a single context string.

    Args:
        chunks: Evidence chunks collected from all agents
                (each is {"id": ..., "text": ..., "doc_id": ...}).
        report_topic: The unified research topic for relevance scoring.
        java_client: HTTP client for Java embedding API.
        budget_tokens: Target token budget for the assembled context.

    Returns:
        (assembled_context_string, stats)
    """
    stats = AggregationStats()
    stats.input_chunks = len(chunks)
    stats.input_tokens = estimate_tokens("\n".join(c["text"] for c in chunks))

    # ① Protect — mark legally sensitive chunks FIRST
    for c in chunks:
        c["_protected"] = contains_protected(c["text"])
    stats.protected_count = sum(1 for c in chunks if c["_protected"])

    # ② Dedup — cross-agent exact + semantic deduplication
    chunks = await deduplicate(chunks, java_client)
    stats.after_dedup = len(chunks)

    # ③ Re-filter — re-score against the unified report topic
    chunks = await refilter_by_report_topic(chunks, report_topic, java_client)
    stats.after_refilter = len(chunks)

    # ④ Compress — extractive compression only if over budget
    current = "\n".join(c["text"] for c in chunks)
    if estimate_tokens(current) > budget_tokens:
        stats.compression_triggered = True
        for c in chunks:
            if c["_protected"]:
                c["text"] = c["text"]  # protected chunks kept verbatim
            else:
                c["text"] = await extractive_compress(c["text"], report_topic, java_client)
        result = "\n".join(c["text"] for c in chunks)
    else:
        result = current

    stats.output_tokens = estimate_tokens(result)
    logger.info(
        f"Aggregation: {stats.input_chunks}→{stats.after_refilter} chunks, "
        f"{stats.input_tokens}→{stats.output_tokens} tokens, "
        f"compress={'Y' if stats.compression_triggered else 'N'}"
    )
    # 返回:(拼接字符串, 统计, 精简后的 chunk 列表)
    # 精简列表用于下游对抗审查 —— 让"压缩"真正减少喂给后续环节的 token。
    return _format_output(chunks, result), stats, chunks


def _format_output(chunks: list[dict], text: str) -> str:
    return text
