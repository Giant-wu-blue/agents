from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
RERANK_MODEL = os.getenv("BAILIAN_RERANK_MODEL", "gte-rerank-v2")


def rerank_enabled() -> bool:
    return os.getenv("ENABLE_RERANK", "1") not in ("0", "false", "False", "")


async def rerank(
    query: str,
    docs: list[str],
    top_n: int | None = None,
    timeout: float = 10.0,
) -> list[int] | None:
    """对 docs 按与 query 的相关性重排。

    Args:
        query: 查询文本
        docs:  候选文档文本列表
        top_n: 返回前 N 个（默认全部）

    Returns:
        重排后的【原始下标】列表（例如 [3,0,1,...]），
        失败时返回 None（调用方应据此保持原顺序）。
    """
    if not rerank_enabled() or not docs:
        return None
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key or api_key == "sk-placeholder":
        return None

    payload = {
        "model": RERANK_MODEL,
        "input": {"query": query, "documents": docs},
        "parameters": {"return_documents": False, "top_n": top_n or len(docs)},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(RERANK_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        # 返回结构: {"output": {"results": [{"index": i, "relevance_score": s}, ...]}}
        results = (data.get("output") or {}).get("results") or []
        order = [r["index"] for r in results if "index" in r]
        if not order:
            return None
        logger.info(f"[rerank] {len(docs)} 候选 → 重排完成，返回 {len(order)} 项")
        return order
    except Exception as e:
        logger.warning(f"[rerank] 重排失败，降级为原顺序: {e}")
        return None
