from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.clients.llm_client import LLMClient

logger = logging.getLogger(__name__)


class JsonParseStats:
    """Per-layer hit-rate stats for evaluation."""

    l1_success: int = 0
    l2_success: int = 0
    l3_success: int = 0
    total_failed: int = 0

    @classmethod
    def reset(cls) -> None:
        cls.l1_success = cls.l2_success = cls.l3_success = cls.total_failed = 0

    @classmethod
    def total_called(cls) -> int:
        return cls.l1_success + cls.l2_success + cls.l3_success + cls.total_failed


async def parse_llm_json(raw: str, llm_client: LLMClient, schema_hint: str = "") -> dict | None:
    """Three-layer JSON parse fallback. Returns dict or None if all three layers fail."""

    # === L1: Strict json.loads ===
    try:
        result = json.loads(raw)
        JsonParseStats.l1_success += 1
        return result
    except json.JSONDecodeError:
        pass

    # === L2: Regex extract + clean common LLM artifacts ===
    cleaned = _clean_json_block(raw)
    if cleaned:
        try:
            result = json.loads(cleaned)
            JsonParseStats.l2_success += 1
            return result
        except json.JSONDecodeError:
            pass

    # === L3: LLM self-repair (local 7B, cost-efficient) ===
    fixed_raw = await _ask_llm_to_fix(raw, llm_client, schema_hint)
    if fixed_raw:
        try:
            result = json.loads(fixed_raw)
            JsonParseStats.l3_success += 1
            return result
        except json.JSONDecodeError:
            pass

    JsonParseStats.total_failed += 1
    logger.warning(f"JSON parse 3-layer fallback all failed. raw[:300]={raw[:300]}")
    return None


def _clean_json_block(raw: str) -> str | None:
    """L2: extract outermost JSON block + clean common LLM artifacts."""
    # 1. Strip markdown fences
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)

    # 2. Extract outermost brace block (non-greedy)
    m = re.search(r"\{[\s\S]*?\}", raw)
    if not m:
        return None
    block = m.group(0)

    # 3. Clean common LLM JSON mistakes
    # 3.1 Trailing comma before } or ]  →  {"a":1,}  →  {"a":1}
    block = re.sub(r",(\s*[}\]])", r"\1", block)
    # 3.2 // line comments (LLMs occasionally add these)
    block = re.sub(r"//[^\n]*", "", block)

    return block


async def _ask_llm_to_fix(broken: str, llm_client: LLMClient, schema_hint: str) -> str | None:
    """L3: feed broken JSON back to LLM for self-repair (local 7B, cost-efficient)."""
    prompt = (
        "下面是一段不合法的 JSON，请只输出修复后的合法 JSON，不要任何解释。\n\n"
        f"# 期望格式\n{schema_hint or '(任意合法 JSON 对象)'}\n\n"
        f"# 原始内容\n{broken[:2000]}\n\n"
        "# 输出严格 JSON:"
    )
    try:
        return await llm_client.chat(prompt, use_local=True, temperature=0.0)
    except Exception as e:
        logger.error(f"L3 LLM fix failed: {e}")
        return None
