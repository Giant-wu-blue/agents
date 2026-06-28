from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.adversarial.json_fallback import parse_llm_json
from app.schemas import AttackResult

if TYPE_CHECKING:
    from app.clients.llm_client import LLMClient

logger = logging.getLogger(__name__)

RED_PROMPT_TEMPLATE = """你是土地储备领域的严苛审查官，任务是攻击下面的研究报告，找出所有问题。

# 待审查报告
{report}

# 可验证的知识库证据（只列了与报告相关的 chunk）
{evidence}

# 你的攻击维度（每条问题必须标注维度）
1. FACTUAL（事实性）:
   - 报告中引用的"《XX》第N条"，在证据中能否找到对应文本？找不到就是编造的
   - 报告中的数字（面积/金额/比例），证据中能否核对？
2. LOGICAL（逻辑一致性）:
   - 报告内部是否自相矛盾？（例如成本数字 ≠ 面积×单价）
3. CITATION（引用质量）:
   - 每个关键结论后是否标了 [chunk_id]？无引用的论断算不合格

# 输出严格 JSON，不要任何解释、不要 markdown 围栏
{{
  "attacks": [
    {{
      "dim": "FACTUAL",
      "claim": "报告中的原话（<=50字）",
      "issue": "具体问题（<=80字）",
      "suggested_action": "DELETE",
      "evidence_chunk_id": null
    }}
  ],
  "overall_score": 70
}}

# 注意
- 不要把"措辞优化"当成 attack，只攻击事实/逻辑/引用问题
- 如果报告没问题，attacks 数组可以为空，overall_score 给 95 以上
"""

SCHEMA_HINT = (
    '{"attacks": [{"dim":"FACTUAL|LOGICAL|CITATION","claim":"...","issue":"...",'
    '"suggested_action":"DELETE|MODIFY|VERIFY|ADD","evidence_chunk_id":"..."}], '
    '"overall_score": 0-100}'
)


class RedAgent:
    """Red-team agent that audits a report for hallucinations."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def attack(self, report: str, evidence_chunks: list[dict]) -> AttackResult:
        """Audit the report and return structured attack findings."""
        evidence_str = "\n".join(
            f"[{c.get('id', '?')}] {c.get('text', '')[:300]}" for c in evidence_chunks[:15]
        )
        prompt = RED_PROMPT_TEMPLATE.format(report=report, evidence=evidence_str)

        raw = await self.llm_client.chat(prompt, use_local=True, temperature=0.2)

        parsed = await parse_llm_json(raw, self.llm_client, schema_hint=SCHEMA_HINT)
        if parsed is None:
            logger.warning("Red Agent JSON parse all layers failed, returning empty attack")
            return AttackResult(attacks=[], overall_score=50)

        try:
            return AttackResult(**parsed)
        except Exception as e:
            logger.warning(f"Red Agent pydantic validation failed: {e}")
            return AttackResult(attacks=[], overall_score=50)


async def red_agent_attack(
    report: str,
    retrieved_chunks: list[dict],
    llm_client: LLMClient,
) -> AttackResult:
    """Backward-compatible wrapper called by the adversarial loop."""
    agent = RedAgent(llm_client)
    return await agent.attack(report, retrieved_chunks)
