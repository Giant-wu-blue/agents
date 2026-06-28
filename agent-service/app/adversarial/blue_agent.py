from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.schemas import AttackResult

if TYPE_CHECKING:
    from app.clients.llm_client import LLMClient

logger = logging.getLogger(__name__)

BLUE_AGENT_PROMPT = """你是土地储备研究报告的质量修复专家。审查官指出了以下问题，请逐条修复。

# 当前报告
{report}

# 审查官发现的问题
{attacks}

# 修复动作说明
- ADD:  在该论断后补充 [chunk_id] 引用（若未提供 evidence_chunk_id 则用 [需补充引用] 占位）
- DELETE: 完全删除该论断
- MODIFY: 修正错误的数据/法条引用
- VERIFY: 如果审查官的问题可能存在但你不确定，标注"[待核实]"而非删除


# 任务
逐条处理每个问题，输出修复后的完整报告。对于每个修复，在报告内用注释标注修复类型。
保持报告整体结构和可读性。


# 注意
- 只动被攻击的内容，其他保持原样
- 修复后报告整体结构、章节顺序不变
- 直接输出修复后的报告全文（纯 markdown，不要任何解释/前言/总结）"""


class BlueAgent:
    """Blue-team agent that repairs a report based on Red Agent's findings."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def repair(self, report: str, attack_result: AttackResult) -> str:
        if not attack_result.attacks:
            return report

        attacks_str = "\n".join(
            f"{i+1}. [{a.dim}] 原文:'{a.claim}' | 问题:{a.issue} | 动作:{a.suggested_action}"
            + (f" | 应引用:{a.evidence_chunk_id}" if a.evidence_chunk_id else "")
            for i, a in enumerate(attack_result.attacks)
        )

        prompt = BLUE_AGENT_PROMPT.format(report=report, attacks=attacks_str)

        try:
            repaired = await self.llm_client.chat(prompt, use_local=False, temperature=0.3)
            return repaired.strip()
        except Exception as e:
            logger.error(f"Blue Agent repair failed: {e}")
            return report


async def blue_agent_repair(
    report: str,
    attack_result: AttackResult,
    llm_client: LLMClient,
) -> str:
    """Backward-compatible wrapper called by the adversarial loop."""
    agent = BlueAgent(llm_client)
    return await agent.repair(report, attack_result)
