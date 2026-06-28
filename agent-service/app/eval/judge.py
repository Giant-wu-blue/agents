from app.clients.llm_client import LLMClient
from app.schemas import JudgeScore

JUDGE_PROMPT = """你是土地储备研究报告的资深评审。从 5 个维度对下面报告评分（0-10）。

# 研究主题
{topic}

# 标准要点（必须覆盖）
{gold_facts}

# 待评审报告
{report}

# 评分维度
1. completeness: 是否覆盖政策/地块/供应/成本所有维度
2. accuracy: 关键事实是否与标准要点一致
3. traceability: 是否有明确的引用支撑
4. coherence: 内部逻辑是否自洽
5. actionability: 结论是否可执行（对储备决策有指导意义）

# 输出严格 JSON
{{"completeness": 8, "accuracy": 7, "traceability": 9, "coherence": 8, "actionability": 7, "reason": "一句话总结"}}"""


async def judge_report(
    report: str,
    topic: str,
    gold_facts: list[str],
    llm_client: LLMClient,
) -> JudgeScore:
    """Score a report using local LLM judge (14B)."""
    import json
    import re

    facts_text = "\n".join(f"- {f}" for f in gold_facts) if gold_facts else "(无标准要点)"
    prompt = JUDGE_PROMPT.format(topic=topic, gold_facts=facts_text, report=report)

    raw = await llm_client.chat(prompt, use_local=True)

    # Robust JSON extraction
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                # Fallback: return mid-range scores
                return JudgeScore(
                    completeness=5, accuracy=5, traceability=5,
                    coherence=5, actionability=5,
                    reason=f"Judge JSON parse failed: {raw[:100]}",
                )
        else:
            return JudgeScore(
                completeness=5, accuracy=5, traceability=5,
                coherence=5, actionability=5,
                reason=f"Judge JSON parse failed: {raw[:100]}",
            )

    return JudgeScore(**data)
