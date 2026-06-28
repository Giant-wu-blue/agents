"""SupplyPlanner — 检索并分析供应计划 via ReAct(模板感知版)。

根据 task_template 自动调整分析侧重:
- REGIONAL_SUPPLY → 侧重区域供需全局（作为主要分析产出）
- ZONE_OPTIMIZATION → 侧重片区级别的供应适配
- PARCEL_FEASIBILITY → 侧重供应计划对该地块的约束
"""

from typing import Any

from app.agents.base import BaseReActAgent


class SupplyPlanner(BaseReActAgent):
    name = "SupplyPlanner"
    available_actions = ["SEARCH_KB"]
    max_steps = 5   # 2 次检索 + 1 次分析 + 2 步缓冲

    def __init__(self, java_client=None, llm_client=None):
        super().__init__(java_client=java_client, llm_client=llm_client)

    def build_system_prompt(self, ctx: dict[str, Any], upstream: dict[str, Any]) -> str:
        topic = ctx.get("topic", "")
        region = ctx.get("region", "")
        agent_goal = ctx.get("agent_goal", "分析区域供应计划")
        template = ctx.get("task_template", "")

        if template == "REGIONAL_SUPPLY":
            search_hint = (
                "本任务是区域供需研判，供应分析是核心产出。侧重：区域供应总量/结构/节奏，"
                "供给能力评估（存量/新增/储备），供需匹配度，供应缺口和瓶颈。\n"
            )
        elif template == "ZONE_OPTIMIZATION":
            search_hint = (
                "本任务是片区组合优化。侧重：片区级别的供应计划和时序安排，"
                "多地块供应优先级和协同关系，基础设施配套保障。\n"
            )
        else:
            search_hint = (
                "侧重：区域供应计划总量/结构/导向，目标地块是否符合供应方向，"
                "储备时序和优先级建议。\n"
            )

        return (
            "你是土地供应计划分析专家。\n\n"
            f"# 研究主题\n{topic}\n"
            f"# 目标区域\n{region or '(从主题推断)'}\n"
            f"# 你的任务目标\n{agent_goal}\n"
            f"# 检索侧重\n{search_hint}\n"
            "# ══ 你的最终产出（最重要，先读）══\n"
            "检索完成后，你必须 FINISH 并输出一份「供应计划分析」。\n"
            "检索只是手段，分析才是目的——你必须用自己的话写，不能复制粘贴检索原文。\n\n"
            "## answer 怎么写（直接按这个结构写）\n\n"
            "【供应指标】\n"
            "从资料提炼供应总量、结构、导向等关键指标。例如：\n"
            "\"杭州市 2026 年工矿仓储用地计划供应 698.90 公顷，余杭区新增工业用地"
            "70% 以上须用于先进制造业集群项目。\"\n\n"
            "【匹配性评估】\n"
            "目标地块是否符合供应计划方向，给出判断和理由。\n\n"
            "【时序与优先级】\n"
            "建议储备时序安排和优先级。例如：\n"
            "\"该地块符合产业集聚导向，建议纳入 2026 年第一批次储备计划。\"\n\n"
            "每个判断后标注（来源：文档名）。answer 不少于 200 字。\n\n"
            "# ══ 工作流程 ══\n"
            "最多 2 次 SEARCH_KB（换不同关键词），之后必须 FINISH。\n"
            "第 5 步是硬上限——到了也必须 FINISH。\n\n"
            "# 可用动作\n"
            '- SEARCH_KB: 检索知识库。action_input: {{"query": "供应计划 储备流程 时序", "top_k": 6, "intent": "supply"}}\n'
            '- FINISH: 输出分析报告。action_input: {{"answer": "你的分析报告全文"}}\n\n'
            "⚠️ FINISH 时 answer 如果为空或只有一两句过程描述，你的输出将完全无效，"
            "任务视为失败。请务必写出有实质内容的分析。\n"
        )
