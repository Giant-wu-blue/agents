from typing import Any

from app.agents.base import BaseReActAgent


class CostEstimator(BaseReActAgent):
    name = "CostEstimator"
    available_actions = []    # 纯综合，无外部动作
    max_steps = 1             # 直接生成，无需迭代

    def __init__(self, llm_client=None):
        super().__init__(java_client=None, llm_client=llm_client)

    async def __call__(self, ctx: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
        """重载:无可用动作,跳过 ReAct JSON 循环,直接生成报告。

        原 ReAct 循环要求 LLM 输出 JSON 格式的 FINISH,对无工具 agent
        完全是多余的格式开销——LLM 被迫输出 JSON,容易产出"我将开始分析..."
        之类的描述而非报告正文。直接生成绕过 JSON,讲人话即可。
        """
        llm = self.llm_client
        if llm is None:
            return {"content": "", "citations": [], "error": "llm_client not injected"}

        prompt = self.build_system_prompt(ctx, upstream)
        # 直接让 LLM 输出报告,不走 JSON
        full_prompt = (
            prompt
            + "\n\n请直接输出你的分析报告正文。不要输出 JSON，不要写"
            "'我将开始'、'我已经获取了'之类的过程描述——从报告标题开始直接写内容。"
        )
        response = await llm.chat(full_prompt, temperature=0.2, max_tokens=4096)
        return {"content": response, "citations": [], "react_steps": 0}

    def build_system_prompt(self, ctx: dict[str, Any], upstream: dict[str, Any]) -> str:
        topic = ctx.get("topic", "")
        template = ctx.get("task_template", "PARCEL_FEASIBILITY")
        agent_goal = ctx.get("agent_goal", "")

        upstream_text = ctx.get("upstream_rendered", "")
        if not upstream_text:
            upstream_text = self._extract_upstream(upstream)

        role_desc, output_spec = self._template_prompt(template, agent_goal)

        return (
            f"{role_desc}\n\n"
            f"# 研究主题\n{topic}\n"
            f"# 你的任务\n{agent_goal or '综合上游分析结果'}\n\n"
            f"# 上游分析结果\n{upstream_text}\n\n"
            f"# 可用动作\n"
            f'- FINISH: 输出最终报告。action_input: {{"answer": "报告全文"}}\n\n'
            f"# ══ 关键要求 ══\n"
            f"你只有一步——直接 FINISH，在 answer 里输出完整报告。\n"
            f"禁止写'我将开始'、'我已经获取了上游资料，可以整合输出'等过程描述。\n"
            f"那些不是分析——你必须直接写出分析结论和具体数字。\n"
            f"⚠️ 如果 answer 为空或只有过程描述，任务视为失败。\n\n"
            f"{output_spec}\n"
        )

    @staticmethod
    def _extract_upstream(upstream: dict[str, Any]) -> str:
        parts = []
        for key, label in [("policy", "政策合规分析"), ("parcel", "地块现状分析"),
                           ("supply", "供应计划分析"), ("tool", "工具计算结果")]:
            entry = upstream.get(key, {})
            if isinstance(entry, dict):
                content = entry.get("content", "")
            else:
                content = str(entry) if entry else ""
            if content:
                parts.append(f"## {label}\n{content}")
        return "\n\n".join(parts) if parts else "(上游数据暂缺)"

    @staticmethod
    def _template_prompt(template: str, goal: str) -> tuple[str, str]:
        """返回 (角色描述, 输出规范) —— 按研究模板切换。"""
        if template == "REGIONAL_SUPPLY":
            return (
                "你是土地储备供需分析专家。综合上游政策解读和供应计划分析，"
                "输出区域供需研判报告。",
                "## 报告结构（直接在 answer 里按此结构写）\n\n"
                "【供应能力评估】\n"
                "区域当前土地供应总量、结构、节奏，引用上游数据。\n\n"
                "【需求判断】\n"
                "结合政策导向和区域发展趋势判断需求。\n\n"
                "【供需匹配度】\n"
                "供应是否能满足需求，缺口在哪里。\n\n"
                "【建议】\n"
                "供应计划优化方向和优先级。\n\n"
                "不少于 300 字，包含具体数据，直接写分析内容，禁止过程描述。\n"
            )
        elif template == "COST_COMPARISON":
            return (
                "你是土地储备成本对比分析专家。综合上游政策基准和成本数据，"
                "输出多类型用地成本对比报告。",
                "## 报告结构（直接在 answer 里按此结构写）\n\n"
                "【各类型用地成本结构对比】\n"
                "工业/商业/住宅等类型的取得、补偿、税费差异对比。\n\n"
                "【成本差异原因分析】\n"
                "政策、区位、用途等因素如何影响成本。\n\n"
                "【经济可行性排序】\n"
                "按储备性价比排列不同类型用地。\n\n"
                "【建议】\n"
                "优先储备哪种类型及理由。\n\n"
                "不少于 300 字，包含具体对比数字，直接写分析内容，禁止过程描述。\n"
            )
        elif template == "ZONE_OPTIMIZATION":
            return (
                "你是土地储备片区优化专家。综合上游政策、多地块分析和供应计划，"
                "输出片区组合储备优化方案。",
                "## 报告结构（直接在 answer 里按此结构写）\n\n"
                "【多地块综合评估】\n"
                "各地块储备条件、成本、时序的横向比较。\n\n"
                "【组合策略】\n"
                "建议的组合方式、分期安排和优先级。\n\n"
                "【总成本估算】\n"
                "组合方案的总投资估算和每亩成本。\n\n"
                "【优化效果】\n"
                "相比单独储备的成本节约和风险分散效果。\n\n"
                "不少于 300 字，包含具体数字对比，直接写分析内容，禁止过程描述。\n"
            )
        else:  # PARCEL_FEASIBILITY（默认）
            return (
                "你是土地储备成本测算专家。综合上游政策、地块、供应分析，"
                "输出完整的成本测算报告。",
                "## 报告结构（直接在 answer 里按此结构写）\n\n"
                "【成本构成测算】\n"
                "逐项列出（土地取得补偿、附着物补偿、社保资金、前期开发、税费等），"
                "每项给出金额或单价区间。例：征地区片综合地价 28-52 万元/亩。\n\n"
                "【总成本汇总】\n"
                "给出每亩综合成本估算区间。\n\n"
                "【成本合理性评估】\n"
                "对比基准，判断是否合理。\n\n"
                "【假设与不确定性】\n"
                "标注测算的前提假设和数据缺口。\n\n"
                "不少于 300 字，包含具体数字，直接写分析内容，禁止过程描述。\n"
            )
