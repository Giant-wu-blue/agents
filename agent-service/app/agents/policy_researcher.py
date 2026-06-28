"""PolicyResearcher — 检索并分析政策法规 via ReAct(模板感知版)。

根据 task_template 自动调整检索侧重:
- COST_COMPARISON → 侧重成本基准与收费标准
- POLICY_INTERPRET → 侧重法规解读与适用条件
- 其余模板 → 通用政策法规检索
"""

from typing import Any

from app.agents.base import BaseReActAgent


class PolicyResearcher(BaseReActAgent):
    name = "PolicyResearcher"
    available_actions = ["SEARCH_KB"]
    max_steps = 5   # 2 次检索 + 1 次分析 + 2 步缓冲

    def __init__(self, java_client=None, llm_client=None):
        super().__init__(java_client=java_client, llm_client=llm_client)

    def build_system_prompt(self, ctx: dict[str, Any], upstream: dict[str, Any]) -> str:
        topic = ctx.get("topic", "")
        agent_goal = ctx.get("agent_goal", "检索相关政策法规")
        template = ctx.get("task_template", "")
        search_hint = self._search_hint(template)

        return (
            "你是土地储备政策研究专家。\n\n"
            f"# 研究主题\n{topic}\n"
            f"# 你的任务目标\n{agent_goal}\n"
            f"{search_hint}\n"
            "# ══ 你的最终产出（最重要，先读）══\n"
            "检索完成后，你必须 FINISH 并输出一份「政策合规分析」。\n"
            "这是你作为专家的核心价值——检索只是手段，分析才是目的。\n\n"
            "## answer 怎么写（直接按这个结构写，不要再列清单）\n\n"
            "【合规要点】\n"
            "从检索资料提炼 3-5 条关键合规要求，每条一两句话。例如：\n"
            "\"根据《浙江省土地储备管理办法》第十二条，工业用地储备须满足产业规划符合性"
            "和土地利用总体规划一致性，储备机构应开展前期调查核实权属与用途。\"\n\n"
            "【政策风险】\n"
            "指出本任务面临的政策风险，1-2 条。例如：\n"
            "\"若地块现状用途与规划用途不一致，收储审批可能延期 3-6 个月。\"\n\n"
            "【结论】\n"
            "一句话总结政策合规判断。\n\n"
            "每条分析后标注（来源：文档名）。answer 不少于 200 字。\n\n"
            "# ══ 工作流程 ══\n"
            "最多 2 次 SEARCH_KB（换不同关键词），之后必须 FINISH。\n"
            "第 5 步是硬上限——到了也必须 FINISH。\n\n"
            "# 可用动作\n"
            '- SEARCH_KB: 检索知识库。action_input: {{"query": "检索词", "top_k": 8, "intent": "policy"}}\n'
            '- FINISH: 输出分析报告。action_input: {{"answer": "你的分析报告全文"}}\n\n'
            "⚠️ FINISH 时 answer 如果为空或只有一两句过程描述（如'我将开始分析'），"
            "你的输出将完全无效，任务视为失败。请务必写出有实质内容的分析。\n"
        )

    @staticmethod
    def _search_hint(template: str) -> str:
        if template == "COST_COMPARISON":
            return (
                "# 检索侧重\n"
                "本任务是成本对比分析，请侧重检索：\n"
                "- 不同用地类型的成本基准和收费标准\n"
                "- 各类补偿标准和税费政策\n"
                "- 各类型用地的储备经济性比较数据\n"
            )
        if template == "POLICY_INTERPRET":
            return (
                "# 检索侧重\n"
                "本任务是纯政策解读，请侧重检索：\n"
                "- 法规的适用范围和条件\n"
                "- 政策条款的具体含义和解释\n"
                "- 相关配套规定和实施细则\n"
            )
        return ""
