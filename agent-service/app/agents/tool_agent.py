from typing import Any

from app.agents.base import BaseReActAgent


class ToolAgent(BaseReActAgent):
    name = "ToolAgent"
    available_actions = ["CALL_TOOL"]
    max_steps = 5   # 1-2 次代码执行 + 1 次分析 + 缓冲

    def __init__(self, java_client=None, llm_client=None):
        super().__init__(java_client=java_client, llm_client=llm_client)

    def build_system_prompt(self, ctx: dict[str, Any], upstream: dict[str, Any]) -> str:
        topic = ctx.get("topic", "")
        subtask = ctx.get("agent_goal", ctx.get("current_subtask", topic))
        upstream_text = ctx.get("upstream_rendered", "")

        return (
            "你是工具执行智能体。当任务需要数值计算、数据处理或公式推算时,"
            "你编写 Python 代码并在安全沙箱中运行,用计算结果来回答。\n\n"
            f"# 当前子任务\n{subtask}\n\n"
            f"# 可参考的上游结果\n{upstream_text or '(无)'}\n\n"
            "# ══ 你的最终产出（最重要，先读）══\n"
            "代码执行后，你必须 FINISH 并输出分析结论。\n"
            "代码只是手段——你必须在 answer 里解释计算结果、给出结论。\n\n"
            "## answer 怎么写\n"
            "【计算过程】简述你做了什么计算。\n"
            "【结果】列出具体数值。\n"
            "【结论】用一两句话解释这个结果对研究主题的意义。\n\n"
            "# ══ 工作流程 ══\n"
            "1. 分析子任务需要什么计算\n"
            "2. 用 CALL_TOOL 执行代码（最多 2 次）\n"
            '   action_input: {{"tool_id": "code_exec", "params": {{"code": "..."}}}}\n'
            "   代码中必须把最终结果赋值给变量 result\n"
            "3. 拿到结果后 FINISH，给出带计算依据的答案\n"
            "第 5 步是硬上限——到了也必须 FINISH。\n\n"
            "# 可用动作\n"
            "- CALL_TOOL: 在沙箱执行 Python 代码。\n"
            "- FINISH: 给出最终答案。action_input: {{\"answer\": \"你的分析结论\"}}\n\n"
            "⚠️ FINISH 时 answer 如果为空或只有过程描述，你的输出将完全无效，"
            "任务视为失败。请务必写出有实质内容的分析。\n"
        )
