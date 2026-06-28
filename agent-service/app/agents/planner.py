from typing import Any

from app.agents.base import BaseReActAgent


class Planner(BaseReActAgent):
    name = "Planner"
    available_actions = []   # 纯规划,无工具
    max_steps = 1            # 一步出规划

    def __init__(self, llm_client=None):
        super().__init__(java_client=None, llm_client=llm_client)

    def build_system_prompt(self, ctx: dict[str, Any], upstream: dict[str, Any]) -> str:
        topic = ctx.get("topic", "")
        return (
            "你是多智能体研究任务的规划专家。\n\n"
            f"# 研究主题\n{topic}\n\n"
            "# 可协作的下游角色\n"
            "- 检索(retriever):查政策法规知识库\n"
            "- 执行(executor):查地块数据、供应计划、做数值计算\n"
            "- 总结(summarizer):综合各方结果做成本测算\n\n"
            "# 任务\n"
            "把研究主题拆解为 3-5 个有序子任务,标注每个子任务应由哪类角色承担。\n"
            "直接用 FINISH 输出,answer 为 JSON 字符串,格式:\n"
            '{"subtasks": [{"step": 1, "role": "retriever", "goal": "检索...的政策"}, ...]}\n'
        )

    async def plan(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """便捷入口:跑一次规划,把 subtasks 写进 ctx 供下游/调度参考。"""
        result = await self.run(ctx, {})
        # 把规划结果挂到 ctx,记忆模块和调度都能读
        import json
        try:
            parsed = json.loads(result.get("content", "{}"))
            ctx["plan"] = parsed
            ctx["subtasks"] = parsed.get("subtasks", [])
        except (json.JSONDecodeError, TypeError):
            ctx["plan"] = {"raw": result.get("content", "")}
            ctx["subtasks"] = []
        return result
