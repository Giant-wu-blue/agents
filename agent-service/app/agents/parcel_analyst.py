"""ParcelAnalyst — 检索并分析地块现状 via ReAct(模板感知版)。

根据 task_template 自动调整分析侧重:
- ZONE_OPTIMIZATION → 侧重多地块横向比较
- PARCEL_FEASIBILITY → 侧重单一地块的储备适宜性
"""

from typing import Any

from app.agents.base import BaseReActAgent


class ParcelAnalyst(BaseReActAgent):
    name = "ParcelAnalyst"
    available_actions = ["SEARCH_KB"]
    max_steps = 5   # 2 次检索 + 1 次分析 + 2 步缓冲

    def __init__(self, java_client=None, llm_client=None):
        super().__init__(java_client=java_client, llm_client=llm_client)

    def build_system_prompt(self, ctx: dict[str, Any], upstream: dict[str, Any]) -> str:
        topic = ctx.get("topic", "")
        parcel_id = ctx.get("parcel_id", "")
        region = ctx.get("region", "")
        agent_goal = ctx.get("agent_goal", "分析地块现状与适宜性")
        template = ctx.get("task_template", "")

        if template == "ZONE_OPTIMIZATION":
            search_hint = (
                "侧重多地块横向比较：各自面积/用途/位置/现状，储备适宜性对比，"
                "组合储备的协同效应和约束条件。\n"
            )
        else:
            search_hint = (
                "侧重目标地块的面积/用途/位置/现状，储备适宜性及理由，"
                "存在的政策或规划限制。\n"
            )

        return (
            "你是地块分析专家。\n\n"
            f"# 研究主题\n{topic}\n"
            f"# 地块 ID\n{parcel_id or '(从主题推断)'}\n"
            f"# 目标区域\n{region or '(从主题推断)'}\n"
            f"# 你的任务目标\n{agent_goal}\n"
            f"# 检索侧重\n{search_hint}\n"
            "# ══ 你的最终产出（最重要，先读）══\n"
            "检索完成后，你必须 FINISH 并输出一份「地块现状分析」。\n"
            "检索只是手段，分析才是目的——你必须用自己的话写，不能复制粘贴检索原文。\n\n"
            "## answer 怎么写（直接按这个结构写）\n\n"
            "【地块属性】\n"
            "从资料提炼：面积、用途、位置、现状等关键信息，用自己的话概括。例如：\n"
            "\"目标地块位于余杭区仁和街道，面积约 21.6 亩，规划为二类工业用地，"
            "现状为集体农用地，需完成征收储备后方可出让。\"\n\n"
            "【储备适宜性】\n"
            "判断是否适合储备并给出理由。例如：\n"
            "\"该地块交通配套完善，容积率 1.5-2.2 符合工业用地标准，适宜储备。\"\n\n"
            "【限制条件】\n"
            "识别风险或限制点。例如：\n"
            "\"现状为农用地，涉及征地补偿和农转用指标审批，周期约 8-12 个月。\"\n\n"
            "每个判断后标注（来源：文档名）。answer 不少于 200 字。\n\n"
            "# ══ 工作流程 ══\n"
            "最多 2 次 SEARCH_KB（换不同关键词），之后必须 FINISH。\n"
            "第 5 步是硬上限——到了也必须 FINISH。\n\n"
            "# 可用动作\n"
            '- SEARCH_KB: 检索知识库。action_input: {{"query": "地块现状 储备适宜性", "top_k": 6, "intent": "parcel"}}\n'
            '- FINISH: 输出分析报告。action_input: {{"answer": "你的分析报告全文"}}\n\n'
            "⚠️ FINISH 时 answer 如果为空或只有一两句过程描述，你的输出将完全无效，"
            "任务视为失败。请务必写出有实质内容的分析。\n"
        )
