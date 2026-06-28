from __future__ import annotations

import logging
from dataclasses import dataclass

from app.orchestrator.dag import TaskNode

logger = logging.getLogger(__name__)


@dataclass
class AgentSpec:
    task_id: str
    agent_key: str
    deps: list[str]
    timeout: float = 90.0


TASK_TEMPLATES: dict[str, list[AgentSpec]] = {
    "PARCEL_FEASIBILITY": [
        AgentSpec("policy", "policy", []),
        AgentSpec("parcel", "parcel", []),
        AgentSpec("supply", "supply", []),
        AgentSpec("cost", "cost", ["policy", "parcel", "supply"], timeout=100.0),
    ],
    # 纯政策解读:只用政策研究 agent
    "POLICY_INTERPRET": [
        AgentSpec("policy", "policy", []),
    ],
    # 区域供需分析:供应+政策,不需单地块分析
    "REGIONAL_SUPPLY": [
        AgentSpec("policy", "policy", []),
        AgentSpec("supply", "supply", []),
        AgentSpec("summary", "cost", ["policy", "supply"], timeout=100.0),
    ],
    # 类型成本对比:成本+政策(对比维度)
    "COST_COMPARISON": [
        AgentSpec("policy", "policy", []),
        AgentSpec("cost", "cost", ["policy"], timeout=100.0),
    ],
    # 片区组合优化:政策+地块+供应+成本(多地块,本期按完整流程,多地块并行为扩展点)
    "ZONE_OPTIMIZATION": [
        AgentSpec("policy", "policy", []),
        AgentSpec("parcel", "parcel", []),
        AgentSpec("supply", "supply", []),
        AgentSpec("cost", "cost", ["policy", "parcel", "supply"], timeout=100.0),
    ],
}

# 类型的中文名(日志/报告/demo 展示用)
TEMPLATE_LABELS = {
    "PARCEL_FEASIBILITY": "具体地块可行性",
    "POLICY_INTERPRET": "纯政策解读",
    "REGIONAL_SUPPLY": "区域供需分析",
    "COST_COMPARISON": "类型成本对比",
    "ZONE_OPTIMIZATION": "片区组合优化",
}

_CLASSIFY_PROMPT = """你是土地储备研究任务的分类器。判断用户问题属于哪种研究类型，只输出类型代号。

研究类型与判定要点:
- PARCEL_FEASIBILITY（具体地块可行性）: 针对**某个具体地块**评估能否储备、方案、成本。
  例:"评估余杭某工业地块的储备方案与成本"、"XX地块适不适合储备"
- POLICY_INTERPRET（纯政策解读）: **查询、解读、分析政策法规/文件**本身，不针对具体地块、不做供需测算。
  关键信号:出现「政策」「法规」「办法」「意见」「规定」「文件」「通知」「条例」「规划」，
  或问「释放什么信号」「有什么导向」「如何理解」「主要内容」「有哪些要求」。
  例:"工业用地容积率有什么规定"、"《关于推进工商业用地改革的实施意见》释放了什么信号"、
  "解读最新的产业用地准入政策"、"这份文件对土地储备有什么影响"
- REGIONAL_SUPPLY（区域供需分析）: **区域层面的供需数据研判**(供应量、需求量、去化等)，不针对单一地块、不解读某份文件。
  关键信号:出现「供需」「供应量」「需求」「去化」「供给」「形势」且要做**量化研判**。
  例:"余杭区2026年工业用地供需形势如何"、"近三年工业用地供应与需求对比"
- COST_COMPARISON（类型成本对比）: 对比不同用地类型/区域的**成本**。
  例:"工业用地和商业用地储备成本差异"、"对比物流与工业地块的成本"
- ZONE_OPTIMIZATION（片区组合优化）: 一个片区**多个地块**的组合储备方案。
  例:"某片区多宗工业地块的组合储备优化"

判定规则:
1. 问题主体是「一份政策/文件/意见/规划」的解读、信号、导向、影响 → POLICY_INTERPRET（即使提到工商业用地、空间配置等词，只要核心是解读文件，就归此类）。
2. 只有在明确要做**区域供需数据研判**(供应/需求/去化量化)时才选 REGIONAL_SUPPLY。
3. 出现具体地块名/编号、要评估该地块 → PARCEL_FEASIBILITY。

用户问题:{topic}

只输出一个类型代号（PARCEL_FEASIBILITY / POLICY_INTERPRET / REGIONAL_SUPPLY / COST_COMPARISON / ZONE_OPTIMIZATION），不要解释。"""


class DynamicOrchestrator:
    """根据土地研究类型动态编排 agent。"""

    def __init__(self, llm_client, registry=None):
        self.llm = llm_client
        self.registry = registry

    async def classify(self, topic: str) -> str:
        """LLM 判断研究类型，失败则降级到 PARCEL_FEASIBILITY(最全)。
        先用强规则兜底,再用 LLM,提升分类稳定性。"""
        import re
        # 强规则1:含书名号《》且提到政策文件类词 → 纯政策解读
        policy_doc = re.search(r"《.+?》", topic) and re.search(
            r"意见|办法|规定|通知|条例|规划|方案|政策|法规|文件", topic)
        signal_words = re.search(r"释放.*信号|什么信号|有什么导向|如何理解|怎么理解|"
                                 r"主要内容|有哪些要求|解读|传递.*信号", topic)
        if policy_doc or (signal_words and re.search(r"政策|文件|意见|办法|规定|规划", topic)):
            logger.info(f"[orchestrator] 规则命中 → POLICY_INTERPRET: {topic[:30]}")
            return "POLICY_INTERPRET"

        try:
            raw = await self.llm.chat(
                _CLASSIFY_PROMPT.format(topic=topic), temperature=0.0
            )
            for t in TASK_TEMPLATES:
                if t in raw.upper():
                    return t
        except Exception as e:
            logger.warning(f"[orchestrator] 分类失败,降级 PARCEL_FEASIBILITY: {e}")
        return "PARCEL_FEASIBILITY"

    def build_nodes(self, template: str, agent_runners: dict) -> list[TaskNode]:
        """按模板把选中的 agent 组装成 DAG 节点列表。"""
        specs = TASK_TEMPLATES.get(template, TASK_TEMPLATES["PARCEL_FEASIBILITY"])
        nodes = []
        for spec in specs:
            runner = agent_runners.get(spec.agent_key)
            if runner is None:
                logger.warning(f"[orchestrator] 缺少 agent '{spec.agent_key}',跳过节点 {spec.task_id}")
                continue
            nodes.append(TaskNode(
                task_id=spec.task_id, agent=runner, deps=spec.deps, timeout=spec.timeout
            ))
        return nodes

    def validate(self, template: str) -> dict:
        """用注册表校验该模板的 agent 依赖能否被满足(握手)。"""
        specs = TASK_TEMPLATES.get(template, [])
        issues = []
        ids = {s.task_id for s in specs}
        for s in specs:
            for d in s.deps:
                if d not in ids:
                    issues.append(f"{s.task_id} 依赖的 {d} 不在组合中")
        return {"ok": not issues, "issues": issues}

    async def plan(self, topic: str, agent_runners: dict) -> tuple[str, list[TaskNode]]:
        """一站式:分类 → 校验 → 建图。返回(模板名, 节点列表)。"""
        template = await self.classify(topic)
        check = self.validate(template)
        if not check["ok"]:
            logger.warning(f"[orchestrator] 模板 {template} 校验失败 {check['issues']},降级")
            template = "PARCEL_FEASIBILITY"
        nodes = self.build_nodes(template, agent_runners)
        label = TEMPLATE_LABELS.get(template, template)
        logger.info(f"[orchestrator] 研究类型={label}({template}), "
                    f"编排 {len(nodes)} 个 agent: {[n.task_id for n in nodes]}")
        return template, nodes
