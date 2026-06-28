from __future__ import annotations

from enum import Enum
from app.collab.protocol import CollabMode


class EdgeType(str, Enum):
    EVIDENCE = "evidence"   # 证据分发:语义关系,走向量池
    PRODUCT = "product"     # 产出传递:值依赖,走结构化


# 角色 → 它产出的数据属于哪类
# retriever 产出原始证据;其余角色产出推理结论
_ROLE_OUTPUT = {
    "retriever": EdgeType.EVIDENCE,
    "planner": EdgeType.PRODUCT,
    "executor": EdgeType.PRODUCT,
    "summarizer": EdgeType.PRODUCT,
}


class EdgeRouter:
    """按边类型决定每条 agent 间数据流用什么协作机制。"""

    def __init__(self, registry=None):
        # registry: CapabilityRegistry,用来查 agent 的角色
        self.registry = registry

    def classify_edge(self, src_agent: str, dst_agent: str) -> EdgeType:
        """判断 src→dst 这条边的数据流类型。"""
        role = self._role_of(src_agent)
        return _ROLE_OUTPUT.get(role, EdgeType.PRODUCT)

    def mode_for_edge(self, src_agent: str, dst_agent: str) -> CollabMode:
        """这条边该用哪种协作模式。"""
        etype = self.classify_edge(src_agent, dst_agent)
        return CollabMode.VECTOR if etype == EdgeType.EVIDENCE else CollabMode.STRUCTURED

    def _role_of(self, agent_name: str) -> str:
        """查 agent 的功能角色。优先用注册表,兜底用命名约定。"""
        if self.registry:
            cap = self.registry._caps.get(agent_name)
            if cap:
                return cap.role
        # 兜底:按 task_id/agent 名猜
        name = agent_name.lower()
        if "policy" in name or "retriev" in name:
            return "retriever"
        if "cost" in name or "summ" in name:
            return "summarizer"
        return "executor"

    def route_graph(self, edges: list[tuple[str, str]]) -> dict:
        """对整张 DAG 的所有边批量路由,返回每条边的类型与模式。
        edges: [(src_task_id, dst_task_id), ...]
        用于 demo 展示和文档:一张图里哪些边走向量池、哪些走结构化。
        """
        result = {}
        for src, dst in edges:
            etype = self.classify_edge(src, dst)
            mode = CollabMode.VECTOR if etype == EdgeType.EVIDENCE else CollabMode.STRUCTURED
            result[f"{src}->{dst}"] = {"edge_type": etype.value, "mode": mode.value}
        return result
