from __future__ import annotations

import logging
from app.collab.protocol import AgentCapability

logger = logging.getLogger(__name__)


class FunctionRole:
    PLANNER = "planner"        # 规划:拆解任务、决定调度
    RETRIEVER = "retriever"    # 检索:查知识库
    SUMMARIZER = "summarizer"  # 总结:综合上游产出
    EXECUTOR = "executor"      # 执行:调工具/沙箱算数


AGENT_ROLE_MAP = {
    "Planner": FunctionRole.PLANNER,
    "PolicyResearcher": FunctionRole.RETRIEVER,
    "ParcelAnalyst": FunctionRole.EXECUTOR,
    "SupplyPlanner": FunctionRole.EXECUTOR,
    "CostEstimator": FunctionRole.SUMMARIZER,
    "ToolAgent": FunctionRole.EXECUTOR,        # 第三块: 工具使用 agent
}


class CapabilityRegistry:
    """运行时能力注册中心。"""

    def __init__(self):
        self._caps: dict[str, AgentCapability] = {}

    def register(self, cap: AgentCapability) -> None:
        self._caps[cap.agent_name] = cap
        logger.info(f"[registry] 注册 {cap.agent_name} 角色={cap.role} 产出={cap.produces}")

    def register_from_agent(self, agent) -> None:
        """从一个 agent 实例自动生成能力并注册。"""
        role = AGENT_ROLE_MAP.get(agent.name, "unknown")
        produces, consumes = self._infer_io(agent.name)
        cap = AgentCapability(
            agent_name=agent.name,
            role=role,
            actions=list(getattr(agent, "available_actions", [])) + ["FINISH"],
            produces=produces,
            consumes=consumes,
        )
        self.register(cap)

    @staticmethod
    def _infer_io(agent_name: str) -> tuple[list[str], list[str]]:
        """声明每个 agent 产出/消费哪些字段(用于能力发现与协议映射)。"""
        table = {
            "Planner": (["subtasks", "plan"], ["topic"]),
            "PolicyResearcher": (["policy_analysis", "citations"], ["topic"]),
            "ParcelAnalyst": (["parcel_analysis"], ["parcel_id"]),
            "SupplyPlanner": (["supply_analysis"], ["region"]),
            "CostEstimator": (["cost_report"],
                              ["policy_analysis", "parcel_analysis",
                               "supply_analysis", "tool_result"]),
            "ToolAgent": (["tool_result"], ["subtask"]),   # 第三块
        }
        return table.get(agent_name, ([], []))

    def discover(self, needed_field: str) -> list[str]:
        return [n for n, c in self._caps.items() if needed_field in c.produces]

    def by_role(self, role: str) -> list[str]:
        return [n for n, c in self._caps.items() if c.role == role]

    def handshake(self, src: str, dst: str) -> dict:
        s, d = self._caps.get(src), self._caps.get(dst)
        if not s or not d:
            return {"ok": False, "reason": "agent not registered", "matched_fields": []}
        matched = sorted(set(s.produces) & set(d.consumes))
        return {
            "ok": bool(matched),
            "src_role": s.role,
            "dst_role": d.role,
            "matched_fields": matched,
        }

    def snapshot(self) -> dict:
        return {n: c.model_dump() for n, c in self._caps.items()}
