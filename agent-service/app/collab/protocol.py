from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CollabMode(str, Enum):
    TEXT = "text"              # 基线:自然语言长文本透传
    STRUCTURED = "structured"  # 本方案:结构化高密度透传
    VECTOR = "vector"          # 进阶:语义证据走向量直传(见 vector_pool / edge_router)


class AgentCapability(BaseModel):
    """能力描述 —— 用于握手/能力发现/协议映射。"""
    agent_name: str
    role: str
    actions: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)  # 产出字段键
    consumes: list[str] = Field(default_factory=list)  # 依赖字段键


class CollabMessage(BaseModel):
    """Agent 间结构化协作消息 —— 替代自然语言长文本透传。"""
    msg_id: str
    src_agent: str
    dst_agent: str
    action: str                                          # 动作类型
    params: dict[str, Any] = Field(default_factory=dict) # 输入参数
    result: dict[str, Any] = Field(default_factory=dict) # 返回结果(结构化)
    capability_ref: str = ""                             # 能力描述引用
    vector_refs: list[str] = Field(default_factory=list) # 关联向量句柄


def upstream_to_text(upstream: dict[str, Any]) -> str:
    """TEXT 模式:把上游每个 agent 的产出渲染成冗长自然语言段落。

    这是"传统纯文本协作"基线 —— 故意保留完整上下文,token 高。
    """
    parts = []
    for dep_id, result in upstream.items():
        if not isinstance(result, dict):
            parts.append(f"【上游 {dep_id} 的完整输出】\n{result}\n")
            continue
        content = result.get("content", "")
        citations = result.get("citations", [])
        scratchpad = result.get("scratchpad", [])
        block = [f"【上游智能体 {dep_id} 的完整分析输出】"]
        block.append(f"分析内容:{content}")
        if citations:
            block.append(f"引用来源:{', '.join(citations)}")
        if scratchpad:
            block.append("推理过程:")
            block.extend(str(s) for s in scratchpad)
        parts.append("\n".join(block) + "\n")
    return "\n".join(parts)


def upstream_to_structured(upstream: dict[str, Any]) -> str:
    """STRUCTURED 模式:只保留高密度结构化字段,丢弃冗余推理过程。

    下游直接解析这些字段,不需要 LLM 重新理解大段文字。
    """
    compact = {}
    for dep_id, result in upstream.items():
        if not isinstance(result, dict):
            compact[dep_id] = {"summary": str(result)[:200]}
            continue
        compact[dep_id] = {
            "conclusion": result.get("content", "")[:300],  # 只保留结论摘要
            "citations": result.get("citations", []),       # 结构化引用
        }
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def render_upstream(upstream: dict[str, Any], mode: CollabMode) -> str:
    if mode == CollabMode.TEXT:
        return upstream_to_text(upstream)
    # STRUCTURED 和 VECTOR 都用结构化文本承载(VECTOR 的向量另走 vector_pool/edge_router)
    return upstream_to_structured(upstream)
