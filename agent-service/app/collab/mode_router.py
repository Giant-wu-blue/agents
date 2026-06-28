
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.collab.protocol import CollabMode

logger = logging.getLogger(__name__)


@dataclass
class LoadFeatures:
    """通信负载特征 —— 路由决策的输入。"""
    upstream_count: int          # 上游 agent 数(协作复杂度)
    evidence_count: int          # 检索证据条数
    total_evidence_chars: int    # 证据总字符数(文本传递成本的代理)
    latency_sensitive: bool = False   # 任务是否对延迟敏感


class ModeRouter:
    """根据负载特征路由协作模式。阈值可配置,默认值来自对照实验经验。"""

    def __init__(self,
                 text_char_ceil: int = 600,      # 低于此字符数 → TEXT 够用
                 vector_char_floor: int = 2000,  # 高于此字符数 → VECTOR 收益大
                 vector_evidence_floor: int = 6):# 证据条数多 → VECTOR 筛选收益大
        self.text_char_ceil = text_char_ceil
        self.vector_char_floor = vector_char_floor
        self.vector_evidence_floor = vector_evidence_floor

    def route(self, feat: LoadFeatures) -> tuple[CollabMode, str]:
        """返回(选定模式, 决策理由)。理由用于日志和答辩展示。"""
        if feat.latency_sensitive and feat.total_evidence_chars < self.text_char_ceil:
            return CollabMode.TEXT, "延迟敏感且负载小,用TEXT避免额外开销"

        if (feat.total_evidence_chars >= self.vector_char_floor
                or feat.evidence_count >= self.vector_evidence_floor):
            return CollabMode.VECTOR, (
                f"高负载(证据{feat.evidence_count}条/{feat.total_evidence_chars}字),"
                "用VECTOR向量直传控开销")

        if feat.total_evidence_chars < self.text_char_ceil:
            return CollabMode.TEXT, "低负载,TEXT够用"

        return CollabMode.STRUCTURED, "中等负载,用STRUCTURED结构化降token"

    @staticmethod
    def extract_features(ctx: dict, latency_sensitive: bool = False) -> LoadFeatures:
        """从运行时 ctx 抽取负载特征。"""
        chunks = ctx.get("retrieved_chunks", [])
        total_chars = sum(len(c.get("text", "")) for c in chunks)
        upstream = len({c.get("doc_id", "") for c in chunks})
        return LoadFeatures(
            upstream_count=upstream,
            evidence_count=len(chunks),
            total_evidence_chars=total_chars,
            latency_sensitive=latency_sensitive,
        )
