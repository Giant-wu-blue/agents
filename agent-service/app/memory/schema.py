from __future__ import annotations

import time
import uuid
from pydantic import BaseModel, Field


class MemoryUnit(BaseModel):
    memory_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    source_agent: str                      # 来源 Agent
    created_at: float = Field(default_factory=time.time)  # 创建时间
    task_topic: str                        # 任务主题
    summary: str                           # 摘要描述
    tags: list[str] = Field(default_factory=list)         # 标签
    payload: dict = Field(default_factory=dict)           # 证据链/结论/策略
    access_scope: list[str] = Field(default_factory=list) # 访问控制:空=公开

    def to_metadata(self) -> dict:
        """ChromaDB 元数据(只能存标量,list 转字符串)。"""
        return {
            "memory_id": self.memory_id,
            "source_agent": self.source_agent,
            "created_at": self.created_at,
            "task_topic": self.task_topic,
            "tags": ",".join(self.tags),
            "access_scope": ",".join(self.access_scope),
        }

    @classmethod
    def from_chroma(cls, doc: str, meta: dict) -> "MemoryUnit":
        return cls(
            memory_id=meta.get("memory_id", ""),
            source_agent=meta.get("source_agent", ""),
            created_at=float(meta.get("created_at", 0)),
            task_topic=meta.get("task_topic", ""),
            summary=doc,
            tags=[t for t in meta.get("tags", "").split(",") if t],
            access_scope=[s for s in meta.get("access_scope", "").split(",") if s],
        )
