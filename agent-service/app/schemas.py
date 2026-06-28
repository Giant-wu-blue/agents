from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────

class TaskState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DEGRADED = "degraded"


class AgentRole(str, Enum):
    POLICY_RESEARCHER = "policy_researcher"
    PARCEL_ANALYST = "parcel_analyst"
    SUPPLY_PLANNER = "supply_planner"
    COST_ESTIMATOR = "cost_estimator"


class AttackDimension(str, Enum):
    FACTUAL = "FACTUAL"
    LOGICAL = "LOGICAL"
    CITATION = "CITATION"


class RepairAction(str, Enum):
    ADD = "ADD"
    DELETE = "DELETE"
    MODIFY = "MODIFY"
    VERIFY = "VERIFY"


class Domain(str, Enum):
    INDUSTRIAL = "industrial_land_reserve"
    RESIDENTIAL = "residential_land_reserve"
    COMMERCIAL = "commercial_land_reserve"
    PUBLIC = "public_land_reserve"


# ── API models ─────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str
    parcel_id: str | None = None
    region: str | None = None
    max_rounds: int = 4
    enable_adversarial: bool = True
    enable_compression: bool = True
    collab_mode: str = "text"
    auto_route: bool = False

class ResearchResponse(BaseModel):
    task_id: str
    report: str
    agent_results: dict[str, dict[str, Any]]
    citations: list[str] = Field(default_factory=list)
    adversarial_rounds: int = 0
    adversarial_reason: str = ""
    elapsed_ms: float = 0
    degraded_agents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Java client models ─────────────────────────────────────────

class RetrievalRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, alias="topK")
    intent_node: str = Field(default="general", alias="intentNode")


class ChunkItem(BaseModel):
    id: str
    text: str
    score: float
    doc_id: str = Field(alias="docId")


class RetrievalResponse(BaseModel):
    chunks: list[ChunkItem]


class MCPInvokeRequest(BaseModel):
    tool_id: str = Field(alias="toolId")
    params: dict[str, Any] = Field(default_factory=dict)


class MCPInvokeResponse(BaseModel):
    success: bool
    result: str
    structured: dict[str, Any] | None = None


class EmbeddingRequest(BaseModel):
    texts: list[str]


class EmbeddingResponse(BaseModel):
    vectors: list[list[float]]


# ── Adversarial models ─────────────────────────────────────────

class Attack(BaseModel):
    dim: AttackDimension
    claim: str
    issue: str
    suggested_action: RepairAction
    evidence_chunk_id: str | None = None


class AttackResult(BaseModel):
    attacks: list[Attack]
    overall_score: int  # 0-100


class AdversarialReport(BaseModel):
    final_report: str
    rounds: int
    reason: str  # "converged" | "oscillation" | "max_rounds"


class AdversarialResult(BaseModel):
    final_report: str
    rounds_run: int
    converge_reason: str  # "converged" | "oscillation" | "max_rounds"
    attack_history: list[int] = Field(default_factory=list)
    repair_history: list[dict] = Field(default_factory=list)


# ── Eval models ────────────────────────────────────────────────

class EvalSample(BaseModel):
    id: str
    research_topic: str
    parcel_id: str | None = None
    subtasks: list[str]
    gold_facts: list[str]
    gold_citations: list[str]
    complexity: str  # "easy" | "medium" | "hard"
    domain: Domain


class JudgeScore(BaseModel):
    completeness: int
    accuracy: int
    traceability: int
    coherence: int
    actionability: int
    reason: str


class EvalResult(BaseModel):
    sample_id: str
    config: str  # e.g. "full_pipeline" | "baseline"
    rule_metrics: dict[str, float]
    judge_scores: JudgeScore
    elapsed_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)
