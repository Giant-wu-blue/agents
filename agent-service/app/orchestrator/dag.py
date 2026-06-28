import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.orchestrator.state_machine import TaskState, is_terminal
from app.collab.protocol import render_upstream, CollabMode
from app.collab.edge_router import EdgeType   # Day3: 按边类型路由

logger = logging.getLogger(__name__)

AgentFn = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class TaskNode:
    task_id: str
    agent: AgentFn
    deps: list[str] = field(default_factory=list)
    timeout: float = 90.0
    max_retries: int = 1
    state: TaskState = TaskState.PENDING
    result: dict[str, Any] | None = None
    error: Exception | None = None
    started_at: float = 0.0   # 真实开始时间(perf_counter)
    ended_at: float = 0.0     # 真实结束时间


class DAGDeadlockError(RuntimeError):
    """Raised when pending tasks remain but none are ready (circular or all failed)."""


class DAGScheduler:
    def __init__(
        self,
        nodes: list[TaskNode],
        global_timeout: float = 240.0,
        max_concurrency: int = 4,
    ):
        self.nodes: dict[str, TaskNode] = {n.task_id: n for n in nodes}
        self.global_timeout = global_timeout
        self.sem = asyncio.Semaphore(max_concurrency)
        self._validate_dag()

    def _validate_dag(self) -> None:
        for tid, node in self.nodes.items():
            for dep in node.deps:
                if dep not in self.nodes:
                    raise ValueError(f"Task '{tid}' depends on unknown task '{dep}'")

    async def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(self._run_dag(ctx), self.global_timeout)
        except asyncio.TimeoutError:
            logger.warning("Global timeout reached, forcing synthesis from partial results")
            return self._force_synthesize()

    async def _run_dag(self, ctx: dict[str, Any]) -> dict[str, Any]:
        pending: set[str] = set(self.nodes.keys())

        while pending:
            ready = [
                tid
                for tid in pending
                if all(
                    self.nodes[d].state in (TaskState.SUCCESS, TaskState.DEGRADED)
                    for d in self.nodes[tid].deps
                )
            ]

            if not ready:
                running = any(
                    self.nodes[tid].state == TaskState.RUNNING for tid in pending
                )
                if not running:
                    raise DAGDeadlockError(
                        f"DAG deadlock: {len(pending)} pending tasks but none ready. "
                        f"Pending: {pending}"
                    )
                await asyncio.sleep(0.05)
                continue

            tasks = [self._run_node(tid, ctx) for tid in ready]
            await asyncio.gather(*tasks)
            pending -= set(ready)

        return self._collect_results()

    async def _run_node(self, task_id: str, ctx: dict[str, Any]) -> None:
        node = self.nodes[task_id]
        node.state = TaskState.READY

        upstream = {d: self.nodes[d].result for d in node.deps}

        if upstream:
            edge_router = ctx.get("edge_router")
            vector_pool = ctx.get("vector_pool")
            # 全局模式作为回退(当 edge_router 不可用时)
            mode_str = ctx.get("collab_mode", "text")
            try:
                fallback_mode = CollabMode(mode_str)
            except ValueError:
                fallback_mode = CollabMode.TEXT

            rendered_parts = []
            stats = ctx.get("comm_stats")
            # 当前协作模式(text / structured / vector),决定 Agent 间通信形态
            mode_str = ctx.get("collab_mode", "structured")
            mode_map = {
                "text": CollabMode.TEXT,
                "structured": CollabMode.STRUCTURED,
                "vector": CollabMode.STRUCTURED,  # vector 的文本部分用结构化(最紧凑)
            }
            cur_mode = mode_map.get(mode_str, CollabMode.STRUCTURED)
            # vector 模式下,证据通过向量池传递,其文本不计入文本通信开销
            count_evidence_as_text = (mode_str != "vector")

            for dep_id, result in upstream.items():
                # 1) 判断这条边的类型
                if edge_router:
                    edge_type = edge_router.classify_edge(dep_id, task_id)
                else:
                    edge_type = EdgeType.PRODUCT  # 无路由时默认按产出传递

                # 2) 按边类型选机制
                if edge_type == EdgeType.EVIDENCE and vector_pool and vector_pool._pool:
                    # EVIDENCE 边:从共享向量池按需精准取用
                    query = ctx.get("topic", "")
                    evidence = await vector_pool.fetch(
                        query, top_k=5, requester=task_id)
                    if evidence:
                        text_lines = [
                            f"[{e['id']}] (score={e['score']:.3f}) {e['text']}"
                            for e in evidence
                        ]
                        evidence_block = (
                            f"【上游 {dep_id} 的检索证据(向量池语义精准取用)】\n"
                            + "\n".join(text_lines)
                        )
                        # 证据始终要喂给 agent 推理(rendered_parts),
                        # 但 vector 模式下它走的是向量直传,不计入文本通信 token。
                        rendered_parts.append((evidence_block, count_evidence_as_text))
                        # 记录向量传输
                        if stats:
                            stats.record_vector_transfer(len(evidence), 1024)
                    else:
                        # 池空,回退当前模式渲染
                        rendered_parts.append(
                            (render_upstream({dep_id: result}, cur_mode), True))
                else:
                    # PRODUCT 边:按当前协作模式渲染
                    #   text       → 完整自然语言(token 多)
                    #   structured → 紧凑结构化协议(token 少)
                    rendered_parts.append(
                        (render_upstream({dep_id: result}, cur_mode), True))

            # 拼接喂给 agent 的完整上文(含所有部分)
            payload = "\n\n".join(p for p, _ in rendered_parts)
            ctx["upstream_rendered"] = payload
            # 仅把"计入文本通信"的部分计入 text_tokens
            # (vector 模式下证据走向量池,不计文本开销)
            if stats:
                billable = "\n\n".join(p for p, billable in rendered_parts if billable)
                if billable:
                    stats.record_text_message(billable)
        else:
            # 无上游依赖的节点,清空渲染缓存,避免串台
            ctx["upstream_rendered"] = ""

        async with self.sem:
            # 注入当前节点的目标描述,供 agent build_system_prompt 动态适配
            agent_goals = ctx.get("agent_goals", {})
            ctx["agent_goal"] = agent_goals.get(task_id, ctx.get("topic", ""))
            import time as _time
            node.started_at = _time.perf_counter()
            for attempt in range(node.max_retries + 1):
                try:
                    node.state = TaskState.RUNNING
                    node.result = await asyncio.wait_for(
                        node.agent(ctx, upstream), node.timeout
                    )
                    node.state = TaskState.SUCCESS
                    node.ended_at = _time.perf_counter()
                    logger.info(f"Task '{task_id}' succeeded (attempt {attempt + 1})")
                    return
                except (asyncio.TimeoutError, Exception) as e:
                    node.error = e
                    node.state = TaskState.FAILED
                    logger.warning(
                        f"Task '{task_id}' failed (attempt {attempt + 1}): {e}"
                    )
                    if attempt < node.max_retries:
                        continue

            node.result = self._degrade(task_id)
            node.state = TaskState.DEGRADED
            node.ended_at = _time.perf_counter()
            logger.info(f"Task '{task_id}' degraded after {node.max_retries + 1} attempts")

    def _degrade(self, task_id: str) -> dict[str, Any]:
        return {"status": "DEGRADED", "task_id": task_id, "data": None}

    def _force_synthesize(self) -> dict[str, Any]:
        return self._collect_results()

    def _collect_results(self) -> dict[str, Any]:
        # 以最早的开始时间为基准,输出每个节点的相对起止(ms),供前端真实回放
        starts = [n.started_at for n in self.nodes.values() if n.started_at > 0]
        t0 = min(starts) if starts else 0.0

        def _summary(node: TaskNode) -> str:
            r = node.result
            if isinstance(r, dict):
                for k in ("content", "summary", "analysis", "report"):
                    v = r.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()[:160]
            return ""

        return {
            tid: {
                "state": node.state.value,
                "result": node.result,
                "error": str(node.error) if node.error else None,
                # 真实执行信息(供前端回放流转)
                "start_ms": round((node.started_at - t0) * 1000, 1) if node.started_at else None,
                "end_ms": round((node.ended_at - t0) * 1000, 1) if node.ended_at else None,
                "elapsed_ms": round((node.ended_at - node.started_at) * 1000, 1) if (node.started_at and node.ended_at) else None,
                "deps": list(node.deps),
                "summary": _summary(node),
            }
            for tid, node in self.nodes.items()
        }
