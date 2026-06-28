import logging
import time
import uuid

from app.orchestrator.dag import DAGScheduler
from app.orchestrator.state_machine import TaskState
from app.orchestrator.dynamic_orchestrator import DynamicOrchestrator, TASK_TEMPLATES
from app.agents.planner import Planner
from app.agents.policy_researcher import PolicyResearcher
from app.agents.parcel_analyst import ParcelAnalyst
from app.agents.supply_planner import SupplyPlanner
from app.agents.cost_estimator import CostEstimator
from app.agents.tool_agent import ToolAgent
from app.adversarial.loop import AdversarialLoop
from app.compression.pipeline import aggregate
from app.clients.llm_client import LLMClient
from app.local.provider import LocalProvider
from app.collab.instrument import CommStats
from app.collab.registry import CapabilityRegistry
from app.collab.vector_pool import SharedVectorPool
from app.collab.edge_router import EdgeRouter
from app.collab.mode_router import ModeRouter
from app.memory.schema import MemoryUnit
from app.memory.layered import LayeredMemory
from app.schemas import ResearchRequest, ResearchResponse

logger = logging.getLogger(__name__)


class ResearchScheduler:
    def __init__(
        self,
        java_client=None,
        llm_client: LLMClient | None = None,
    ):
        # 变量名仍叫 java_client(duck typing),底层是全 Python 的 LocalProvider
        self.java_client = java_client or LocalProvider()
        self.llm_client = llm_client or LLMClient()
        self.memory = LayeredMemory(embedder=self.java_client.embedder)

    async def execute(self, request: ResearchRequest) -> ResearchResponse:

        task_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()

        logger.info(f"[{task_id}] Starting research: {request.topic[:80]}...")

        # ── 通信埋点 + 协作模式 ──
        stats = CommStats()
        stats.start()

        ctx = {
            "topic": request.topic,
            "parcel_id": request.parcel_id,
            "region": request.region,
            "retrieved_chunks": [],
            "collab_mode": getattr(request, "collab_mode", "text"),
            "comm_stats": stats,
        }

        # ── 实例化所有 agent（含新增的 Planner 和 ToolAgent）──
        planner_agent = Planner(self.llm_client)
        policy_agent = PolicyResearcher(self.java_client, self.llm_client)
        parcel_agent = ParcelAnalyst(self.java_client, self.llm_client)
        supply_agent = SupplyPlanner(self.java_client, self.llm_client)
        cost_agent = CostEstimator(self.llm_client)
        tool_agent = ToolAgent(self.java_client, self.llm_client)

        all_agents = [planner_agent, policy_agent, parcel_agent,
                      supply_agent, cost_agent, tool_agent]

        # ── 能力注册（握手 / 能力发现 / 协议映射）──
        registry = CapabilityRegistry()
        for ag in all_agents:
            registry.register_from_agent(ag)
        ctx["registry_snapshot"] = registry.snapshot()

        vector_pool = SharedVectorPool(embedder=self.java_client.embedder)
        edge_router = EdgeRouter(registry=registry)
        mode_router = ModeRouter()
        ctx["vector_pool"] = vector_pool
        ctx["edge_router"] = edge_router
        ctx["mode_router"] = mode_router

        # ── 记忆召回（在 agent 实例化之后,确保 reused_memory 能被 agent prompt 读到）──
        hits = await self.memory.recall_semantic(
            request.topic, top_k=3, requester_role="Planner")
        ctx["comm_stats"].record_memory_query(hit=bool(hits))
        if hits:
            ctx["reused_memory"] = "\n".join(
                f"[复用]{h.task_topic}:{h.summary}" for h in hits)

        # ── 协作模式选择 ──
        # auto_route=True:由 ModeRouter 按负载特征自动选模式;
        # auto_route=False:严格使用用户显式指定的 collab_mode(评估对比必需,
        #                   否则三种模式会被 ModeRouter 冲成同一个,token 无差异)。
        if getattr(request, "auto_route", False):
            features = ModeRouter.extract_features(ctx)
            routed_mode, route_reason = mode_router.route(features)
            ctx["collab_mode"] = routed_mode.value
            ctx["route_reason"] = route_reason
        else:
            ctx["collab_mode"] = getattr(request, "collab_mode", "text")
            ctx["route_reason"] = f"显式指定模式={ctx['collab_mode']}（未启用自动路由）"
        logger.info(f"[{task_id}] 协作模式: {ctx['collab_mode']} · {ctx['route_reason']}")

        # ── 动态编排（一次 LLM 调用:分类→校验→建图,替代原 Planner+Orchestrator 两次调用）──
        agent_runners = {
            "policy": policy_agent.run,
            "parcel": parcel_agent.run,
            "supply": supply_agent.run,
            "cost":   cost_agent.run,
            "tool":   tool_agent.run,
        }
        orchestrator = DynamicOrchestrator(self.llm_client, registry=registry)
        template, nodes = await orchestrator.plan(request.topic, agent_runners)
        ctx["task_template"] = template
        # 从模板生成 subtasks（无需额外 LLM 调用,模板已隐含子任务结构）
        ctx["subtasks"] = self._template_to_subtasks(template)
        # 将每个 task_id 对齐到它的 goal,供 agent build_system_prompt 动态适配
        ctx["agent_goals"] = self._agent_goals(template)
        logger.info(f"[{task_id}] 任务类型={template}, 节点={[n.task_id for n in nodes]}")

        # ── Run DAG（节点由动态编排产生，不再写死）──
        dag = DAGScheduler(nodes=nodes, global_timeout=240.0)
        dag_results = await dag.run(ctx)

        # Collect degraded agents
        degraded = [
            tid for tid, r in dag_results.items()
            if r["state"] == TaskState.DEGRADED.value
        ]

        # Assemble draft report（兼容动态节点）
        draft_report = await self._assemble_draft(ctx, dag_results)

        all_evidence = ctx.get("retrieved_chunks", [])

        # ── 三级上下文压缩(前置):法条/数字保护 → 跨Agent去重 →
        #    面向报告二次筛选 → 抽取式压缩(仅超 budget 触发)。
        #    压缩后的精简证据用于后续对抗审查,真正降低 token 开销。
        compression_stats = None
        if request.enable_compression and all_evidence:
            try:
                _ctx_str, comp_stats, kept_chunks = await aggregate(
                    all_evidence, request.topic, self.java_client, budget_tokens=7500,
                )
                compression_stats = comp_stats.to_dict()
                all_evidence = kept_chunks  # 用精简后的证据进入对抗审查
                logger.info(
                    f"[{task_id}] 压缩: {compression_stats['input_chunks']}→"
                    f"{compression_stats['after_refilter']} 块, "
                    f"token {compression_stats['input_tokens']}→"
                    f"{compression_stats['output_tokens']} "
                    f"(省 {compression_stats['saved_pct']}%)"
                )
            except Exception as e:
                logger.warning(f"[{task_id}] 压缩跳过: {e}")

        # Adversarial review(吃压缩后的精简证据)
        adv_rounds, adv_reason = 0, ""
        attack_history: list[int] = []
        final_report = draft_report
        if request.enable_adversarial:
            import time as _t
            adv_loop = AdversarialLoop(
                llm_client=self.llm_client,
                max_rounds=request.max_rounds,
            )
            _adv_t0 = _t.perf_counter()
            adv_result = await adv_loop.run(draft_report, all_evidence)
            _adv_ms = (_t.perf_counter() - _adv_t0) * 1000
            final_report = adv_result.final_report
            adv_rounds = adv_result.rounds_run
            adv_reason = adv_result.converge_reason
            attack_history = adv_result.attack_history

            # 把红蓝对抗过程注入 agent_results,供执行直播显示红/蓝 Agent 节点
            total_attacks = sum(attack_history) if attack_history else 0
            # 让红、蓝节点排在常规 Agent 之后(用较大的 start_ms)
            base_ms = max(
                (r.get("end_ms") or 0) for r in dag_results.values()
            ) if dag_results else 0
            dag_results["red_agent"] = {
                "state": "success",
                "result": {"attack_history": attack_history, "rounds": adv_rounds},
                "error": None,
                "start_ms": base_ms,
                "end_ms": base_ms + _adv_ms * 0.5,
                "elapsed_ms": _adv_ms * 0.5,
                "deps": list(dag_results.keys()),
                "summary": f"红方质疑 {adv_rounds} 轮，累计提出 {total_attacks} 处疑点（事实核查/数字/法条）",
            }
            dag_results["blue_agent"] = {
                "state": "success",
                "result": {"converge_reason": adv_reason, "rounds": adv_rounds},
                "error": None,
                "start_ms": base_ms + _adv_ms * 0.5,
                "end_ms": base_ms + _adv_ms,
                "elapsed_ms": _adv_ms * 0.5,
                "deps": ["red_agent"],
                "summary": f"蓝方据证据修订 {adv_rounds} 轮，最终{('收敛' if adv_reason=='converged' else adv_reason)}",
            }

        # ── 空输出降级:若多 Agent 报告实质为空/全降级,
        #    回退使用基座模型自身能力直接回答原问题,避免给用户"数据暂缺"。
        report_fallback_used = False
        if self._looks_empty(final_report):
            logger.warning(f"[{task_id}] 报告实质为空,回退基座模型直答")
            try:
                base_ans = await self._base_model_answer(request.topic, all_evidence)
                if base_ans.strip():
                    final_report = (
                        f"# 研究报告: {request.topic}\n\n"
                        f"> 说明:多 Agent 协作未能产出充分结果,以下由基座模型"
                        f"结合现有资料直接作答。\n\n{base_ans}"
                    )
                    report_fallback_used = True
            except Exception as e:
                logger.warning(f"[{task_id}] 基座模型兜底失败: {e}")

        # Collect citations
        all_citations: list[str] = []
        for tid, entry in dag_results.items():
            result = entry.get("result", {})
            if isinstance(result, dict):
                for c in result.get("citations", []):
                    if c not in all_citations:
                        all_citations.append(c)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        stats.stop()

        # ── 记忆写入:短期记忆 → 固化(consolidate 内部调用 remember_long 带脱敏) ──
        for tid, r in dag_results.items():
            res = r.get("result", {})
            if isinstance(res, dict) and res.get("content"):
                self.memory.remember_short(MemoryUnit(
                    source_agent=tid, task_topic=request.topic,
                    summary=res["content"][:500], tags=[request.region or "general"]))
        await self.memory.consolidate()

        pool_stats = vector_pool.stats()
        ctx["comm_stats"].record_vector_transfer(
            n_vectors=pool_stats["pool_size"], dim=1024)

        return ResearchResponse(
            task_id=task_id,
            report=final_report,
            agent_results=dag_results,
            citations=all_citations,
            adversarial_rounds=adv_rounds,
            adversarial_reason=adv_reason,
            elapsed_ms=elapsed_ms,
            degraded_agents=degraded,
            metadata={
                "adversarial": {
                    "rounds": adv_rounds,
                    "reason": adv_reason,
                    "attack_history": attack_history,
                },
                "collab": {
                    "mode": ctx.get("collab_mode"),
                    "route_reason": ctx.get("route_reason"),
                    "task_template": ctx.get("task_template"),
                    "subtasks": ctx.get("subtasks", []),
                    "registry": ctx.get("registry_snapshot"),
                    "comm_stats": stats.to_dict(),
                    "vector_pool": pool_stats,
                },
                "compression": compression_stats,
            },
        )

    async def _assemble_draft(self, ctx: dict, dag_results: dict) -> str:
        """兼容动态编排：按实际跑了哪些节点拼报告，标题用 agent 的实际 goal。
        对降级/失败的章节，回退用基座模型自身能力补写该章节内容。"""
        template = ctx.get("task_template", "-")
        topic = ctx["topic"]
        parts = [f"# 研究报告: {topic}（编排模式: {template}）\n\n"]

        # 优先用 agent_goals 中的真实目标作为章节标题,兜底用固定标签
        agent_goals = ctx.get("agent_goals", {})
        fallback_labels = {
            "policy": "政策合规分析",
            "parcel": "地块现状分析",
            "supply": "供应计划分析",
            "cost": "最终分析报告",
            "summary": "综合总结",
            "tool": "工具计算结果",
        }
        evidence = ctx.get("retrieved_chunks", [])

        for tid, r in dag_results.items():
            state = r.get("state", "unknown")
            label = agent_goals.get(tid) or fallback_labels.get(tid, tid)
            parts.append(f"\n## {label}\n")
            if state == "success":
                result = r.get("result", {})
                if isinstance(result, dict):
                    parts.append(result.get("content", str(result)))
                else:
                    parts.append(str(result))
            else:
                # 降级/失败章节:用基座模型结合证据补写,不再显示"数据暂缺"
                try:
                    filled = await self._fill_section(topic, label, evidence)
                    parts.append(filled if filled.strip() else "*(本节暂无足够资料)*\n")
                except Exception as e:
                    logger.warning(f"章节 {tid} 基座兜底失败: {e}")
                    parts.append("*(本节暂无足够资料)*\n")

        return "\n".join(parts)

    async def _fill_section(self, topic: str, section_label: str, evidence: list) -> str:
        """用基座模型为某个失败章节补写内容(结合已检索证据)。"""
        ev = ""
        if evidence:
            snippets = [("- " + (c.get("text", "") if isinstance(c, dict) else str(c))[:200])
                        for c in evidence[:5]]
            snippets = [s for s in snippets if s.strip("- ")]
            if snippets:
                ev = "\n\n可参考的资料:\n" + "\n".join(snippets)
        prompt = (
            f"你是土地储备与评估领域专家。围绕研究主题「{topic}」,"
            f"请就其中的「{section_label}」这一部分,给出专业、简洁的分析(200-400字)。"
            f"若有资料请结合,但不要编造不存在的具体数字或法条。{ev}"
        )
        return await self.llm_client.chat(prompt, temperature=0.5, max_tokens=800)

    @staticmethod
    def _looks_empty(report: str) -> bool:
        """判断报告是否实质为空:去掉标题/降级占位后几乎没有有效内容。"""
        if not report or not report.strip():
            return True
        import re
        body = report
        # 去掉 markdown 标题行
        body = re.sub(r"(?m)^#{1,6}.*$", "", body)
        # 去掉降级/状态占位
        body = body.replace("*(此部分因执行失败已降级，数据暂缺)*", "")
        body = re.sub(r"\*\(状态:.*?\)\*", "", body)
        body = re.sub(r"\s+", "", body)
        # 有效正文少于 40 个字符,视为实质为空
        return len(body) < 40

    async def _base_model_answer(self, topic: str, evidence: list | None = None) -> str:
        """空输出兜底:用基座模型自身能力直接回答(可带上已检索到的证据)。"""
        ev = ""
        if evidence:
            snippets = []
            for c in evidence[:6]:
                t = c.get("text", "") if isinstance(c, dict) else str(c)
                if t:
                    snippets.append("- " + t[:200])
            if snippets:
                ev = "\n\n已检索到的参考资料:\n" + "\n".join(snippets)
        prompt = (
            f"请作为土地储备与评估领域的专家,直接、专业地回答下面的问题。"
            f"若有参考资料请结合,但不要编造不存在的具体数字或法条。\n\n"
            f"问题:{topic}{ev}"
        )
        return await self.llm_client.chat(prompt, temperature=0.5, max_tokens=2000)

    @staticmethod
    def _template_to_subtasks(template: str) -> list[dict]:
        """从编排模板生成子任务清单，无需额外 LLM 调用。

        模板已隐含子任务结构——本方法只是把隐含结构显式化,
        供报告 metadata 展示和下游 agent(如 ToolAgent)参考。
        """
        mapping = {
            "PARCEL_FEASIBILITY": [
                {"step": 1, "role": "retriever", "goal": "检索相关政策法规"},
                {"step": 2, "role": "executor", "goal": "分析地块现状与适宜性"},
                {"step": 3, "role": "executor", "goal": "分析区域供应计划"},
                {"step": 4, "role": "summarizer", "goal": "综合测算储备成本"},
            ],
            "POLICY_INTERPRET": [
                {"step": 1, "role": "retriever", "goal": "检索并解读相关政策法规"},
            ],
            "REGIONAL_SUPPLY": [
                {"step": 1, "role": "retriever", "goal": "检索相关政策法规"},
                {"step": 2, "role": "executor", "goal": "分析区域供应计划"},
                {"step": 3, "role": "summarizer", "goal": "综合供需分析"},
            ],
            "COST_COMPARISON": [
                {"step": 1, "role": "retriever", "goal": "检索相关政策法规与成本基准"},
                {"step": 2, "role": "summarizer", "goal": "多类型成本对比分析"},
            ],
            "ZONE_OPTIMIZATION": [
                {"step": 1, "role": "retriever", "goal": "检索相关政策法规"},
                {"step": 2, "role": "executor", "goal": "分析多地块现状"},
                {"step": 3, "role": "executor", "goal": "分析片区供应计划"},
                {"step": 4, "role": "summarizer", "goal": "综合测算多地块组合方案"},
            ],
        }
    
        return mapping.get(template, mapping["PARCEL_FEASIBILITY"])

    @staticmethod
    def _agent_goals(template: str) -> dict[str, str]:
        """将模板的 AgentSpec 与 subtask 的 goal 按顺序对齐,产出 {task_id: goal}。

        原理: TASK_TEMPLATES 的 specs 和 _template_to_subtasks 的列表同序同长,
        所以按 zip(specs, subtasks) 即可一一对应。
        dag.py 在跑每个节点前从 ctx["agent_goals"] 取出对应 goal,写入 ctx["agent_goal"],
        agent 的 build_system_prompt 通过 ctx["agent_goal"] 感知自己的当前使命。
        """
        from app.orchestrator.dynamic_orchestrator import TASK_TEMPLATES
        specs = TASK_TEMPLATES.get(template, TASK_TEMPLATES["PARCEL_FEASIBILITY"])
        subtasks = ResearchScheduler._template_to_subtasks(template)
        goals: dict[str, str] = {}
        for spec, subtask in zip(specs, subtasks):
            goals[spec.task_id] = subtask["goal"]
        return goals
