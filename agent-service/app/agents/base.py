from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.clients.java_client import JavaClient
    from app.clients.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ReActStep:
    """Record of one ReAct iteration."""

    def __init__(self, thought: str, action: str, action_input: dict, observation: str):
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.observation = observation

    def to_prompt_str(self) -> str:
        return (
            f"Thought: {self.thought}\n"
            f"Action: {self.action}\n"
            f"Action Input: {json.dumps(self.action_input, ensure_ascii=False)}\n"
            f"Observation: {self.observation}\n"
        )


class BaseReActAgent(ABC):
    """Base class with built-in ReAct loop — all specialist agents inherit this."""

    name: str = "base"
    max_steps: int = 3
    available_actions: list[str] = []

    def __init__(self, java_client: JavaClient | None = None, llm_client: LLMClient | None = None):
        self.java_client: JavaClient | None = java_client
        self.llm_client: LLMClient | None = llm_client

    # ── Subclass interface ──────────────────────────────────────

    @abstractmethod
    def build_system_prompt(self, ctx: dict[str, Any], upstream: dict[str, Any]) -> str:
        """Return the system prompt describing the agent's role, goal, and available actions."""

    # ── DAG entry point ─────────────────────────────────────────

    async def run(self, ctx: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
        """Entry point called by DAG scheduler. Delegates to the ReAct loop."""
        return await self(ctx, upstream)

    async def __call__(self, ctx: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
        """两阶段执行：

        阶段一（行动）：ReAct 循环只负责检索/工具调用。每步输出简单 JSON
        （action + action_input），JSON 简单所以解析稳定。当 LLM 决定 FINISH
        或步数用尽，进入阶段二。

        阶段二（分析）：用一次【自由文本】调用让 LLM 基于检索结果写完整分析。
        这一步不做 JSON 解析——整段输出就是分析内容，从而彻底摆脱"长分析塞进
        JSON answer 字段"导致的格式崩溃 → 兜底吐原文问题。
        """
        scratchpad: list[ReActStep] = []
        citations: list[str] = []

        llm = self.llm_client
        if llm is None:
            return {"content": "", "citations": [], "error": "llm_client not injected"}

        # ── 阶段一：行动循环（只做检索/工具，不在这里写分析）──
        for _ in range(self.max_steps):
            prompt = self._build_action_prompt(ctx, upstream, scratchpad)
            llm_output = await llm.chat(prompt, temperature=0.2)
            decision = self._parse_decision(llm_output)

            action = decision.get("action", "")
            if action == "FINISH" or action not in self.available_actions:
                # LLM 认为信息够了，结束行动阶段，进入分析阶段
                break

            observation, new_citations = await self._execute_action(
                action, decision.get("action_input", {}), ctx
            )
            citations.extend(new_citations)
            scratchpad.append(ReActStep(
                thought=decision.get("thought", ""),
                action=action,
                action_input=decision.get("action_input", {}),
                observation=observation[:800],
            ))

        # ── 阶段二：分析（自由文本，不做 JSON 解析）──
        answer = await self._synthesize(ctx, upstream, scratchpad)

        return {
            "content": answer,
            "citations": citations,
            "react_steps": len(scratchpad),
            "scratchpad": [s.to_prompt_str() for s in scratchpad],
        }

    async def _synthesize(
        self, ctx: dict[str, Any], upstream: dict[str, Any], scratchpad: list[ReActStep]
    ) -> str:
        """阶段二：基于检索/工具结果，用自由文本生成最终分析。

        关键：这次调用要求 LLM 直接输出分析正文（不是 JSON），
        所以 LLM 可以放开写几百字，不受格式约束。
        """
        llm = self.llm_client

        # 汇总行动阶段的观察结果作为分析素材
        evidence = "\n".join(
            f"{s.observation}" for s in scratchpad if s.observation
        ) or "(无检索结果，请基于你的专业知识分析)"

        # 复用子类的系统提示（含角色、任务目标、analysis 结构要求）
        sys = self.build_system_prompt(ctx, upstream)
        reused = ctx.get("reused_memory", "")
        if reused:
            sys += f"\n\n# 跨任务记忆复用\n{reused}\n"

        synth_prompt = (
            f"{sys}\n\n"
            f"# 你已检索到的资料\n{evidence}\n\n"
            f"# 现在直接输出你的分析报告\n"
            f"基于上述资料，用你自己的话写出完整的分析（按系统提示里要求的结构）。\n"
            f"直接输出分析正文，不要输出 JSON，不要写'我将开始'之类的过程话，\n"
            f"不要复制粘贴原文。现在开始写分析："
        )
        analysis = await llm.chat(synth_prompt, temperature=0.3)
        analysis = (analysis or "").strip()

        # 极端兜底：万一 LLM 返回空
        if not analysis:
            analysis = self._fallback_answer("", scratchpad)
        return analysis

    # ── Prompt assembly ─────────────────────────────────────────

    def _build_action_prompt(
        self, ctx: dict[str, Any], upstream: dict[str, Any], scratchpad: list[ReActStep]
    ) -> str:
        """阶段一提示：只让 LLM 决定"继续检索"还是"信息够了、结束行动"。
        不在这里要求写分析，所以 JSON 始终简单、解析稳定。
        """
        sys = self.build_system_prompt(ctx, upstream)
        reused = ctx.get("reused_memory", "")
        if reused:
            sys += f"\n\n# 跨任务记忆复用(来自历史相似任务)\n{reused}\n"
        history = "\n".join(s.to_prompt_str() for s in scratchpad)
        actions = self.available_actions + ["FINISH"]

        prompt = (
            f"{sys}\n\n"
            f"# 当前阶段：信息收集\n"
            f"你现在只需决定：继续检索，还是信息已足够可以开始分析。\n"
            f"（分析报告会在下一阶段单独生成，这里【不要】写分析内容。）\n\n"
            f"每步输出严格 JSON:\n"
            f'{{\n  "thought": "你的推理",\n'
            f'  "action": "{" | ".join(actions)}",\n'
            f'  "action_input": {{...}}\n'
            f"}}\n\n"
            f"# 已有检索历史\n{history if history else '(无)'}\n\n"
        )

        if scratchpad:
            last_actions = [s.action for s in scratchpad]
            if "SEARCH_KB" in last_actions or "CALL_TOOL" in last_actions:
                prompt += (
                    "你已检索到资料。如果信息已足够，action 输出 FINISH 进入分析阶段；\n"
                    "如果还需补充，再做一次检索。\n\n"
                )

        prompt += "# 输出你的下一步决策（严格 JSON）:\n"
        return prompt

    # ── Fallback when answer is empty ───────────────────────────

    @staticmethod
    def _fallback_answer(llm_output: str, scratchpad: list) -> str:
        """FINISH 时 answer 为空的兜底，避免报告章节空白。

        策略:
        1) 若 LLM 原始输出含 JSON 外的可读正文，用它;
        2) 否则汇总 ReAct 过程中的观察结果(检索/工具结果)作为内容;
        3) 实在没有就用原始输出截断。
        """
        import re

        text = (llm_output or "").strip()

        # 情况1: 输出不是纯 JSON，直接用(去掉可能的 ```json 包裹)
        cleaned = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        if cleaned and not cleaned.startswith("{"):
            return cleaned[:2000]

        # 情况2: 汇总观察结果
        obs_parts = []
        for s in scratchpad:
            if getattr(s, "observation", "").strip():
                obs_parts.append(f"- {s.observation.strip()[:400]}")
        if obs_parts:
            return ("（注：本节 agent 未输出独立分析，以下为检索到的原始资料，供参考）\n"
                    + "\n".join(obs_parts[:8]))

        # 情况3: 尝试从 JSON 里抠出任意长文本字段
        try:
            import json
            data = json.loads(cleaned)
            if isinstance(data, dict):
                # 找最长的字符串值作为内容
                longest = max(
                    (v for v in data.values() if isinstance(v, str)),
                    key=len, default="",
                )
                if longest.strip():
                    return longest[:2000]
        except Exception:
            pass

        return cleaned[:1000] or "（本节未能生成有效内容）"

    # ── JSON parsing ─────────────────────────────────────────────

    def _parse_decision(self, raw: str) -> dict:
        """Best-effort JSON parse with regex fallback.

        Guarantees the returned dict always has "action" and "answer" keys.
        """
        parsed = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*?\}", raw)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        if isinstance(parsed, dict) and "action" in parsed:
            parsed.setdefault("answer", "")
            return parsed

        logger.warning(f"[{self.name}] parse decision failed, raw={raw[:200]}")
        return {"action": "FINISH", "answer": raw[:500]}

    # ── Action execution ────────────────────────────────────────

    async def _execute_action(
        self, action: str, action_input: dict, ctx: dict[str, Any]
    ) -> tuple[str, list[str]]:
        """Execute an action, returning (observation_str, list_of_citation_ids).

        Also populates ctx["retrieved_chunks"] for downstream adversarial use.
        """
        if not action:
            return "no action specified", []

        java = self.java_client
        if java is None:
            return f"java_client not available for action '{action}'", []

        try:
            # 容错:LLM 有时把 action_input 写成字符串而非字典,
            # 统一规整为 dict,避免 'str' object has no attribute 'get'。
            if isinstance(action_input, str):
                action_input = {"query": action_input}
            elif not isinstance(action_input, dict):
                action_input = {}

            if action == "SEARCH_KB":
                resp = await java.retrieval_search(
                    query=action_input.get("query", "") or ctx.get("topic", ""),
                    intent_node=action_input.get("intent", "general"),
                    top_k=action_input.get("top_k", 5),
                )
                chunks = resp.chunks
                ctx.setdefault("retrieved_chunks", []).extend(
                    {"id": c.id, "text": c.text, "doc_id": c.doc_id} for c in chunks
                )
                pool = ctx.get("vector_pool")
                if pool is not None:
                    await pool.publish(
                        [{"id": c.id, "text": c.text, "doc_id": c.doc_id} for c in chunks],
                        source_agent=self.name,
                    )
                # 在检索结果前嵌入指令:这是原材料,不是最终答案
                obs = (
                    "⚠️ 以上是检索到的原始资料。这些不是你自己的分析——"
                    "你必须在下一步用 FINISH 输出你用自己的话写的分析报告。\n"
                    + "\n".join(f"[{c.id}] (score={c.score:.4f}) {c.text}" for c in chunks)
                )
                return obs, [c.id for c in chunks]

            elif action == "CALL_TOOL":
                resp = await java.mcp_invoke(
                    action_input.get("tool_id", ""),
                    action_input.get("params", {}),
                )
                result = resp.result or str(resp.structured or {})
                ctx.setdefault("retrieved_chunks", []).append(
                    {"id": f"mcp_{action_input.get('tool_id', 'unknown')}", "text": result, "doc_id": "mcp"}
                )
                obs = (
                    "⚠️ 以上是工具执行结果（原始数据）。这些不是你自己的分析——"
                    "你必须在下一步用 FINISH 输出你用自己的话写的分析结论。\n"
                    + result
                )
                return obs, []

            else:
                return f"unknown action: {action}", []

        except Exception as e:
            logger.error(f"[{self.name}] action '{action}' failed: {e}")
            return f"action '{action}' failed: {e}", []