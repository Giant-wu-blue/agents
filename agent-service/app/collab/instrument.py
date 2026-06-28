from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict


def estimate_tokens(text: str) -> int:
    """粗估 token 数。中文约 1.5 字符/token,英文约 4 字符/token。
    取折中:按字符数 / 1.6。够用于横向对比(两种模式同口径)。
    """
    if not text:
        return 0
    return max(1, int(len(text) / 1.6))


@dataclass
class CommStats:
    message_count: int = 0          # agent 间消息次数
    text_chars: int = 0             # 文本通信字符数
    text_tokens: int = 0            # 文本通信 token 数
    vector_transfers: int = 0       # 非文本传递次数
    vector_bytes: int = 0           # 向量数据规模(字节)
    llm_calls: int = 0              # LLM 调用次数(间接反映开销)
    memory_queries: int = 0         # 记忆查询次数
    memory_hits: int = 0            # 记忆命中次数
    task_elapsed_ms: float = 0.0    # 单任务总耗时
    _t0: float = field(default=0.0, repr=False)

    # ── 计时 ──────────────────────────────────────────────
    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        self.task_elapsed_ms = (time.perf_counter() - self._t0) * 1000

    # ── 埋点 ──────────────────────────────────────────────
    def record_text_message(self, payload: str) -> None:
        """记录一次文本协作消息。"""
        self.message_count += 1
        self.text_chars += len(payload)
        self.text_tokens += estimate_tokens(payload)

    def record_vector_transfer(self, n_vectors: int, dim: int) -> None:
        """记录一次向量传递。float32 = 4 字节/维。"""
        self.message_count += 1
        self.vector_transfers += 1
        self.vector_bytes += n_vectors * dim * 4

    def record_llm_call(self) -> None:
        self.llm_calls += 1

    def record_memory_query(self, hit: bool) -> None:
        self.memory_queries += 1
        if hit:
            self.memory_hits += 1

    @property
    def memory_hit_rate(self) -> float:
        return self.memory_hits / self.memory_queries if self.memory_queries else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_t0", None)
        d["memory_hit_rate"] = round(self.memory_hit_rate, 4)
        return d
