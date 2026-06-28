from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from app.adversarial.blue_agent import BlueAgent
from app.adversarial.convergence import is_converged, is_oscillating
from app.adversarial.red_agent import RedAgent
from app.schemas import AdversarialResult

if TYPE_CHECKING:
    from app.clients.llm_client import LLMClient

logger = logging.getLogger(__name__)


class AdversarialLoop:
    """Orchestrates Red-Blue adversarial rounds until convergence, oscillation, or max rounds."""

    def __init__(
        self,
        llm_client: LLMClient,
        max_rounds: int = 4,
        converge_threshold: int = 1,
    ):
        self.max_rounds = max_rounds
        self.converge_threshold = converge_threshold
        self.red = RedAgent(llm_client)
        self.blue = BlueAgent(llm_client)

    async def run(
        self, initial_report: str, evidence_chunks: list[dict]
    ) -> AdversarialResult:
        report = initial_report
        attack_history: list[int] = []
        repair_history: list[dict] = []

        for round_i in range(self.max_rounds):
            t0 = time.perf_counter()

            # 1. Red attack (local 7B)
            attack_report = await self.red.attack(report, evidence_chunks)
            attack_count = len(attack_report.attacks)
            attack_history.append(attack_count)
            logger.info(
                f"[Adv Round {round_i + 1}] attacks={attack_count}, score={attack_report.overall_score}"
            )

            # 2. Converged?
            if is_converged(attack_count, self.converge_threshold):
                return AdversarialResult(
                    final_report=report,
                    rounds_run=round_i + 1,
                    converge_reason="converged",
                    attack_history=attack_history,
                    repair_history=repair_history,
                )

            # 3. Oscillating?
            if is_oscillating(attack_history):
                logger.warning(f"[Adv] oscillation detected: {attack_history[-3:]}")
                return AdversarialResult(
                    final_report=report,
                    rounds_run=round_i + 1,
                    converge_reason="oscillation",
                    attack_history=attack_history,
                    repair_history=repair_history,
                )

            # 4. Blue repair (cloud LLM)
            report = await self.blue.repair(report, attack_report)
            repair_history.append({
                "round": round_i + 1,
                "fixed_attacks": attack_count,
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            })

        # Max rounds reached
        return AdversarialResult(
            final_report=report,
            rounds_run=self.max_rounds,
            converge_reason="max_rounds",
            attack_history=attack_history,
            repair_history=repair_history,
        )
