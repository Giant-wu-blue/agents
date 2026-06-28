import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.eval.metrics import citation_coverage, factual_accuracy, hallucination_rate
from app.eval.judge import judge_report
from app.eval.stats import summarize
from app.schemas import EvalSample

logger = logging.getLogger(__name__)

BENCH_PATH = Path(__file__).parent.parent.parent / "data" / "landresearch_bench.jsonl"


def load_bench(path: Path | None = None) -> list[EvalSample]:
    p = path or BENCH_PATH
    if not p.exists():
        logger.warning(f"Bench file not found: {p}")
        return []

    samples = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(EvalSample(**json.loads(line)))
    return samples


class EvalRunner:
    def __init__(self, llm_client, research_scheduler_cls):
        self.llm_client = llm_client
        self.ResearchScheduler = research_scheduler_cls

    async def run_all(self) -> dict[str, Any]:
        """Run all 5 configs on all samples, produce full comparison matrix."""
        samples = load_bench()
        if not samples:
            return {"error": "No bench samples found"}

        configs = {
            "A_baseline": {"enable_adversarial": False, "enable_compression": False},
            "B_dag": {"enable_adversarial": False, "enable_compression": False},
            "C_adversarial": {"enable_adversarial": True, "enable_compression": False},
            "D_compression": {"enable_adversarial": False, "enable_compression": True},
            "E_full": {"enable_adversarial": True, "enable_compression": True},
        }

        results: dict[str, list[dict]] = {cfg: [] for cfg in configs}

        for sample in samples:
            for cfg_name, cfg_opts in configs.items():
                t0 = time.perf_counter()
                try:
                    from app.schemas import ResearchRequest

                    scheduler = self.ResearchScheduler(self.llm_client)
                    resp = await scheduler.execute(ResearchRequest(
                        topic=sample.research_topic,
                        parcel_id=sample.parcel_id,
                        **cfg_opts,
                    ))

                    # Rule metrics
                    rule = {
                        "factual_accuracy": factual_accuracy(resp.report, sample.gold_facts),
                        "hallucination_rate": hallucination_rate(
                            resp.report,
                            [c for c in sample.gold_citations],
                        ),
                        "citation_coverage": citation_coverage(resp.report),
                    }

                    # LLM Judge
                    judge = await judge_report(
                        resp.report,
                        sample.research_topic,
                        sample.gold_facts,
                        self.llm_client,
                    )

                    results[cfg_name].append({
                        "sample_id": sample.id,
                        "rule_metrics": rule,
                        "judge_scores": judge.model_dump(),
                        "elapsed_ms": (time.perf_counter() - t0) * 1000,
                    })
                except Exception as e:
                    logger.error(f"[{cfg_name}][{sample.id}] Failed: {e}")
                    results[cfg_name].append({
                        "sample_id": sample.id,
                        "error": str(e),
                        "elapsed_ms": (time.perf_counter() - t0) * 1000,
                    })

        # Statistical comparison: E (full) vs A (baseline)
        stats = self._compute_stats(results, "A_baseline", "E_full")

        return {
            "n_samples": len(samples),
            "configs": list(configs.keys()),
            "results": results,
            "statistics": stats,
        }

    def _compute_stats(self, results: dict, baseline: str, treatment: str) -> dict:
        """Compute Bootstrap + Cohen's d between two configs."""
        baseline_scores = np.array([
            r["judge_scores"]["completeness"]
            for r in results.get(baseline, [])
            if "judge_scores" in r
        ])
        treatment_scores = np.array([
            r["judge_scores"]["completeness"]
            for r in results.get(treatment, [])
            if "judge_scores" in r
        ])

        if len(baseline_scores) == 0 or len(treatment_scores) == 0:
            return {"error": "Insufficient data for statistical comparison"}

        # Only use paired samples
        n = min(len(baseline_scores), len(treatment_scores))
        return summarize(
            baseline, baseline_scores[:n],
            treatment, treatment_scores[:n],
        )
