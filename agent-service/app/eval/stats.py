import numpy as np
from scipy.stats import bootstrap


def bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean difference (B - A), paired.

    Args:
        scores_a: Baseline scores, shape (n_samples,).
        scores_b: Treatment scores, same shape.
        n_resamples: Bootstrap iterations.
        confidence_level: CI confidence level (default 0.95).

    Returns:
        (ci_low, ci_high) — if both > 0, B is significantly better than A.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Scores arrays must have same length for paired comparison")

    diff = scores_b - scores_a

    res = bootstrap(
        (diff,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
    )

    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d: standardized mean difference (effect size).

    Interpretation:
    - |d| ≈ 0.2: small effect
    - |d| ≈ 0.5: medium effect
    - |d| ≈ 0.8: large effect
    """
    # Pooled standard deviation (assumes equal variance)
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((b.mean() - a.mean()) / pooled_std)


def summarize(
    name_a: str,
    scores_a: np.ndarray,
    name_b: str,
    scores_b: np.ndarray,
) -> dict:
    """Produce a complete statistical summary for a config comparison."""
    ci_low, ci_high = bootstrap_ci(scores_a, scores_b)
    d = cohens_d(scores_a, scores_b)

    magnitude = (
        "large" if abs(d) >= 0.8
        else "medium" if abs(d) >= 0.5
        else "small"
    )

    return {
        "baseline": {"name": name_a, "mean": float(scores_a.mean()), "std": float(scores_a.std(ddof=1))},
        "treatment": {"name": name_b, "mean": float(scores_b.mean()), "std": float(scores_b.std(ddof=1))},
        "mean_diff": float(scores_b.mean() - scores_a.mean()),
        "ci_95": [ci_low, ci_high],
        "cohens_d": d,
        "effect_magnitude": magnitude,
    }
