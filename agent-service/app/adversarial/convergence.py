def is_converged(attack_count: int, threshold: int = 1) -> bool:
    return attack_count <= threshold

def is_oscillating(history: list[int], window: int = 3, tolerance: int = 1) -> bool:
    """
    震荡检测:连续 window 轮攻击数波动 <= tolerance 视为震荡
    例:[5,4,5] 波动=1,认为是震荡;[5,3,1] 波动=4,认为是正常下降
    """
    if len(history) < window:
        return False
    recent = history[-window:]
    return (max(recent) - min(recent)) <= tolerance