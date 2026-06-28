import re

CITATION_RE = re.compile(r"《[^》]{2,30}》第[一二三四五六七八九十百零\d]+条")
NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?(?:亩|平方米|万元|元|%|‰|公顷)")


def mark_protected_spans(text: str) -> list[tuple[int, int]]:
    """返回所有受保护片段的 (start, end) 区间"""
    spans = []
    for pattern in (CITATION_RE, NUMERIC_RE):
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def contains_protected(text: str) -> bool:
    return bool(CITATION_RE.search(text) or NUMERIC_RE.search(text))