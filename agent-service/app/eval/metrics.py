import re

STATUTE_CITE_RE = re.compile(r"《([^》]+)》第([一二三四五六七八九十百千\d]+)条")


def extract_citations(text: str) -> list[str]:
    """Extract all statute citations like 《XX办法》第N条 from text."""
    return [f"《{m[0]}》第{m[1]}条" for m in STATUTE_CITE_RE.findall(text)]


def sentence_split(text: str) -> list[str]:
    """Split text into sentences/claims for citation coverage analysis."""
    sentences = re.split(r"[。！？；\n]+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def has_citation(sentence: str) -> bool:
    """Check if a sentence contains a chunk_id reference like [chunk_xxx]."""
    return bool(re.search(r"\[chunk[_\-\w]+\]|\[\w{8,}\]", sentence))


# ── Metric functions ──────────────────────────────────────────


def factual_accuracy(report: str, gold_facts: list[str]) -> float:
    """Fraction of gold facts whose keywords all appear in the report."""
    if not gold_facts:
        return 1.0

    hits = 0
    for fact in gold_facts:
        keywords = _keywords_of(fact)
        if all(kw in report for kw in keywords):
            hits += 1

    return hits / len(gold_facts)


def hallucination_rate(report: str, knowledge_base_texts: list[str]) -> float:
    """Fraction of statute citations in the report that can't be found in KB."""
    citations = extract_citations(report)
    if not citations:
        return 0.0

    kb_combined = " ".join(knowledge_base_texts)
    fake = sum(1 for c in citations if c not in kb_combined)
    return fake / len(citations)


def citation_coverage(report: str) -> float:
    """Fraction of sentences that have at least one citation marker."""
    claims = sentence_split(report)
    if not claims:
        return 0.0

    cited = sum(1 for c in claims if has_citation(c))
    return cited / len(claims)


def _keywords_of(fact: str) -> list[str]:
    """Extract key tokens from a gold fact for fuzzy matching."""
    # Simple: split by spaces or use longer tokens
    tokens = re.findall(r"[\w一-鿿]+", fact)
    return [t for t in tokens if len(t) >= 2]
