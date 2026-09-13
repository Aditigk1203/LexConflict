import re
from typing import Any, Set


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "within",
    "from",
    "on",
    "in",
    "for",
    "by",
    "after",
    "before",
    "upon",
    "all",
    "any",
    "such",
    "that",
    "this",
    "shall",
    "must",
    "may",
    "will",
    "can",
    "should",
    "be",
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "not",
    "no",
    "their",
    "its",
    "company",
    "party",
    "parties",
    "employee",
    "customer",
    "supplier",
}


def _get(clause: Any, key: str, default=None):
    if isinstance(clause, dict):
        return clause.get(key, default)

    return getattr(clause, key, default)


def normalize_tokens(text: str) -> Set[str]:

    tokens = re.findall(
        r"[a-zA-Z]{3,}",
        text.lower(),
    )

    normalized = set()

    for token in tokens:

        if token in STOPWORDS:
            continue

        # Lightweight stemming
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"

        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]

        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]

        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]

        normalized.add(token)

    return normalized


def legal_action_overlap(
    clause_a: Any,
    clause_b: Any,
) -> float:

    text_a = str(
        _get(clause_a, "text", "") or ""
    )

    text_b = str(
        _get(clause_b, "text", "") or ""
    )

    tokens_a = normalize_tokens(text_a)
    tokens_b = normalize_tokens(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b

    if not intersection:
        return 0.0

    # Jaccard similarity
    union = tokens_a | tokens_b

    return len(intersection) / len(union)


def shared_legal_terms(
    clause_a: Any,
    clause_b: Any,
) -> list:

    text_a = str(
        _get(clause_a, "text", "") or ""
    )

    text_b = str(
        _get(clause_b, "text", "") or ""
    )

    shared = (
        normalize_tokens(text_a)
        &
        normalize_tokens(text_b)
    )

    return sorted(shared)