import re
from typing import Any, Dict, List


# ============================================================
# COMMON WORDS THAT DON'T HELP IDENTIFY THE ACTION
# ============================================================

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
    "customer",
    "supplier",
    "company",
    "party",
    "parties",
    "not",
    "no",
    "their",
    "its",
}


# ============================================================
# GENERIC CLAUSE ACCESS
# ============================================================

def _get(
    clause: Any,
    key: str,
    default=None
):

    if isinstance(
        clause,
        dict
    ):

        return clause.get(
            key,
            default
        )

    return getattr(
        clause,
        key,
        default
    )


# ============================================================
# TOKEN NORMALIZATION
# ============================================================

def _tokens(
    text: str
) -> set:

    tokens = re.findall(
        r"[a-zA-Z]{3,}",
        text.lower()
    )

    normalized = set()


    for token in tokens:

        if token in STOPWORDS:
            continue


        # Lightweight normalization.

        if (
            token.endswith("ies")
            and len(token) > 4
        ):

            token = (
                token[:-3]
                + "y"
            )

        elif (
            token.endswith("ing")
            and len(token) > 5
        ):

            token = token[:-3]

        elif (
            token.endswith("ed")
            and len(token) > 4
        ):

            token = token[:-2]

        elif (
            token.endswith("s")
            and len(token) > 4
        ):

            token = token[:-1]


        normalized.add(
            token
        )


    return normalized


# ============================================================
# EXTRACT TIME PERIODS
# ============================================================

def _extract_temporal(
    text: str
) -> List[Dict[str, Any]]:

    patterns = [

        r"\b"
        r"(\d+(?:\.\d+)?)"
        r"\s*"
        r"(business\s+days?|"
        r"calendar\s+days?|"
        r"days?|"
        r"weeks?|"
        r"months?|"
        r"years?)"
        r"\b",

        r"\b"
        r"(\d+(?:\.\d+)?)"
        r"\s*"
        r"(hours?|minutes?)"
        r"\b",
    ]


    results = []


    for pattern in patterns:

        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE
        ):

            value = float(
                match.group(1)
            )

            unit = re.sub(
                r"\s+",
                " ",
                match.group(2).lower()
            )


            results.append({

                "value": value,

                "unit": unit,

                "text": match.group(0),
            })


    return results


# ============================================================
# TIME UNIT COMPATIBILITY
# ============================================================

def _unit_family(
    unit: str
) -> str:

    unit = unit.lower()


    if (
        "hour" in unit
        or "minute" in unit
    ):

        return "short"


    if "day" in unit:
        return "day"


    if "week" in unit:
        return "week"


    if "month" in unit:
        return "month"


    if "year" in unit:
        return "year"


    return unit


def _compatible_units(
    unit_a: str,
    unit_b: str
) -> bool:

    return (
        _unit_family(unit_a)
        ==
        _unit_family(unit_b)
    )


# ============================================================
# TEMPORAL CONFLICT ANALYSIS
# ============================================================

def analyze_temporal_conflict(
    clause_a: Any,
    clause_b: Any
) -> Dict[str, Any]:

    text_a = str(
        _get(
            clause_a,
            "text",
            ""
        )
        or ""
    )

    text_b = str(
        _get(
            clause_b,
            "text",
            ""
        )
        or ""
    )


    times_a = _extract_temporal(
        text_a
    )

    times_b = _extract_temporal(
        text_b
    )


    party_a = str(
        _get(
            clause_a,
            "party",
            ""
        )
        or ""
    ).lower()


    party_b = str(
        _get(
            clause_b,
            "party",
            ""
        )
        or ""
    ).lower()


    same_party = bool(
        party_a
        and party_b
        and party_a == party_b
    )


    shared_terms = (
        _tokens(text_a)
        &
        _tokens(text_b)
    )


    result = {

        "temporal_conflict_score":
            0.0,

        "conflict_type":
            "no_temporal_conflict",

        "same_party":
            same_party,

        "shared_terms":
            sorted(shared_terms),

        "times_a":
            times_a,

        "times_b":
            times_b,

        "explanation":
            "No temporal inconsistency was detected.",
    }


    # We need temporal expressions in both clauses.
    if not times_a or not times_b:

        return result


    # We also need some common substantive language.
    if not shared_terms:

        return result


    different_pair = None


    for time_a in times_a:

        for time_b in times_b:

            if not _compatible_units(
                time_a["unit"],
                time_b["unit"]
            ):

                continue


            if (
                time_a["value"]
                !=
                time_b["value"]
            ):

                different_pair = (
                    time_a,
                    time_b
                )

                break


        if different_pair:

            break


    if different_pair is None:

        return result


    modality_a = str(
        _get(
            clause_a,
            "modality",
            ""
        )
        or ""
    ).lower()


    modality_b = str(
        _get(
            clause_b,
            "modality",
            ""
        )
        or ""
    ).lower()


    # --------------------------------------------------------
    # Base score
    # --------------------------------------------------------

    score = 0.75


    # Same party strengthens the evidence.
    if same_party:

        score += 0.10


    # Obligation vs permission with different deadlines
    # is particularly important.
    if {
        modality_a,
        modality_b
    } == {
        "obligation",
        "permission"
    }:

        score += 0.10


    # Two obligations with different deadlines
    # are also potentially inconsistent.
    elif (
        modality_a
        ==
        modality_b
        ==
        "obligation"
    ):

        score += 0.05


    score = min(
        score,
        1.0
    )


    time_a, time_b = (
        different_pair
    )


    result.update({

        "temporal_conflict_score":
            score,

        "conflict_type":
            "temporal_deadline_conflict",

        "explanation": (

            "The clauses address related "
            "contractual terms but specify "
            "different time periods "
            f"({time_a['text']} vs "
            f"{time_b['text']})."
        ),
    })


    return result